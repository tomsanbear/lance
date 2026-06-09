// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use either::Either;
use futures::TryFutureExt;
use futures::future::FutureExt;
use lance_core::utils::backoff::SlotBackoff;
use lance_core::{Error, Result};

use crate::Dataset;

/// Maximum value, in milliseconds, that the auto-tuned `SlotBackoff::unit`
/// is allowed to take. See the call site for the rationale.
const MAX_AUTO_TUNED_UNIT_MS: u32 = 500;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub retry_timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 10,
            retry_timeout: Duration::from_secs(30),
        }
    }
}

/// Trait for operations that can be retried on commit conflicts
pub trait RetryExecutor: Clone {
    type Data;
    type Result;

    /// Execute the operation logic without committing
    async fn execute_impl(&self) -> Result<Self::Data>;

    /// Commit the operation data
    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result>;

    /// Update the dataset reference for retry attempts
    fn update_dataset(&mut self, dataset: Arc<Dataset>);
}

fn timeout_error(retry_timeout: Duration, attempts: u32) -> Error {
    Error::too_much_write_contention(format!(
        "Attempted {} times, but failed on retry_timeout of {:.3} seconds.",
        attempts,
        retry_timeout.as_secs_f32()
    ))
}

fn maybe_timeout<T>(
    backoff: &SlotBackoff,
    start: Instant,
    retry_timeout: Duration,
    future: impl Future<Output = T>,
) -> impl Future<Output = Result<T>> {
    let attempt = backoff.attempt();
    if attempt == 0 {
        // No timeout on first attempt
        Either::Left(future.map(|res| Ok(res)))
    } else {
        let remaining = retry_timeout.saturating_sub(start.elapsed());
        Either::Right(
            tokio::time::timeout(remaining, future)
                .map_err(move |_| timeout_error(retry_timeout, attempt + 1)),
        )
    }
}

/// Execute an operation with retry logic for commit conflicts
pub async fn execute_with_retry<E: RetryExecutor>(
    executor: E,
    dataset: Arc<Dataset>,
    config: RetryConfig,
) -> Result<E::Result> {
    let start = Instant::now();
    let mut dataset_ref = dataset;
    let mut backoff = SlotBackoff::default();

    while backoff.attempt() <= config.max_retries {
        let mut executor_clone = executor.clone();
        executor_clone.update_dataset(dataset_ref.clone());

        let execute_fut = executor_clone.execute_impl();
        let execute_fut = maybe_timeout(&backoff, start, config.retry_timeout, execute_fut);
        let data = execute_fut.await??;

        let commit_future = executor.commit(dataset_ref.clone(), data);
        let commit_future = maybe_timeout(&backoff, start, config.retry_timeout, commit_future);

        match commit_future.await? {
            Ok(result) => {
                tracing::info!(
                    event.name = "lance.write_retry.completed",
                    attempts = backoff.attempt() + 1,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    outcome = "success",
                    "lance write retry path completed",
                );
                return Ok(result);
            }
            Err(Error::RetryableCommitConflict { .. }) => {
                // Check whether we have exhausted our retries *before* we sleep.
                if backoff.attempt() >= config.max_retries {
                    break;
                }
                if start.elapsed() > config.retry_timeout {
                    tracing::info!(
                        event.name = "lance.write_retry.completed",
                        attempts = backoff.attempt() + 1,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        outcome = "timeout",
                        retry_timeout_ms = config.retry_timeout.as_millis() as u64,
                        "lance write retry path hit retry_timeout",
                    );
                    return Err(timeout_error(config.retry_timeout, backoff.attempt() + 1));
                }
                if backoff.attempt() == 0 {
                    // We add 10% buffer here, to allow concurrent writes to complete.
                    // We pass the first attempt's time to the backoff so it's used
                    // as the unit for backoff time slots.
                    // See SlotBackoff implementation for more details on how this works.
                    //
                    // The slot unit is capped at MAX_AUTO_TUNED_UNIT_MS so that a
                    // first attempt that is slow for unrelated reasons (e.g. a
                    // dataset with hundreds of thousands of accumulated manifest
                    // versions, where the initial manifest read + write itself
                    // takes seconds) does not compound via the exponential slot
                    // doubling into 30+ second retry tails. The cap is roughly
                    // one S3 round-trip — short enough to bound the tail, long
                    // enough that genuinely-contended writers can finish between
                    // attempts.
                    let auto_unit = (start.elapsed().as_millis() * 11 / 10) as u32;
                    backoff = backoff.with_unit(auto_unit.min(MAX_AUTO_TUNED_UNIT_MS));
                }

                let sleep = backoff.next_backoff();
                tracing::info!(
                    event.name = "lance.write_retry.scheduled",
                    attempt = backoff.attempt(),
                    next_attempt = backoff.attempt() + 1,
                    sleep_ms = sleep.as_millis() as u64,
                    "lance write retry CAS conflict; sleeping before retry",
                );
                let sleep_fut = tokio::time::sleep(sleep);
                let sleep_fut = maybe_timeout(&backoff, start, config.retry_timeout, sleep_fut);
                sleep_fut.await?;

                let mut ds = dataset_ref.as_ref().clone();
                ds.checkout_latest().await?;
                dataset_ref = Arc::new(ds);
                continue;
            }
            Err(e) => {
                tracing::info!(
                    event.name = "lance.write_retry.completed",
                    attempts = backoff.attempt() + 1,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    outcome = "error",
                    error = %e,
                    "lance write retry path errored",
                );
                return Err(e);
            }
        }
    }

    tracing::info!(
        event.name = "lance.write_retry.completed",
        attempts = backoff.attempt() + 1,
        elapsed_ms = start.elapsed().as_millis() as u64,
        outcome = "retries_exhausted",
        max_retries = config.max_retries,
        "lance write retry path retries exhausted",
    );
    Err(Error::too_much_write_contention(format!(
        "Attempted {} retries.",
        config.max_retries
    )))
}

#[cfg(test)]
mod tests {
    use super::MAX_AUTO_TUNED_UNIT_MS;

    #[test]
    fn auto_tuned_backoff_unit_capped_at_max() {
        // Pins the per-attempt slot-unit cap on the merge_insert /
        // update / delete retry path. Mirrors the call-site math at
        // `execute_with_retry` above.
        let slow_first_attempt_ms: u128 = 3_000;
        let raw = (slow_first_attempt_ms * 11 / 10) as u32;
        assert_eq!(raw, 3_300, "raw 1.1x scale");
        assert_eq!(
            raw.min(MAX_AUTO_TUNED_UNIT_MS),
            MAX_AUTO_TUNED_UNIT_MS,
            "slow first attempt clamps to the cap",
        );

        let fast_first_attempt_ms: u128 = 200;
        let raw = (fast_first_attempt_ms * 11 / 10) as u32;
        assert_eq!(raw, 220, "raw 1.1x scale");
        assert_eq!(
            raw.min(MAX_AUTO_TUNED_UNIT_MS),
            220,
            "fast first attempt passes through unchanged",
        );
    }
}
