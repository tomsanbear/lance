// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
#![allow(clippy::print_stdout)]

//! Reproduces the body-stream stall leak that the
//! `do_get_with_outer_retry` timeout patch (c6b6d8c4) was written to fix.
//!
//! The harness wires a custom `ObjectStore` whose `get_opts` returns a
//! healthy 200 OK with a body stream that yields one chunk and then blocks
//! forever for a configurable fraction of calls. That reproduces the
//! production failure mode (idle TCP socket after 200 OK) the patch fixes.
//!
//! Run twice with the same workload, varying only the timeout:
//!
//!   LANCE_STREAM_BODY_TIMEOUT_SECS=3600 cargo run --release --example io_leak_repro -- --csv off.csv
//!   LANCE_STREAM_BODY_TIMEOUT_SECS=1    cargo run --release --example io_leak_repro -- --csv on.csv
//!
//! Compare the tail of each CSV. With the fix off, ACTIVE_BATCHES rises and
//! stays elevated through the drain phase. With the fix on, it plateaus low
//! and returns to ~0 during drain.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, BoxStream, StreamExt};
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreRegistry, WrappingObjectStore,
    storage_options::StorageOptionsAccessor,
};
use lance_io::scheduler::{ACTIVE_BATCHES, ACTIVE_IO_TASKS, ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::{
    Attributes, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as OSResult, path::Path,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokio::io::AsyncWriteExt;

/// Return current process resident set size in bytes, or 0 if unavailable.
///
/// macOS: `task_info(MACH_TASK_BASIC_INFO)`. Linux: parse `VmRSS:` from
/// `/proc/self/status`. RSS is allocator-cached, so the signal lags the
/// underlying retention — useful for corroborating `ACTIVE_BATCHES`,
/// not as a primary signal.
fn current_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        #[allow(deprecated)]
        let task = libc::mach_task_self();
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let kr = libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut _,
            &mut count,
        );
        if kr == libc::KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

#[derive(Parser, Debug)]
#[command(about = "Reproduce the do_get_with_outer_retry body-stall leak")]
struct Args {
    /// Output CSV path
    #[arg(long, default_value = "io_leak_repro.csv")]
    csv: String,
    /// Fraction of body streams that hang forever after the 200 OK
    #[arg(long, default_value_t = 0.10)]
    stall_rate: f64,
    /// Total run duration in seconds
    #[arg(long, default_value_t = 60)]
    duration_secs: u64,
    /// Warmup phase length in seconds (submitters spinning up)
    #[arg(long, default_value_t = 5)]
    warmup_secs: u64,
    /// Drain phase length at the end (submitters stopped, observers continue)
    #[arg(long, default_value_t = 5)]
    drain_secs: u64,
    /// Number of concurrent submitter tasks
    #[arg(long, default_value_t = 32)]
    submitters: usize,
    /// Ranges per submit_request batch
    #[arg(long, default_value_t = 10)]
    ranges_per_batch: usize,
    /// Bytes per range
    #[arg(long, default_value_t = 1 << 20)] // 1 MiB
    range_bytes: usize,
    /// Total size of the synthetic object
    #[arg(long, default_value_t = 64 << 20)] // 64 MiB
    object_size: usize,
    /// Sleep between batches per submitter (millis)
    #[arg(long, default_value_t = 200)]
    submitter_sleep_ms: u64,
    /// `download_retry_count` to thread through (0 makes a stall fail in one
    /// timeout window instead of (retries+1) windows; load-bearing for short runs)
    #[arg(long, default_value_t = 0)]
    download_retry_count: u64,
}

/// Wraps an inner `ObjectStore` and, for `stall_rate` fraction of `get_opts`
/// calls, returns a 200 OK whose body stream emits one trivial chunk and then
/// blocks forever. That is the exact failure mode the timeout patch addresses.
#[derive(Debug)]
struct StallingStore {
    inner: Arc<dyn OSObjectStore>,
    stall_rate: f64,
    counter: AtomicU64,
}

impl StallingStore {
    fn new(inner: Arc<dyn OSObjectStore>, stall_rate: f64) -> Self {
        Self {
            inner,
            stall_rate,
            counter: AtomicU64::new(0),
        }
    }

    fn should_stall(&self) -> bool {
        if self.stall_rate <= 0.0 {
            return false;
        }
        // Deterministic-ish: use a counter + xorshift so we don't depend on
        // RNG state at construction time. We only need rough proportionality.
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut x = n.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        let p = (x as f64) / (u64::MAX as f64);
        p < self.stall_rate
    }
}

impl std::fmt::Display for StallingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StallingStore(rate={})", self.stall_rate)
    }
}

#[async_trait::async_trait]
impl OSObjectStore for StallingStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        // Head requests need to succeed so the scheduler can resolve size.
        if options.head {
            return self.inner.get_opts(location, options).await;
        }
        if !self.should_stall() {
            // Non-stall path: read the inner body, then re-wrap as a fresh
            // owned allocation. InMemory normally returns zero-copy refcount
            // slices of the source buffer, which masks the heap-byte leak —
            // siblings would not be holding their own allocations. Production
            // cloud reads allocate fresh per request; mimic that here so the
            // sibling-pin part of the diagnosis shows up in RSS.
            let inner = self.inner.get_opts(location, options.clone()).await?;
            let meta = inner.meta.clone();
            let range = inner.range.clone();
            let attributes = inner.attributes.clone();
            let bytes = inner.bytes().await?;
            let owned = Bytes::copy_from_slice(&bytes);
            let stream: BoxStream<'static, OSResult<Bytes>> =
                stream::iter(vec![Ok(owned)]).boxed();
            return Ok(GetResult {
                payload: GetResultPayload::Stream(stream),
                meta,
                range,
                attributes,
            });
        }
        // Stall path: emit TWO small chunks before pending(). With one chunk,
        // `collect_bytes` (object_store/util.rs:52) returns zero-copy and
        // never enters its `Vec::with_capacity(size_hint)` branch — so the
        // stalled future would only retain a few bytes, hiding the leak.
        // Yielding two chunks forces collect_bytes into the multi-chunk
        // branch, which allocates a `Vec<u8>` of `size_hint` bytes (= the
        // requested range size, ~1 MiB) and then awaits the next chunk,
        // which never comes. That Vec is the partially-filled body buffer
        // the diagnosis pins to `Vec::with_capacity<u8>` in heap profiles.
        let mut head_opts = options.clone();
        head_opts.head = true;
        let head = self.inner.get_opts(location, head_opts).await?;
        let range = head.range.clone();
        let stream: BoxStream<'static, OSResult<Bytes>> = stream::iter(vec![
            Ok(Bytes::from_static(&[0u8; 1])),
            Ok(Bytes::from_static(&[0u8; 1])),
        ])
        .chain(stream::pending())
        .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream),
            meta: head.meta,
            range,
            attributes: Attributes::default(),
        })
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.inner.delete(location).await
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

#[derive(Debug)]
struct StallingWrapper {
    stall_rate: f64,
}

impl WrappingObjectStore for StallingWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OSObjectStore>,
    ) -> Arc<dyn OSObjectStore> {
        Arc::new(StallingStore::new(original, self.stall_rate))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    println!(
        "io_leak_repro: stall_rate={:.2} duration={}s submitters={} ranges/batch={} range_bytes={} object_size={} download_retry_count={}",
        args.stall_rate,
        args.duration_secs,
        args.submitters,
        args.ranges_per_batch,
        args.range_bytes,
        args.object_size,
        args.download_retry_count,
    );
    println!(
        "LANCE_STREAM_BODY_TIMEOUT_SECS={}",
        std::env::var("LANCE_STREAM_BODY_TIMEOUT_SECS").unwrap_or_else(|_| "<unset>".into())
    );

    let stall_rate = args.stall_rate;
    let storage_opts: std::collections::HashMap<String, String> = [(
        "download_retry_count".to_string(),
        args.download_retry_count.to_string(),
    )]
    .into_iter()
    .collect();
    let params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(StallingWrapper { stall_rate })),
        storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
            storage_opts,
        ))),
        ..Default::default()
    };
    let registry = Arc::new(ObjectStoreRegistry::default());
    let (object_store, base_path) =
        ObjectStore::from_uri_and_params(registry, "memory:///leak", &params).await?;
    println!("object_store ready: scheme={} path={}", object_store.scheme(), base_path);

    // Populate one synthetic object with zeros.
    let object_path = base_path.child("blob");
    {
        let payload = vec![0u8; args.object_size];
        let mut writer = object_store.create(&object_path).await?;
        writer.write_all(&payload).await?;
        writer.shutdown().await?;
    }
    println!("wrote {} bytes to {}", args.object_size, object_path);

    let scheduler_cfg = SchedulerConfig::new(32 << 20);
    let scheduler = ScanScheduler::new(object_store, scheduler_cfg);
    let known_size = CachedFileSize::default();
    let file_sched = scheduler.open_file(&object_path, &known_size).await?;

    // Sampler — every 1s, write {t_secs, active_batches, active_io_tasks}.
    let csv_path = args.csv.clone();
    let stop_flag = Arc::new(AtomicU64::new(0));
    let stop_flag_sampler = stop_flag.clone();
    let sampler = tokio::spawn(async move {
        use tokio::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&csv_path)
            .await
            .expect("open csv");
        file.write_all(b"t_secs,active_batches,active_io_tasks,rss_bytes\n")
            .await
            .expect("write csv header");
        let start = Instant::now();
        let mut tick = tokio::time::interval(Duration::from_millis(1000));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let elapsed = start.elapsed().as_secs_f64();
            let batches = ACTIVE_BATCHES.load(Ordering::Acquire);
            let tasks = ACTIVE_IO_TASKS.load(Ordering::Acquire);
            let rss = current_rss_bytes();
            let line = format!("{elapsed:.3},{batches},{tasks},{rss}\n");
            if file.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            // Echo to stdout so progress is visible.
            print!("{line}");
            if stop_flag_sampler.load(Ordering::Acquire) != 0 {
                break;
            }
        }
        let _ = file.shutdown().await;
    });

    // Submitter tasks.
    let sustained_until = Duration::from_secs(args.duration_secs - args.drain_secs);
    let warmup = Duration::from_secs(args.warmup_secs);
    let start = Instant::now();
    let mut submitter_handles = Vec::with_capacity(args.submitters);
    let object_size = args.object_size as u64;
    let range_bytes = args.range_bytes as u64;
    let ranges_per_batch = args.ranges_per_batch;
    let submitter_sleep = Duration::from_millis(args.submitter_sleep_ms);
    for s_idx in 0..args.submitters {
        let file_sched = file_sched.clone();
        let h = tokio::spawn(async move {
            // Stagger warmup so submitters don't all hit at t=0.
            let stagger = warmup.mul_f64(s_idx as f64 / args.submitters as f64);
            tokio::time::sleep(stagger).await;
            let mut rng = StdRng::seed_from_u64(0xdeadbeef + s_idx as u64);
            loop {
                if start.elapsed() >= sustained_until {
                    break;
                }
                // FileScheduler::submit_request assumes ranges are sorted
                // (its coalesce pass is a single linear merge). Random
                // unsorted ranges trigger underflow in the post-fetch
                // re-slice step. Sort here.
                let mut ranges = Vec::with_capacity(ranges_per_batch);
                for _ in 0..ranges_per_batch {
                    let max_start = object_size.saturating_sub(range_bytes);
                    let start_byte = if max_start == 0 {
                        0
                    } else {
                        rng.random_range(0..max_start)
                    };
                    ranges.push(start_byte..start_byte + range_bytes);
                }
                ranges.sort_by_key(|r| r.start);
                // Ignore individual batch errors — stalls produce errors and
                // that's expected; the leak is about retention, not correctness.
                let _ = file_sched.submit_request(ranges, 0).await;
                tokio::time::sleep(submitter_sleep).await;
            }
        });
        submitter_handles.push(h);
    }

    // Wait for sustained phase + drain to complete.
    let total = Duration::from_secs(args.duration_secs);
    while start.elapsed() < total {
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // Submitters may be parked inside a stalled submit_request (fix off →
    // body never errors). Abort rather than await — we've already captured
    // their contribution via ACTIVE_* counters.
    for h in &submitter_handles {
        h.abort();
    }
    for h in submitter_handles {
        let _ = h.await;
    }
    // Give the sampler one final tick after the deadline before stopping.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    stop_flag.store(1, Ordering::Release);
    let _ = sampler.await;

    let final_batches = ACTIVE_BATCHES.load(Ordering::Acquire);
    let final_tasks = ACTIVE_IO_TASKS.load(Ordering::Acquire);
    println!(
        "done. final ACTIVE_BATCHES={} ACTIVE_IO_TASKS={} csv={}",
        final_batches, final_tasks, args.csv
    );
    Ok(())
}
