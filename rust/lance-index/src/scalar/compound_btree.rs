// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compound (multi-column) B-tree scalar index implementation.
//!
//! This module provides the training and storage infrastructure for compound
//! indices that enable efficient lookups on multi-column predicates like
//! `WHERE tenant_id = X AND status = Y AND timestamp > T`.
//!
//! # Architecture
//!
//! Compound indices use a similar page-based structure to single-column BTree indices:
//!
//! - `compound_page_data.lance`: Contains indexed rows [col1, col2, ..., colN, row_id]
//! - `compound_page_lookup.lance`: Contains per-page, per-column statistics for pruning
//!
//! # Key Differences from Single-Column BTree
//!
//! 1. Multiple value columns instead of one
//! 2. Per-column min/max/null_count statistics (enables skip-scan, non-prefix pruning)
//! 3. Uses Arrow Row Format for compound key comparison
//! 4. Separate file names to avoid confusion

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray, new_empty_array, types::UInt64Type, Array, ArrayRef, RecordBatch, UInt32Array,
    UInt64Array,
};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use lance_core::{Error, Result, ROW_ID};
use lance_datafusion::chunker::chunk_concat_stream;
use snafu::location;

use super::compound::{CompoundIndexSchema, COMPOUND_SORT_OPTIONS};
use super::{IndexStore, IndexWriter, ScalarIndex};

// ============================================================================
// Constants
// ============================================================================

/// File name for compound page data (the actual indexed rows).
pub const COMPOUND_PAGES_NAME: &str = "compound_page_data.lance";

/// File name for compound page lookup (per-page statistics).
pub const COMPOUND_LOOKUP_NAME: &str = "compound_page_lookup.lance";

/// Default batch size for compound index pages.
pub const DEFAULT_COMPOUND_BATCH_SIZE: u64 = 4096;

/// Metadata key for batch size in the lookup file schema.
pub const COMPOUND_BATCH_SIZE_META_KEY: &str = "batch_size";

/// Column name for row IDs in the compound index.
pub const COMPOUND_IDS_COLUMN: &str = "_rowid";

// ============================================================================
// CompoundBTreeSubIndex Trait
// ============================================================================

/// Trait for compound B-tree subindex implementations.
///
/// This is analogous to `BTreeSubIndex` but designed for multi-column indices.
/// It handles batches with multiple value columns plus a row ID column.
///
/// # Schema
///
/// The expected schema for training batches is:
/// ```text
/// [value_col_0, value_col_1, ..., value_col_N-1, _rowid]
/// ```
///
/// # Differences from BTreeSubIndex
///
/// - Operates on multiple value columns
/// - Statistics are per-column (min, max, null_count for each)
/// - Uses separate file names (compound_page_*)
#[async_trait]
pub trait CompoundBTreeSubIndex: Debug + Send + Sync + DeepSizeOf {
    /// Trains the subindex on a single batch of compound data and serializes it to Arrow.
    ///
    /// The input batch should contain value columns followed by the row ID column.
    /// Returns the trained batch in the format expected for storage.
    async fn train(&self, batch: RecordBatch) -> Result<RecordBatch>;

    /// Deserialize a subindex from Arrow.
    ///
    /// Note: This is deferred to Milestone 3 (search implementation).
    async fn load_subindex(&self, serialized: RecordBatch) -> Result<Arc<dyn ScalarIndex>>;

    /// Retrieve the data used to originally train this page.
    ///
    /// Used during index updates to merge old and new data.
    async fn retrieve_data(&self, serialized: RecordBatch) -> Result<RecordBatch>;

    /// The schema of the subindex when serialized to Arrow.
    ///
    /// Format: [value_col_0, value_col_1, ..., value_col_N-1, _rowid]
    fn schema(&self) -> &Arc<Schema>;

    /// Given a serialized page, remap the row IDs and re-serialize.
    ///
    /// Used during compaction to update row addresses.
    /// Note: Full implementation deferred to Milestone 4.
    async fn remap_subindex(
        &self,
        serialized: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch>;
}

// ============================================================================
// CompoundFlatIndexMetadata
// ============================================================================

/// Metadata and training implementation for flat compound subindex.
///
/// This stores compound index pages as flat Arrow record batches with
/// the schema: [col_0, col_1, ..., col_N-1, _rowid].
///
/// Unlike FlatIndexMetadata for single-column indices, this handles
/// multiple value columns.
#[derive(Debug)]
pub struct CompoundFlatIndexMetadata {
    /// Schema for stored pages.
    schema: Arc<Schema>,
    /// Number of value columns (excludes _rowid).
    num_columns: usize,
    /// Original column names from the dataset.
    column_names: Vec<String>,
}

impl DeepSizeOf for CompoundFlatIndexMetadata {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.schema.metadata.deep_size_of_children(context)
            + self
                .schema
                .fields
                .iter()
                .map(|f| {
                    std::mem::size_of::<Field>()
                        + f.name().deep_size_of_children(context)
                        + f.metadata().deep_size_of_children(context)
                })
                .sum::<usize>()
            + self
                .column_names
                .iter()
                .map(|n| n.deep_size_of_children(context))
                .sum::<usize>()
    }
}

impl CompoundFlatIndexMetadata {
    /// Create a new CompoundFlatIndexMetadata.
    ///
    /// # Arguments
    ///
    /// * `column_names` - Names of the value columns in index order
    /// * `data_types` - Data types for each value column
    ///
    /// # Schema
    ///
    /// The resulting schema will be:
    /// ```text
    /// [column_names[0]: data_types[0], ..., column_names[N-1]: data_types[N-1], _rowid: UInt64]
    /// ```
    pub fn new(column_names: Vec<String>, data_types: Vec<DataType>) -> Self {
        assert_eq!(
            column_names.len(),
            data_types.len(),
            "Column names and data types must have the same length"
        );

        let mut fields: Vec<Field> = column_names
            .iter()
            .zip(data_types.iter())
            .map(|(name, dt)| Field::new(name, dt.clone(), true))
            .collect();
        fields.push(Field::new(COMPOUND_IDS_COLUMN, DataType::UInt64, false));

        Self {
            schema: Arc::new(Schema::new(fields)),
            num_columns: column_names.len(),
            column_names,
        }
    }

    /// Get the number of value columns.
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Get the column names.
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }
}

#[async_trait]
impl CompoundBTreeSubIndex for CompoundFlatIndexMetadata {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    async fn train(&self, batch: RecordBatch) -> Result<RecordBatch> {
        // Extract columns by position - input may have generic names (col0, col1, ...)
        // or original names, but the order is always: [value_cols..., _rowid]
        let mut columns = Vec::with_capacity(self.num_columns + 1);

        // Take value columns by position
        for i in 0..self.num_columns {
            if i >= batch.num_columns() {
                return Err(Error::Index {
                    message: format!(
                        "Training batch has {} columns, expected at least {}",
                        batch.num_columns(),
                        self.num_columns + 1
                    ),
                    location: location!(),
                });
            }
            columns.push(batch.column(i).clone());
        }

        // Add row ID column (last column)
        let row_id_idx = batch.num_columns() - 1;
        columns.push(batch.column(row_id_idx).clone());

        // Output with original column names from self.schema
        Ok(RecordBatch::try_new(self.schema.clone(), columns)?)
    }

    async fn load_subindex(&self, _serialized: RecordBatch) -> Result<Arc<dyn ScalarIndex>> {
        // Deferred to Milestone 3 - search implementation
        Err(Error::NotSupported {
            source: "Compound index loading not yet implemented (planned for M3)".into(),
            location: location!(),
        })
    }

    async fn retrieve_data(&self, serialized: RecordBatch) -> Result<RecordBatch> {
        // The flat storage format preserves the original data
        Ok(serialized)
    }

    async fn remap_subindex(
        &self,
        serialized: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch> {
        // Get the row ID column (last column)
        let row_id_col_idx = serialized.num_columns() - 1;
        let row_ids = serialized.column(row_id_col_idx).as_primitive::<UInt64Type>();

        // Filter and remap row IDs
        let val_idx_and_new_id: Vec<(usize, u64)> = row_ids
            .values()
            .iter()
            .enumerate()
            .filter_map(|(idx, old_id)| {
                mapping
                    .get(old_id)
                    .copied()
                    .unwrap_or(Some(*old_id))
                    .map(|new_id| (idx, new_id))
            })
            .collect();

        // Create new row IDs array
        let new_ids = Arc::new(UInt64Array::from_iter_values(
            val_idx_and_new_id.iter().copied().map(|(_, new_id)| new_id),
        ));

        // Create indices for taking from value columns
        let take_indices = UInt64Array::from_iter_values(
            val_idx_and_new_id
                .into_iter()
                .map(|(val_idx, _)| val_idx as u64),
        );

        // Take from all value columns and build new batch
        let mut new_columns = Vec::with_capacity(serialized.num_columns());
        for i in 0..row_id_col_idx {
            let new_col = arrow_select::take::take(serialized.column(i), &take_indices, None)?;
            new_columns.push(new_col);
        }
        new_columns.push(new_ids);

        Ok(RecordBatch::try_new(serialized.schema(), new_columns)?)
    }
}

// ============================================================================
// Per-Column Statistics
// ============================================================================

/// Statistics for a single column within a page.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Minimum value in this column for this page (first row, since data is sorted).
    pub min: ScalarValue,
    /// Maximum value in this column for this page (last row).
    pub max: ScalarValue,
    /// Number of null values in this column for this page.
    pub null_count: u32,
}

/// Statistics for all columns in a compound batch/page.
#[derive(Debug)]
pub struct CompoundBatchStats {
    /// Per-column statistics.
    pub column_stats: Vec<ColumnStats>,
    /// Page number (0-indexed).
    pub page_number: u32,
}

// ============================================================================
// CompoundBTreeLookup - In-Memory Page Routing
// ============================================================================

/// Statistics for a single page in the compound index.
///
/// Contains per-column min/max/null_count statistics that enable
/// efficient page pruning for queries.
#[derive(Debug, Clone)]
pub struct CompoundPageStats {
    /// Minimum value per column.
    pub mins: Vec<ScalarValue>,
    /// Maximum value per column.
    pub maxs: Vec<ScalarValue>,
    /// Null count per column.
    pub null_counts: Vec<u32>,
    /// Page number (0-indexed).
    pub page_number: u32,
}

impl DeepSizeOf for CompoundPageStats {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.mins.iter().map(std::mem::size_of_val).sum::<usize>()
            + self.maxs.iter().map(std::mem::size_of_val).sum::<usize>()
            + self.null_counts.deep_size_of_children(context)
    }
}

/// In-memory lookup structure for compound index pages.
///
/// This structure provides efficient page routing based on per-column
/// statistics. Unlike single-column BTreeLookup which uses a BTreeMap,
/// this stores per-page statistics and performs linear pruning across
/// pages using per-column bounds.
///
/// # Pruning Strategy
///
/// For each query, pages are pruned if any column predicate guarantees
/// no rows can match:
/// - Equality: prune if value < min OR value > max
/// - Range: prune if range doesn't overlap [min, max]
/// - IS NULL: prune if null_count = 0
///
/// This enables pruning even for non-prefix queries (e.g., timestamp > T
/// without specifying tenant_id).
#[derive(Debug)]
pub struct CompoundBTreeLookup {
    /// Per-column statistics for each page.
    page_stats: Vec<CompoundPageStats>,
    /// Number of columns in the index.
    num_columns: usize,
    /// Column data types (extracted from lookup schema).
    data_types: Vec<DataType>,
}

impl DeepSizeOf for CompoundBTreeLookup {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.page_stats.deep_size_of_children(context)
    }
}

impl CompoundBTreeLookup {
    /// Create a new CompoundBTreeLookup from parsed page statistics.
    pub fn new(page_stats: Vec<CompoundPageStats>, data_types: Vec<DataType>) -> Self {
        let num_columns = data_types.len();
        Self {
            page_stats,
            num_columns,
            data_types,
        }
    }

    /// Parse a CompoundBTreeLookup from the serialized lookup batch.
    ///
    /// The lookup batch has the schema:
    /// ```text
    /// [min_col0, max_col0, null_count_col0,
    ///  min_col1, max_col1, null_count_col1,
    ///  ...,
    ///  page_idx]
    /// ```
    ///
    /// Data types are extracted from the schema (min_col* columns).
    pub fn try_from_serialized(
        lookup_batch: RecordBatch,
        column_names: &[String],
    ) -> Result<Self> {
        let schema = lookup_batch.schema();
        let num_columns = column_names.len();

        // Extract data types from the min_* columns
        let data_types: Vec<DataType> = (0..num_columns)
            .map(|i| {
                let field_idx = i * 3; // min_col0, max_col0, null_count_col0, min_col1, ...
                schema.field(field_idx).data_type().clone()
            })
            .collect();

        if lookup_batch.num_rows() == 0 {
            return Ok(Self::new(vec![], data_types));
        }

        let mut page_stats = Vec::with_capacity(lookup_batch.num_rows());

        // Get the page_idx column (last column)
        let page_idx_col = lookup_batch
            .column(lookup_batch.num_columns() - 1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Index {
                message: "page_idx column is not UInt32".to_string(),
                location: location!(),
            })?;

        for row_idx in 0..lookup_batch.num_rows() {
            let mut mins = Vec::with_capacity(num_columns);
            let mut maxs = Vec::with_capacity(num_columns);
            let mut null_counts = Vec::with_capacity(num_columns);

            for (col_idx, col_name) in column_names.iter().enumerate() {
                let base_idx = col_idx * 3;

                // min_col
                let min_col = lookup_batch.column(base_idx);
                let min_val = ScalarValue::try_from_array(min_col, row_idx).map_err(|e| {
                    Error::Index {
                        message: format!(
                            "Failed to read min value for column {}: {}",
                            col_name, e
                        ),
                        location: location!(),
                    }
                })?;
                mins.push(min_val);

                // max_col
                let max_col = lookup_batch.column(base_idx + 1);
                let max_val = ScalarValue::try_from_array(max_col, row_idx).map_err(|e| {
                    Error::Index {
                        message: format!(
                            "Failed to read max value for column {}: {}",
                            col_name, e
                        ),
                        location: location!(),
                    }
                })?;
                maxs.push(max_val);

                // null_count_col
                let null_count_col = lookup_batch
                    .column(base_idx + 2)
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| Error::Index {
                        message: format!(
                            "null_count column for {} is not UInt32",
                            col_name
                        ),
                        location: location!(),
                    })?;
                null_counts.push(null_count_col.value(row_idx));
            }

            let page_number = page_idx_col.value(row_idx);

            page_stats.push(CompoundPageStats {
                mins,
                maxs,
                null_counts,
                page_number,
            });
        }

        Ok(Self::new(page_stats, data_types))
    }

    /// Get the number of pages in this lookup.
    pub fn num_pages(&self) -> usize {
        self.page_stats.len()
    }

    /// Get the data types for the indexed columns.
    pub fn data_types(&self) -> &[DataType] {
        &self.data_types
    }

    /// Find all pages that may contain rows matching the query.
    ///
    /// Returns page numbers that cannot be pruned based on per-column statistics.
    pub fn find_candidate_pages(&self, query: &super::compound::CompoundSargableQuery) -> Vec<u32> {
        self.page_stats
            .iter()
            .filter(|stats| !self.can_prune_page(stats, query))
            .map(|stats| stats.page_number)
            .collect()
    }

    /// Check if a page can be pruned based on query predicates.
    ///
    /// Returns true if the page definitely cannot contain matching rows.
    fn can_prune_page(
        &self,
        stats: &CompoundPageStats,
        query: &super::compound::CompoundSargableQuery,
    ) -> bool {
        use super::compound::CompoundSargableQuery;

        match query {
            CompoundSargableQuery::FullKeyLookup(key) => {
                // For full key lookup, we can't easily compare compound keys to per-column stats
                // without the RowConverter. For now, don't prune based on full key.
                // The per-column bounds check would require deconstructing the key.
                // This is conservative but correct - we may load more pages than necessary.
                let _ = key; // unused for now
                false
            }
            CompoundSargableQuery::PrefixLookup { prefix, range } => {
                // Check each prefix column for pruning
                for (col_idx, value) in prefix.iter().enumerate() {
                    if self.can_prune_by_equality(stats, col_idx, value) {
                        return true;
                    }
                }

                // Check range on next column if present
                if let Some((lower, upper)) = range {
                    let range_col_idx = prefix.len();
                    if range_col_idx < self.num_columns
                        && self.can_prune_by_range(stats, range_col_idx, lower, upper)
                    {
                        return true;
                    }
                }

                false
            }
            CompoundSargableQuery::Range { lower, upper } => {
                // Range on compound keys is harder to prune with per-column stats.
                // We could check the first column bounds as an approximation.
                // For now, be conservative and don't prune.
                let _ = (lower, upper);
                false
            }
        }
    }

    /// Check if a page can be pruned based on an equality predicate on a column.
    fn can_prune_by_equality(
        &self,
        stats: &CompoundPageStats,
        col_idx: usize,
        value: &ScalarValue,
    ) -> bool {
        if col_idx >= self.num_columns {
            return false;
        }

        // Handle NULL values
        if value.is_null() {
            // Looking for NULL - prune if no nulls in this column
            return stats.null_counts[col_idx] == 0;
        }

        // If the page is entirely NULL for this column, prune (looking for non-NULL value)
        if stats.mins[col_idx].is_null() && stats.maxs[col_idx].is_null() {
            return true;
        }

        // Check if value is outside [min, max] range
        // value < min OR value > max -> prune
        if !stats.mins[col_idx].is_null() {
            if let Some(ordering) = value.partial_cmp(&stats.mins[col_idx]) {
                if ordering == std::cmp::Ordering::Less {
                    return true; // value < min
                }
            }
        }

        if !stats.maxs[col_idx].is_null() {
            if let Some(ordering) = value.partial_cmp(&stats.maxs[col_idx]) {
                if ordering == std::cmp::Ordering::Greater {
                    return true; // value > max
                }
            }
        }

        false
    }

    /// Check if a page can be pruned based on a range predicate on a column.
    fn can_prune_by_range(
        &self,
        stats: &CompoundPageStats,
        col_idx: usize,
        lower: &std::ops::Bound<ScalarValue>,
        upper: &std::ops::Bound<ScalarValue>,
    ) -> bool {
        use std::ops::Bound;

        if col_idx >= self.num_columns {
            return false;
        }

        // If the page is entirely NULL for this column, prune (range doesn't match NULL)
        if stats.mins[col_idx].is_null() && stats.maxs[col_idx].is_null() {
            return true;
        }

        // Check if range is completely below page min
        // upper < min (exclusive) or upper <= min (inclusive with upper < min)
        if !stats.mins[col_idx].is_null() {
            match upper {
                Bound::Included(val) => {
                    if let Some(ordering) = val.partial_cmp(&stats.mins[col_idx]) {
                        if ordering == std::cmp::Ordering::Less {
                            return true; // upper < min
                        }
                    }
                }
                Bound::Excluded(val) => {
                    if let Some(ordering) = val.partial_cmp(&stats.mins[col_idx]) {
                        if ordering != std::cmp::Ordering::Greater {
                            return true; // upper <= min
                        }
                    }
                }
                Bound::Unbounded => {}
            }
        }

        // Check if range is completely above page max
        // lower > max (exclusive) or lower >= max (inclusive with lower > max)
        if !stats.maxs[col_idx].is_null() {
            match lower {
                Bound::Included(val) => {
                    if let Some(ordering) = val.partial_cmp(&stats.maxs[col_idx]) {
                        if ordering == std::cmp::Ordering::Greater {
                            return true; // lower > max
                        }
                    }
                }
                Bound::Excluded(val) => {
                    if let Some(ordering) = val.partial_cmp(&stats.maxs[col_idx]) {
                        if ordering != std::cmp::Ordering::Less {
                            return true; // lower >= max
                        }
                    }
                }
                Bound::Unbounded => {}
            }
        }

        false
    }

    /// Get pages that may contain NULL values in the specified column.
    pub fn pages_with_nulls(&self, col_idx: usize) -> Vec<u32> {
        self.page_stats
            .iter()
            .filter(|stats| col_idx < stats.null_counts.len() && stats.null_counts[col_idx] > 0)
            .map(|stats| stats.page_number)
            .collect()
    }
}

/// Analyze a compound batch to extract per-column statistics.
///
/// # Arguments
///
/// * `batch` - The record batch to analyze
/// * `column_names` - Names of value columns (excluding _rowid)
///
/// # Assumptions
///
/// The batch is assumed to be sorted by the compound key. Therefore:
/// - min is the first row's value
/// - max is the last row's value
///
/// # Returns
///
/// Statistics for each column in the batch.
fn analyze_compound_batch(batch: &RecordBatch, num_value_columns: usize) -> Result<Vec<ColumnStats>> {
    if batch.num_rows() == 0 {
        return Err(Error::Internal {
            message: "Received an empty batch in compound btree training".to_string(),
            location: location!(),
        });
    }

    let mut stats = Vec::with_capacity(num_value_columns);

    for i in 0..num_value_columns {
        let col = batch.column(i);

        // For sorted data: min is first row, max is last row
        let min = ScalarValue::try_from_array(col, 0).map_err(|e| Error::Internal {
            message: format!("Failed to get min value for column {}: {}", i, e),
            location: location!(),
        })?;

        let max = ScalarValue::try_from_array(col, col.len() - 1).map_err(|e| Error::Internal {
            message: format!("Failed to get max value for column {}: {}", i, e),
            location: location!(),
        })?;

        stats.push(ColumnStats {
            min,
            max,
            null_count: col.null_count() as u32,
        });
    }

    Ok(stats)
}

/// Encoded batch result from training a single page.
struct EncodedCompoundBatch {
    stats: Vec<ColumnStats>,
    page_number: u32,
}

/// Train a single compound page.
async fn train_compound_page(
    batch: RecordBatch,
    batch_idx: u32,
    num_value_columns: usize,
    sub_index_trainer: &dyn CompoundBTreeSubIndex,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedCompoundBatch> {
    let stats = analyze_compound_batch(&batch, num_value_columns)?;
    let trained = sub_index_trainer.train(batch).await?;
    writer.write_record_batch(trained).await?;
    Ok(EncodedCompoundBatch {
        stats,
        page_number: batch_idx,
    })
}

// ============================================================================
// Lookup File Generation
// ============================================================================

/// Convert per-column statistics to a lookup record batch.
///
/// # Schema
///
/// For N columns, the schema is:
/// ```text
/// [min_col0, max_col0, null_count_col0,
///  min_col1, max_col1, null_count_col1,
///  ...,
///  min_colN-1, max_colN-1, null_count_colN-1,
///  page_idx]
/// ```
///
/// This enables per-column pruning during query planning.
fn compound_stats_as_batch(
    stats: Vec<EncodedCompoundBatch>,
    column_names: &[String],
    data_types: &[DataType],
) -> Result<RecordBatch> {
    if stats.is_empty() {
        // Create empty schema with the expected structure
        let mut fields = Vec::new();
        for (i, dt) in data_types.iter().enumerate() {
            fields.push(Field::new(format!("min_{}", column_names[i]), dt.clone(), true));
            fields.push(Field::new(format!("max_{}", column_names[i]), dt.clone(), true));
            fields.push(Field::new(
                format!("null_count_{}", column_names[i]),
                DataType::UInt32,
                false,
            ));
        }
        fields.push(Field::new("page_idx", DataType::UInt32, false));

        let schema = Arc::new(Schema::new(fields));
        return Ok(RecordBatch::new_empty(schema));
    }

    let num_columns = column_names.len();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(num_columns * 3 + 1);

    // For each column, create min, max, null_count arrays
    for (col_idx, dt) in data_types.iter().enumerate() {
        let mins: ArrayRef = if stats.is_empty() {
            new_empty_array(dt)
        } else {
            ScalarValue::iter_to_array(stats.iter().map(|s| s.stats[col_idx].min.clone()))?
        };

        let maxs: ArrayRef = if stats.is_empty() {
            new_empty_array(dt)
        } else {
            ScalarValue::iter_to_array(stats.iter().map(|s| s.stats[col_idx].max.clone()))?
        };

        let null_counts: ArrayRef = Arc::new(UInt32Array::from_iter_values(
            stats.iter().map(|s| s.stats[col_idx].null_count),
        ));

        columns.push(mins);
        columns.push(maxs);
        columns.push(null_counts);
    }

    // Add page index column
    let page_numbers: ArrayRef =
        Arc::new(UInt32Array::from_iter_values(stats.iter().map(|s| s.page_number)));
    columns.push(page_numbers);

    // Build schema
    let mut fields = Vec::new();
    for (i, _dt) in data_types.iter().enumerate() {
        // min and max can be null if entire page is null
        fields.push(Field::new(
            format!("min_{}", column_names[i]),
            columns[i * 3].data_type().clone(),
            true,
        ));
        fields.push(Field::new(
            format!("max_{}", column_names[i]),
            columns[i * 3 + 1].data_type().clone(),
            true,
        ));
        fields.push(Field::new(
            format!("null_count_{}", column_names[i]),
            DataType::UInt32,
            false,
        ));
    }
    fields.push(Field::new("page_idx", DataType::UInt32, false));

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, columns)?)
}

// ============================================================================
// Training Function
// ============================================================================

/// Train a compound BTree index from a stream of sorted batches.
///
/// # Arguments
///
/// * `batches_source` - Stream of record batches containing value columns + row_id.
///   **Must be pre-sorted** by compound key (all value columns in index order).
/// * `sub_index_trainer` - The subindex implementation to use for training pages.
/// * `index_store` - Storage backend for writing index files.
/// * `compound_schema` - Schema defining the compound index structure.
/// * `batch_size` - Number of rows per page.
/// * `fragment_ids` - Optional fragment IDs for distributed indexing.
///
/// # Output Files
///
/// Creates two files in the index store:
/// - `compound_page_data.lance`: Contains the actual indexed rows
/// - `compound_page_lookup.lance`: Contains per-page statistics for query pruning
///
/// # Example
///
/// ```ignore
/// let schema = CompoundIndexSchema::new(
///     vec!["tenant_id".to_string(), "timestamp".to_string()],
///     vec![DataType::Utf8, DataType::Int64],
/// )?;
///
/// train_compound_btree_index(
///     sorted_data_stream,
///     &CompoundFlatIndexMetadata::new(schema.columns().to_vec(), schema.data_types().to_vec()),
///     &index_store,
///     &schema,
///     4096,
///     None,
/// ).await?;
/// ```
pub async fn train_compound_btree_index(
    batches_source: SendableRecordBatchStream,
    sub_index_trainer: &dyn CompoundBTreeSubIndex,
    index_store: &dyn IndexStore,
    compound_schema: &CompoundIndexSchema,
    batch_size: u64,
    fragment_ids: Option<Vec<u32>>,
) -> Result<()> {
    // Create fragment mask for distributed indexing (matches btree.rs pattern)
    let fragment_mask = fragment_ids.as_ref().and_then(|frag_ids| {
        if !frag_ids.is_empty() {
            Some((frag_ids[0] as u64) << 32)
        } else {
            None
        }
    });

    // Determine file names based on whether this is a full or partial index
    let (page_data_name, page_lookup_name) = if fragment_mask.is_none() {
        (COMPOUND_PAGES_NAME.to_string(), COMPOUND_LOOKUP_NAME.to_string())
    } else {
        (
            format!("part_{}_compound_page_data.lance", fragment_mask.unwrap()),
            format!("part_{}_compound_page_lookup.lance", fragment_mask.unwrap()),
        )
    };

    // Create page data file
    let mut page_data_file = index_store
        .new_index_file(&page_data_name, sub_index_trainer.schema().clone())
        .await?;

    let column_names = compound_schema.columns().to_vec();
    let data_types = compound_schema.data_types().to_vec();

    let mut encoded_batches = Vec::new();
    let mut batch_idx = 0u32;

    // Chunk the input stream into page-sized batches
    let mut batches_source = chunk_concat_stream(batches_source, batch_size as usize);

    while let Some(batch) = batches_source.try_next().await? {
        let encoded = train_compound_page(
            batch,
            batch_idx,
            column_names.len(),
            sub_index_trainer,
            page_data_file.as_mut(),
        )
        .await?;
        encoded_batches.push(encoded);
        batch_idx += 1;
    }

    page_data_file.finish().await?;

    // Create lookup file with per-column statistics
    let lookup_batch = compound_stats_as_batch(encoded_batches, &column_names, &data_types)?;

    let mut file_schema = lookup_batch.schema().as_ref().clone();
    file_schema.metadata.insert(
        COMPOUND_BATCH_SIZE_META_KEY.to_string(),
        batch_size.to_string(),
    );

    let mut lookup_file = index_store
        .new_index_file(&page_lookup_name, Arc::new(file_schema))
        .await?;

    lookup_file.write_record_batch(lookup_batch).await?;
    lookup_file.finish().await?;

    Ok(())
}

// ============================================================================
// Row Converter Helper
// ============================================================================

/// Create a RowConverter for compound key comparison.
///
/// Uses NULLS FIRST, ASC ordering to match Lance's existing behavior.
pub fn create_compound_row_converter(data_types: &[DataType]) -> Result<RowConverter> {
    let fields: Vec<SortField> = data_types
        .iter()
        .map(|dt| SortField::new_with_options(dt.clone(), COMPOUND_SORT_OPTIONS))
        .collect();

    RowConverter::new(fields).map_err(|e| Error::Index {
        message: format!("Failed to create compound RowConverter: {}", e),
        location: location!(),
    })
}

// ============================================================================
// CompoundBTreeIndex - Main Index Structure
// ============================================================================

use crate::frag_reuse::FragReuseIndex;
use crate::metrics::{MetricsCollector, NoOpMetricsCollector};
use crate::pb;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::registry::{ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest};
use crate::scalar::{AnyQuery, CreatedIndex, IndexReader, SearchResult, UpdateCriteria};
use crate::Index;
use arrow_schema::SortOptions;
use datafusion::physical_plan::{
    sorts::sort_preserving_merge::SortPreservingMergeExec, stream::RecordBatchStreamAdapter,
    union::UnionExec, ExecutionPlan,
};
use datafusion_common::DataFusionError;
use datafusion_physical_expr::{expressions::Column, PhysicalSortExpr};
use futures::stream::{self, StreamExt};
use lance_core::cache::{LanceCache, WeakLanceCache};
use lance_core::utils::mask::RowAddrTreeMap;
use lance_datafusion::exec::{execute_plan, LanceExecutionOptions, OneShotExec};
use roaring::RoaringBitmap;
use std::any::Any;
use tracing::debug;

use super::compound::CompoundSargableQuery;

/// Lazy index reader for compound index pages.
/// 
/// Only opens the file reader if/when needed (e.g., if pages aren't cached).
#[derive(Clone)]
struct LazyCompoundIndexReader {
    index_reader: Arc<tokio::sync::Mutex<Option<Arc<dyn IndexReader>>>>,
    store: Arc<dyn IndexStore>,
}

impl LazyCompoundIndexReader {
    fn new(store: Arc<dyn IndexStore>) -> Self {
        Self {
            index_reader: Arc::new(tokio::sync::Mutex::new(None)),
            store,
        }
    }

    async fn get(&self) -> Result<Arc<dyn IndexReader>> {
        let mut reader = self.index_reader.lock().await;
        if reader.is_none() {
            let index_reader = self.store.open_index_file(COMPOUND_PAGES_NAME).await?;
            *reader = Some(index_reader);
        }
        Ok(reader.as_ref().unwrap().clone())
    }
}

/// Cache key for compound index pages.
#[derive(Debug, Clone)]
pub struct CompoundBTreePageKey {
    pub page_number: u32,
}

impl lance_core::cache::CacheKey for CompoundBTreePageKey {
    type ValueType = CachedCompoundPage;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("compound-page-{}", self.page_number).into()
    }
}

/// Cached compound index page data.
#[derive(Debug, Clone)]
pub struct CachedCompoundPage(RecordBatch);

impl DeepSizeOf for CachedCompoundPage {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Approximate size based on batch
        self.0.num_rows() * self.0.num_columns() * 8
    }
}

impl CachedCompoundPage {
    pub fn new(batch: RecordBatch) -> Self {
        Self(batch)
    }

    pub fn into_inner(self) -> RecordBatch {
        self.0
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.0
    }
}

/// Compound B-tree index for multi-column queries.
///
/// This index enables efficient lookups on predicates like:
/// - `WHERE tenant_id = 'acme' AND status = 'active'` (prefix lookup)
/// - `WHERE tenant_id = 'acme' AND timestamp > '2024-01-01'` (prefix + range)
/// - `WHERE tenant_id = 'acme'` (partial prefix)
///
/// # Architecture
///
/// Similar to single-column BTreeIndex but with:
/// - Multiple value columns per page
/// - Per-column min/max/null_count statistics for pruning
/// - Arrow Row Format for compound key comparison
#[derive(Clone, Debug)]
pub struct CompoundBTreeIndex {
    /// Column names in index order.
    columns: Vec<String>,
    /// Column data types.
    data_types: Vec<DataType>,
    /// Page lookup structure with per-column statistics.
    page_lookup: Arc<CompoundBTreeLookup>,
    /// Cache for loaded pages.
    index_cache: WeakLanceCache,
    /// Storage backend.
    store: Arc<dyn IndexStore>,
    /// Subindex metadata for loading pages.
    sub_index: Arc<dyn CompoundBTreeSubIndex>,
    /// Rows per page.
    batch_size: u64,
    /// Fragment reuse index for row ID remapping.
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
}

impl DeepSizeOf for CompoundBTreeIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.page_lookup.deep_size_of_children(context)
            + self.store.deep_size_of_children(context)
    }
}

impl CompoundBTreeIndex {
    /// Load a compound index from storage.
    ///
    /// # Arguments
    ///
    /// * `store` - Storage backend containing index files
    /// * `column_names` - Column names in index order
    /// * `frag_reuse_index` - Optional fragment reuse index for row ID remapping
    /// * `index_cache` - Cache for loaded pages
    pub async fn load(
        store: Arc<dyn IndexStore>,
        column_names: Vec<String>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        // Load the lookup file
        let page_lookup_file = store.open_index_file(COMPOUND_LOOKUP_NAME).await?;
        let num_rows = page_lookup_file.num_rows();
        let serialized_lookup = page_lookup_file.read_range(0..num_rows, None).await?;

        // Extract batch size from schema metadata
        let file_schema = page_lookup_file.schema();
        let batch_size = file_schema
            .metadata
            .get(COMPOUND_BATCH_SIZE_META_KEY)
            .map(|bs| bs.parse().unwrap_or(DEFAULT_COMPOUND_BATCH_SIZE))
            .unwrap_or(DEFAULT_COMPOUND_BATCH_SIZE);

        // Build the lookup structure (extracts data types from schema)
        let page_lookup = CompoundBTreeLookup::try_from_serialized(serialized_lookup, &column_names)?;
        let data_types = page_lookup.data_types().to_vec();

        // Create sub_index metadata
        let sub_index = Arc::new(CompoundFlatIndexMetadata::new(
            column_names.clone(),
            data_types.clone(),
        ));

        Ok(Arc::new(Self {
            columns: column_names,
            data_types,
            page_lookup: Arc::new(page_lookup),
            index_cache: WeakLanceCache::from(index_cache),
            store,
            sub_index,
            batch_size,
            frag_reuse_index,
        }))
    }

    /// Get the column names in this index.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Get the data types for indexed columns.
    pub fn data_types(&self) -> &[DataType] {
        &self.data_types
    }

    /// Get the number of pages in this index.
    pub fn num_pages(&self) -> usize {
        self.page_lookup.num_pages()
    }

    /// Look up a page, using cache if available.
    async fn lookup_page(
        &self,
        page_number: u32,
        index_reader: LazyCompoundIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        self.index_cache
            .get_or_insert_with_key(CompoundBTreePageKey { page_number }, move || async move {
                let result = self.read_page(page_number, index_reader, metrics).await?;
                Ok(CachedCompoundPage::new(result))
            })
            .await
            .map(|v| v.as_ref().clone().into_inner())
    }

    /// Read a page from storage.
    async fn read_page(
        &self,
        page_number: u32,
        index_reader: LazyCompoundIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        metrics.record_part_load();
        let reader = index_reader.get().await?;
        let mut batch = reader
            .read_record_batch(page_number as u64, self.batch_size)
            .await?;

        // Apply fragment reuse remapping if present
        if let Some(fri) = &self.frag_reuse_index {
            batch = fri.remap_row_ids_record_batch(batch, self.columns.len())?;
        }

        Ok(batch)
    }

    /// Search a single page for matching rows.
    async fn search_page(
        &self,
        query: &CompoundSargableQuery,
        page_number: u32,
        index_reader: LazyCompoundIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowAddrTreeMap> {
        let page_batch = self.lookup_page(page_number, index_reader, metrics).await?;
        self.search_batch(&page_batch, query)
    }

    /// Search a batch for rows matching the query.
    fn search_batch(
        &self,
        batch: &RecordBatch,
        query: &CompoundSargableQuery,
    ) -> Result<RowAddrTreeMap> {
        match query {
            CompoundSargableQuery::FullKeyLookup(key) => {
                self.search_full_key(batch, key)
            }
            CompoundSargableQuery::PrefixLookup { prefix, range } => {
                self.search_prefix(batch, prefix, range.as_ref())
            }
            CompoundSargableQuery::Range { lower, upper } => {
                self.search_range(batch, lower, upper)
            }
        }
    }

    /// Search for an exact full key match.
    fn search_full_key(
        &self,
        batch: &RecordBatch,
        key: &super::compound::CompoundKey,
    ) -> Result<RowAddrTreeMap> {
        // Create row converter for comparison
        let converter = create_compound_row_converter(&self.data_types)?;

        // Convert page columns to rows
        let value_cols: Vec<ArrayRef> = self.columns
            .iter()
            .map(|name| batch.column_by_name(name).cloned())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::Index {
                message: "Missing value columns in page".to_string(),
                location: location!(),
            })?;

        let page_rows = converter.convert_columns(&value_cols).map_err(|e| Error::Index {
            message: format!("Failed to convert page to rows: {}", e),
            location: location!(),
        })?;

        // Binary search for the key
        let key_bytes = key.as_bytes();
        let mut results = RowAddrTreeMap::new();

        // Find matching rows
        let row_ids = batch
            .column_by_name(COMPOUND_IDS_COLUMN)
            .ok_or_else(|| Error::Index {
                message: "Missing _rowid column in page".to_string(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>();

        for idx in 0..page_rows.num_rows() {
            let row = page_rows.row(idx);
            if row.as_ref() == key_bytes {
                results.insert(row_ids.value(idx));
            } else if row.as_ref() > key_bytes {
                // Since data is sorted, we can stop early
                break;
            }
        }

        Ok(results)
    }

    /// Search for rows matching a prefix (with optional range on next column).
    fn search_prefix(
        &self,
        batch: &RecordBatch,
        prefix: &[ScalarValue],
        range: Option<&(std::ops::Bound<ScalarValue>, std::ops::Bound<ScalarValue>)>,
    ) -> Result<RowAddrTreeMap> {
        let mut results = RowAddrTreeMap::new();

        // Get row IDs column
        let row_ids = batch
            .column_by_name(COMPOUND_IDS_COLUMN)
            .ok_or_else(|| Error::Index {
                message: "Missing _rowid column in page".to_string(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>();

        // Check each row against prefix predicates
        for row_idx in 0..batch.num_rows() {
            let mut matches = true;

            // Check prefix columns for equality
            for (col_idx, expected_value) in prefix.iter().enumerate() {
                let col = batch.column_by_name(&self.columns[col_idx]).ok_or_else(|| {
                    Error::Index {
                        message: format!("Missing column {} in page", self.columns[col_idx]),
                        location: location!(),
                    }
                })?;

                let actual_value = ScalarValue::try_from_array(col, row_idx).map_err(|e| {
                    Error::Index {
                        message: format!("Failed to get value at row {}: {}", row_idx, e),
                        location: location!(),
                    }
                })?;

                if actual_value != *expected_value {
                    matches = false;
                    break;
                }
            }

            // Check range on next column if present and prefix matched
            if matches {
                if let Some((lower, upper)) = range {
                    let range_col_idx = prefix.len();
                    if range_col_idx < self.columns.len() {
                        matches = self.matches_range(batch, row_idx, range_col_idx, lower, upper)?;
                    }
                }
            }

            if matches {
                results.insert(row_ids.value(row_idx));
            }
        }

        Ok(results)
    }

    /// Check if a row matches a range predicate.
    fn matches_range(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
        col_idx: usize,
        lower: &std::ops::Bound<ScalarValue>,
        upper: &std::ops::Bound<ScalarValue>,
    ) -> Result<bool> {
        use std::ops::Bound;

        let col = batch.column_by_name(&self.columns[col_idx]).ok_or_else(|| Error::Index {
            message: format!("Missing column {} in page", self.columns[col_idx]),
            location: location!(),
        })?;

        let value = ScalarValue::try_from_array(col, row_idx).map_err(|e| Error::Index {
            message: format!("Failed to get value at row {}: {}", row_idx, e),
            location: location!(),
        })?;

        // NULL doesn't match any range
        if value.is_null() {
            return Ok(false);
        }

        let lower_ok = match lower {
            Bound::Unbounded => true,
            Bound::Included(v) => {
                value.partial_cmp(v).is_some_and(|o| o != std::cmp::Ordering::Less)
            }
            Bound::Excluded(v) => {
                value.partial_cmp(v) == Some(std::cmp::Ordering::Greater)
            }
        };

        let upper_ok = match upper {
            Bound::Unbounded => true,
            Bound::Included(v) => {
                value.partial_cmp(v).is_some_and(|o| o != std::cmp::Ordering::Greater)
            }
            Bound::Excluded(v) => {
                value.partial_cmp(v) == Some(std::cmp::Ordering::Less)
            }
        };

        Ok(lower_ok && upper_ok)
    }

    /// Search for rows within a compound key range.
    fn search_range(
        &self,
        batch: &RecordBatch,
        lower: &std::ops::Bound<super::compound::CompoundKey>,
        upper: &std::ops::Bound<super::compound::CompoundKey>,
    ) -> Result<RowAddrTreeMap> {
        use std::ops::Bound;

        // Create row converter for comparison
        let converter = create_compound_row_converter(&self.data_types)?;

        // Convert page columns to rows
        let value_cols: Vec<ArrayRef> = self.columns
            .iter()
            .map(|name| batch.column_by_name(name).cloned())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::Index {
                message: "Missing value columns in page".to_string(),
                location: location!(),
            })?;

        let page_rows = converter.convert_columns(&value_cols).map_err(|e| Error::Index {
            message: format!("Failed to convert page to rows: {}", e),
            location: location!(),
        })?;

        let row_ids = batch
            .column_by_name(COMPOUND_IDS_COLUMN)
            .ok_or_else(|| Error::Index {
                message: "Missing _rowid column in page".to_string(),
                location: location!(),
            })?
            .as_primitive::<UInt64Type>();

        let mut results = RowAddrTreeMap::new();

        for idx in 0..page_rows.num_rows() {
            let row = page_rows.row(idx);
            let row_bytes = row.as_ref();

            let lower_ok = match lower {
                Bound::Unbounded => true,
                Bound::Included(k) => row_bytes >= k.as_bytes(),
                Bound::Excluded(k) => row_bytes > k.as_bytes(),
            };

            let upper_ok = match upper {
                Bound::Unbounded => true,
                Bound::Included(k) => row_bytes <= k.as_bytes(),
                Bound::Excluded(k) => row_bytes < k.as_bytes(),
            };

            if lower_ok && upper_ok {
                results.insert(row_ids.value(idx));
            }
        }

        Ok(results)
    }
}

// Implement Index trait for CompoundBTreeIndex
#[async_trait]
impl Index for CompoundBTreeIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "CompoundBTreeIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        let index_reader = LazyCompoundIndexReader::new(self.store.clone());
        let reader = index_reader.get().await?;
        let num_rows = reader.num_rows();
        let batch_size = self.batch_size as usize;
        let num_pages = num_rows.div_ceil(batch_size);

        for page_idx in 0..num_pages {
            let page = self
                .read_page(page_idx as u32, index_reader.clone(), &NoOpMetricsCollector)
                .await?;
            let inserted = self
                .index_cache
                .insert_with_key(
                    &CompoundBTreePageKey {
                        page_number: page_idx as u32,
                    },
                    Arc::new(CachedCompoundPage::new(page)),
                )
                .await;

            if !inserted {
                return Err(Error::Internal {
                    message: "Failed to prewarm index: cache is no longer available".to_string(),
                    location: location!(),
                });
            }
        }

        Ok(())
    }

    fn index_type(&self) -> crate::IndexType {
        crate::IndexType::Scalar
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "CompoundBTree",
            "columns": self.columns,
            "num_pages": self.page_lookup.num_pages(),
        }))
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::default();

        let page_reader = self.store.open_index_file(COMPOUND_PAGES_NAME).await?;
        let num_batches = page_reader.num_batches(self.batch_size).await;

        for page_idx in 0..num_batches {
            let batch = page_reader
                .read_record_batch(page_idx as u64, self.batch_size)
                .await?;

            let row_ids = batch
                .column_by_name(COMPOUND_IDS_COLUMN)
                .ok_or_else(|| Error::Index {
                    message: "Missing _rowid column".to_string(),
                    location: location!(),
                })?
                .as_primitive::<UInt64Type>();

            for i in 0..row_ids.len() {
                let row_id = row_ids.value(i);
                let frag_id = (row_id >> 32) as u32;
                frag_ids.insert(frag_id);
            }
        }

        Ok(frag_ids)
    }
}

// ============================================================================
// Update Support Methods for CompoundBTreeIndex
// ============================================================================

impl CompoundBTreeIndex {
    /// Create a stream of data from the index.
    ///
    /// Returns a stream of batches with schema [col0, col1, ..., colN, _rowid].
    /// Column names are generic (col0, col1, ...) to match the training input format.
    async fn into_data_stream(self) -> Result<SendableRecordBatchStream> {
        let reader = self.store.open_index_file(COMPOUND_PAGES_NAME).await?;
        let num_batches = reader.num_batches(self.batch_size).await;
        
        // Build output schema with generic column names: col0, col1, ..., colN, _rowid
        let num_value_cols = self.columns.len();
        let mut fields = Vec::with_capacity(num_value_cols + 1);
        for (i, dt) in self.data_types.iter().enumerate() {
            fields.push(Field::new(format!("col{}", i), dt.clone(), true));
        }
        fields.push(Field::new(COMPOUND_IDS_COLUMN, DataType::UInt64, false));
        let new_schema = Arc::new(Schema::new(fields));
        let new_schema_clone = new_schema.clone();

        // Create stream that reads pages and renames columns
        let reader = Arc::new(reader);
        let batch_size = self.batch_size;
        let source_col_count = num_value_cols;

        let page_stream = stream::iter((0..num_batches).map(move |page_num| {
            let reader = reader.clone();
            async move {
                reader.read_record_batch(page_num as u64, batch_size).await
            }
        }));

        let batches = page_stream
            .buffered(self.store.io_parallelism())
            .map_err(DataFusionError::from)
            .map_ok(move |batch| {
                // Rename columns to generic names
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(source_col_count + 1);
                for i in 0..source_col_count {
                    columns.push(batch.column(i).clone());
                }
                // The last column is _rowid
                columns.push(batch.column(source_col_count).clone());
                
                RecordBatch::try_new(new_schema.clone(), columns).unwrap()
            })
            .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            new_schema_clone,
            batches,
        )))
    }

    /// Create an execution plan from the index data.
    async fn into_old_data(self) -> Result<Arc<dyn ExecutionPlan>> {
        let stream = self.into_data_stream().await?;
        Ok(Arc::new(OneShotExec::new(stream)))
    }

    /// Combine old index data with new data in sorted order.
    ///
    /// Creates a merged stream suitable for retraining the index.
    async fn combine_old_new(
        self,
        new_data: SendableRecordBatchStream,
        chunk_size: u64,
    ) -> Result<SendableRecordBatchStream> {
        let num_value_cols = self.columns.len();

        let new_input = Arc::new(OneShotExec::new(new_data));
        let old_input = self.into_old_data().await?;

        debug_assert_eq!(
            old_input.schema().flattened_fields().len(),
            new_input.schema().flattened_fields().len()
        );

        // Build sort expressions for all value columns (compound key ordering)
        let mut sort_exprs = Vec::with_capacity(num_value_cols);
        for i in 0..num_value_cols {
            let col_name = format!("col{}", i);
            sort_exprs.push(PhysicalSortExpr {
                expr: Arc::new(Column::new(&col_name, i)),
                options: SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            });
        }

        // Union the two inputs and merge them in sorted order
        let all_data = Arc::new(UnionExec::new(vec![old_input, new_input]));
        let ordering = datafusion_physical_expr::LexOrdering::new(sort_exprs)
            .ok_or_else(|| Error::Internal {
                message: "Failed to create LexOrdering for compound index merge".to_string(),
                location: location!(),
            })?;
        let ordered = Arc::new(SortPreservingMergeExec::new(ordering, all_data));

        let unchunked = execute_plan(
            ordered,
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )?;

        Ok(chunk_concat_stream(unchunked, chunk_size as usize))
    }
}

// Implement ScalarIndex trait for CompoundBTreeIndex
#[async_trait]
impl ScalarIndex for CompoundBTreeIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query
            .as_any()
            .downcast_ref::<CompoundSargableQuery>()
            .ok_or_else(|| Error::Index {
                message: "CompoundBTreeIndex expects CompoundSargableQuery".to_string(),
                location: location!(),
            })?;

        // Find candidate pages using per-column statistics pruning
        let pages = self.page_lookup.find_candidate_pages(query);

        debug!("Searching {} compound btree pages", pages.len());

        // Search each candidate page in parallel
        let lazy_reader = LazyCompoundIndexReader::new(self.store.clone());
        let page_tasks: Vec<_> = pages
            .into_iter()
            .map(|page_idx| {
                let reader = lazy_reader.clone();
                async move { self.search_page(query, page_idx, reader, metrics).await }
            })
            .collect();

        // Collect results
        let row_ids = stream::iter(page_tasks)
            .buffered(self.store.io_parallelism())
            .try_collect::<RowAddrTreeMap>()
            .await?;

        Ok(SearchResult::Exact(row_ids))
    }

    fn can_remap(&self) -> bool {
        true
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // Remap and write pages
        let mut page_file = dest_store
            .new_index_file(COMPOUND_PAGES_NAME, self.sub_index.schema().clone())
            .await?;

        let page_reader = self.store.open_index_file(COMPOUND_PAGES_NAME).await?;
        let num_batches = page_reader.num_batches(self.batch_size).await;

        for page_idx in 0..num_batches {
            let batch = page_reader
                .read_record_batch(page_idx as u64, self.batch_size)
                .await?;
            let remapped = self.sub_index.remap_subindex(batch, mapping).await?;
            page_file.write_record_batch(remapped).await?;
        }

        page_file.finish().await?;

        // Copy lookup file as-is
        self.store
            .copy_index_file(COMPOUND_LOOKUP_NAME, dest_store)
            .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::CompoundBTreeIndexDetails {
                column_names: self.columns.clone(),
                num_columns: self.columns.len() as u32,
            })
            .map_err(|e| Error::Internal {
                message: format!("Failed to serialize index details: {}", e),
                location: location!(),
            })?,
            index_version: COMPOUND_BTREE_INDEX_VERSION,
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // Merge the existing index data with the new data
        let merged_data_source = self
            .clone()
            .combine_old_new(new_data, self.batch_size)
            .await?;

        // Create compound schema for training
        let compound_schema = CompoundIndexSchema::new(
            self.columns.clone(),
            self.data_types.clone(),
        )?;

        // Retrain the index with merged data
        train_compound_btree_index(
            merged_data_source,
            self.sub_index.as_ref(),
            dest_store,
            &compound_schema,
            self.batch_size,
            None,
        )
        .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::CompoundBTreeIndexDetails {
                column_names: self.columns.clone(),
                num_columns: self.columns.len() as u32,
            })
            .map_err(|e| Error::Internal {
                message: format!("Failed to serialize index details: {}", e),
                location: location!(),
            })?,
            index_version: COMPOUND_BTREE_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::Values).with_row_id())
    }

    fn derive_index_params(&self) -> Result<super::ScalarIndexParams> {
        let params = serde_json::to_value(CompoundBTreeParameters {
            page_size: Some(self.batch_size),
            column_names: self.columns.clone(),
        })?;
        Ok(super::ScalarIndexParams::new("CompoundBTree".to_string()).with_params(&params))
    }
}

// ============================================================================
// Plugin Implementation
// ============================================================================

/// Version number for compound BTree index.
const COMPOUND_BTREE_INDEX_VERSION: u32 = 1;

// ============================================================================
// CompoundQueryParser - Query Parsing for Compound Indices
// ============================================================================

use super::expression::IndexedExpression;
use datafusion_expr::Operator;

/// Parser for compound index queries.
///
/// This parser recognizes AND predicates that match the compound index's
/// column structure and builds a `CompoundSargableQuery`.
///
/// # Supported Query Patterns
///
/// - Full key lookup: `col1 = v1 AND col2 = v2 AND col3 = v3`
/// - Prefix lookup: `col1 = v1 AND col2 = v2`
/// - Prefix + range: `col1 = v1 AND col2 > v2`
///
/// # Leftmost Prefix Rule
///
/// The parser follows the leftmost prefix rule: predicates must cover
/// contiguous columns starting from the first column. Gaps are not allowed.
#[derive(Debug)]
pub struct CompoundQueryParser {
    /// Index name.
    index_name: String,
    /// Column names in index order.
    columns: Vec<String>,
    /// Column data types.
    data_types: Vec<DataType>,
}

impl CompoundQueryParser {
    /// Create a new CompoundQueryParser.
    pub fn new(index_name: String, columns: Vec<String>, data_types: Vec<DataType>) -> Self {
        Self {
            index_name,
            columns,
            data_types,
        }
    }

    /// Get the column names in this index.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Get the data types for indexed columns.
    pub fn data_types(&self) -> &[DataType] {
        &self.data_types
    }

    /// Get the index name.
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Check if a column is the first column in this index.
    pub fn is_first_column(&self, col: &str) -> bool {
        self.columns.first().is_some_and(|c| c == col)
    }

    /// Check if a column is part of this index.
    pub fn contains_column(&self, col: &str) -> bool {
        self.columns.iter().any(|c| c == col)
    }

    /// Get the position of a column in this index (0-indexed).
    pub fn column_position(&self, col: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == col)
    }
}

impl ScalarQueryParser for CompoundQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: &std::ops::Bound<ScalarValue>,
        high: &std::ops::Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        // For compound indices, we handle BETWEEN as a range on the first column
        // This is a simplified implementation - full support requires collecting
        // predicates from the AND expression context
        if !self.is_first_column(column) {
            return None;
        }

        // Create a prefix lookup with range on the first column
        let range = (low.clone(), high.clone());
        let query = CompoundSargableQuery::prefix_lookup_with_range(vec![], range);

        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_in_list(&self, _column: &str, _in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        // IN list queries on compound indices are complex - defer for now
        None
    }

    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        // Boolean equality on first column
        if !self.is_first_column(column) {
            return None;
        }

        let query = CompoundSargableQuery::prefix_lookup(vec![ScalarValue::Boolean(Some(value))]);

        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        // NULL check on first column
        if !self.is_first_column(column) {
            return None;
        }

        let query = CompoundSargableQuery::prefix_lookup(vec![ScalarValue::Null]);

        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        // For compound indices, single-column comparisons are only useful
        // on the first column (prefix lookup pattern)
        if !self.is_first_column(column) {
            return None;
        }

        let query = match op {
            Operator::Eq => {
                // Equality on first column -> prefix lookup
                CompoundSargableQuery::prefix_lookup(vec![value.clone()])
            }
            Operator::Lt => {
                // Range on first column
                CompoundSargableQuery::prefix_lookup_with_range(
                    vec![],
                    (std::ops::Bound::Unbounded, std::ops::Bound::Excluded(value.clone())),
                )
            }
            Operator::LtEq => {
                CompoundSargableQuery::prefix_lookup_with_range(
                    vec![],
                    (std::ops::Bound::Unbounded, std::ops::Bound::Included(value.clone())),
                )
            }
            Operator::Gt => {
                CompoundSargableQuery::prefix_lookup_with_range(
                    vec![],
                    (std::ops::Bound::Excluded(value.clone()), std::ops::Bound::Unbounded),
                )
            }
            Operator::GtEq => {
                CompoundSargableQuery::prefix_lookup_with_range(
                    vec![],
                    (std::ops::Bound::Included(value.clone()), std::ops::Bound::Unbounded),
                )
            }
            // NotEq will be handled by caller via maybe_not()
            Operator::NotEq => CompoundSargableQuery::prefix_lookup(vec![value.clone()]),
            _ => return None,
        };

        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_scalar_function(
        &self,
        _column: &str,
        _data_type: &DataType,
        _func: &datafusion_expr::ScalarUDF,
        _args: &[datafusion_expr::Expr],
    ) -> Option<IndexedExpression> {
        // Scalar functions not supported on compound indices
        None
    }
}

/// Parameters for compound BTree index training.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct CompoundBTreeParameters {
    /// Size of each page in the index (number of rows).
    pub page_size: Option<u64>,
    /// Column names in index order.
    pub column_names: Vec<String>,
}

/// Training request for compound BTree index.
#[derive(Debug)]
pub struct CompoundBTreeTrainingRequest {
    criteria: TrainingCriteria,
    parameters: CompoundBTreeParameters,
}

impl CompoundBTreeTrainingRequest {
    pub fn new(parameters: CompoundBTreeParameters) -> Self {
        Self {
            criteria: TrainingCriteria::new(TrainingOrdering::Values).with_row_id(),
            parameters,
        }
    }
}

impl TrainingRequest for CompoundBTreeTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// Plugin for compound (multi-column) BTree scalar indices.
///
/// This plugin enables creation and loading of compound indices that support
/// efficient lookups on multi-column predicates.
#[derive(Debug, Default)]
pub struct CompoundBTreeIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for CompoundBTreeIndexPlugin {
    fn name(&self) -> &str {
        "CompoundBTree"
    }

    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::InvalidInput {
                source: "A compound btree index cannot include nested fields.".into(),
                location: location!(),
            });
        }

        let parameters: CompoundBTreeParameters = if params.is_empty() {
            CompoundBTreeParameters::default()
        } else {
            serde_json::from_str(params).map_err(|e| Error::InvalidInput {
                source: format!("Invalid compound btree parameters: {}", e).into(),
                location: location!(),
            })?
        };

        Ok(Box::new(CompoundBTreeTrainingRequest::new(parameters)))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        COMPOUND_BTREE_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // Parse index details to get column names
        let details: pb::CompoundBTreeIndexDetails =
            prost_types::Any::to_msg(index_details).ok()?;

        // We need data types, but they're not stored in the protobuf message.
        // For now, we can't create a fully functional parser without loading the index.
        // Return None until the index is loaded (data types come from the lookup file).
        //
        // Note: A future improvement would be to store data types in the protobuf message
        // or have a different mechanism to provide them at query parse time.
        //
        // For M3, the CompoundQueryParser will be used via get_compound_index() on
        // IndexInformationProvider, which can provide the data types from the loaded index.
        if details.column_names.is_empty() {
            return None;
        }

        // Create parser with empty data types - it will work for simple cases
        // where we don't need type coercion, but full functionality requires
        // the data types from the loaded index.
        let data_types: Vec<DataType> = vec![DataType::Null; details.column_names.len()];

        Some(Box::new(CompoundQueryParser::new(
            index_name,
            details.column_names,
            data_types,
        )))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex> {
        let request = request
            .as_any()
            .downcast_ref::<CompoundBTreeTrainingRequest>()
            .ok_or_else(|| Error::Internal {
                message: "Invalid training request type for CompoundBTree".to_string(),
                location: location!(),
            })?;

        // Extract column information from the data schema
        let schema = data.schema();
        let column_names: Vec<String> = if request.parameters.column_names.is_empty() {
            // If not specified, use all columns except _rowid
            schema
                .fields()
                .iter()
                .filter(|f| f.name() != ROW_ID)
                .map(|f| f.name().clone())
                .collect()
        } else {
            request.parameters.column_names.clone()
        };

        let data_types: Vec<DataType> = column_names
            .iter()
            .map(|name| {
                schema
                    .field_with_name(name)
                    .map(|f| f.data_type().clone())
                    .map_err(|_| Error::Index {
                        message: format!("Column '{}' not found in training data", name),
                        location: location!(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // Create compound schema for validation
        let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone())?;

        // Create flat index metadata for training
        let flat_metadata = CompoundFlatIndexMetadata::new(column_names.clone(), data_types);

        // Train the index
        train_compound_btree_index(
            data,
            &flat_metadata,
            index_store,
            &compound_schema,
            request.parameters.page_size.unwrap_or(DEFAULT_COMPOUND_BATCH_SIZE),
            fragment_ids,
        )
        .await?;

        // Create index details
        let details = pb::CompoundBTreeIndexDetails {
            column_names,
            num_columns: compound_schema.num_columns() as u32,
        };

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).map_err(|e| Error::Internal {
                message: format!("Failed to serialize compound index details: {}", e),
                location: location!(),
            })?,
            index_version: COMPOUND_BTREE_INDEX_VERSION,
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let details: pb::CompoundBTreeIndexDetails =
            prost_types::Any::to_msg(index_details).map_err(|e| Error::Internal {
                message: format!("Failed to deserialize compound index details: {}", e),
                location: location!(),
            })?;

        let index = CompoundBTreeIndex::load(
            index_store,
            details.column_names,
            frag_reuse_index,
            cache,
        )
        .await?;

        Ok(index)
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let details: pb::CompoundBTreeIndexDetails =
            prost_types::Any::to_msg(details).map_err(|e| Error::Internal {
                message: format!("Failed to deserialize compound index details: {}", e),
                location: location!(),
            })?;

        Ok(serde_json::json!({
            "type": "CompoundBTree",
            "column_names": details.column_names,
            "num_columns": details.num_columns,
        }))
    }
}

// ============================================================================
// Distributed Training Support - Merge Functions
// ============================================================================

use lance_io::object_store::ObjectStore;
use object_store::path::Path;

/// Merge multiple partition compound index files into a complete index.
///
/// In a distributed training environment, each worker writes partition files
/// (e.g., `part_123_compound_page_data.lance` and `part_123_compound_page_lookup.lance`).
/// This function merges them into the final `compound_page_data.lance` and `compound_page_lookup.lance`.
///
/// # Arguments
///
/// * `object_store` - Object store for listing partition files
/// * `index_dir` - Directory containing the partition files
/// * `store` - Index store for reading/writing files
/// * `batch_readhead` - Optional batch readhead for parallel I/O
///
/// # Example
///
/// ```no_run
/// # use lance_index::scalar::compound_btree::merge_compound_index_files;
/// # use lance_io::object_store::ObjectStore;
/// # use std::sync::Arc;
/// # async fn example() -> lance_core::Result<()> {
/// // After distributed training completes:
/// // merge_compound_index_files(&object_store, &index_dir, store, None).await?;
/// # Ok(())
/// # }
/// ```
pub async fn merge_compound_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    store: Arc<dyn IndexStore>,
    batch_readhead: Option<usize>,
) -> Result<()> {
    let (part_page_files, part_lookup_files) =
        list_compound_page_lookup_files(object_store, index_dir).await?;
    merge_compound_metadata_files(store, &part_page_files, &part_lookup_files, batch_readhead).await
}

/// List compound index partition files from a directory.
///
/// Returns (page_files, lookup_files) vectors.
async fn list_compound_page_lookup_files(
    object_store: &ObjectStore,
    index_dir: &Path,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut part_page_files = Vec::new();
    let mut part_lookup_files = Vec::new();

    let mut list_stream = object_store.list(Some(index_dir.clone()));

    while let Some(item) = list_stream.next().await {
        match item {
            Ok(meta) => {
                let file_name = meta.location.filename().unwrap_or_default();
                // Filter files matching the pattern part_*_compound_page_data.lance
                if file_name.starts_with("part_") && file_name.ends_with("_compound_page_data.lance")
                {
                    part_page_files.push(file_name.to_string());
                }
                // Filter files matching the pattern part_*_compound_page_lookup.lance
                if file_name.starts_with("part_")
                    && file_name.ends_with("_compound_page_lookup.lance")
                {
                    part_lookup_files.push(file_name.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    if part_page_files.is_empty() || part_lookup_files.is_empty() {
        return Err(Error::Internal {
            message: format!(
                "No compound partition files found in index directory: {} (page_files: {}, lookup_files: {})",
                index_dir, part_page_files.len(), part_lookup_files.len()
            ),
            location: location!(),
        });
    }

    Ok((part_page_files, part_lookup_files))
}

/// Extract partition ID from partition file name.
/// Expected format: "part_{partition_id}_{suffix}.lance"
fn extract_compound_partition_id(filename: &str) -> Result<u64> {
    if !filename.starts_with("part_") {
        return Err(Error::Internal {
            message: format!("Invalid partition file name format: {}", filename),
            location: location!(),
        });
    }

    let parts: Vec<&str> = filename.split('_').collect();
    if parts.len() < 3 {
        return Err(Error::Internal {
            message: format!("Invalid partition file name format: {}", filename),
            location: location!(),
        });
    }

    parts[1].parse::<u64>().map_err(|_| Error::Internal {
        message: format!(
            "Failed to parse partition ID from filename: {}",
            filename
        ),
        location: location!(),
    })
}

/// Merge partition files into final compound index files.
async fn merge_compound_metadata_files(
    store: Arc<dyn IndexStore>,
    part_page_files: &[String],
    part_lookup_files: &[String],
    batch_readhead: Option<usize>,
) -> Result<()> {
    if part_lookup_files.is_empty() || part_page_files.is_empty() {
        return Err(Error::Internal {
            message: "No partition files provided for merging".to_string(),
            location: location!(),
        });
    }

    // Validate matching counts
    if part_lookup_files.len() != part_page_files.len() {
        return Err(Error::Internal {
            message: format!(
                "Number of partition lookup files ({}) does not match page files ({})",
                part_lookup_files.len(),
                part_page_files.len()
            ),
            location: location!(),
        });
    }

    // Create lookup map for page files by partition ID
    let mut page_files_map = HashMap::new();
    for page_file in part_page_files {
        let partition_id = extract_compound_partition_id(page_file)?;
        page_files_map.insert(partition_id, page_file);
    }

    // Validate all lookup files have corresponding page files
    for lookup_file in part_lookup_files {
        let partition_id = extract_compound_partition_id(lookup_file)?;
        if !page_files_map.contains_key(&partition_id) {
            return Err(Error::Internal {
                message: format!(
                    "No corresponding page file for lookup file: {} (partition_id: {})",
                    lookup_file, partition_id
                ),
                location: location!(),
            });
        }
    }

    // Extract metadata from first lookup file
    let first_lookup_reader = store.open_index_file(&part_lookup_files[0]).await?;
    let batch_size = first_lookup_reader
        .schema()
        .metadata
        .get(COMPOUND_BATCH_SIZE_META_KEY)
        .map(|bs| bs.parse().unwrap_or(DEFAULT_COMPOUND_BATCH_SIZE))
        .unwrap_or(DEFAULT_COMPOUND_BATCH_SIZE);

    // Get page schema from first partition
    let partition_id = extract_compound_partition_id(&part_lookup_files[0])?;
    let page_file = page_files_map.get(&partition_id).unwrap();
    let page_reader = store.open_index_file(page_file).await?;
    let page_schema = page_reader.schema().clone();
    let arrow_schema = Arc::new(Schema::from(&page_schema));

    // Determine column names and data types from page schema
    let num_value_cols = arrow_schema.fields().len() - 1; // All except _rowid
    let column_names: Vec<String> = (0..num_value_cols)
        .map(|i| arrow_schema.field(i).name().clone())
        .collect();
    let data_types: Vec<DataType> = (0..num_value_cols)
        .map(|i| arrow_schema.field(i).data_type().clone())
        .collect();

    // Create output page file
    let mut output_page_file = store
        .new_index_file(COMPOUND_PAGES_NAME, arrow_schema.clone())
        .await?;

    // Merge pages and collect statistics
    let encoded_batches = merge_compound_pages(
        part_lookup_files,
        &page_files_map,
        &store,
        batch_size,
        &mut output_page_file,
        arrow_schema,
        &column_names,
        batch_readhead,
    )
    .await?;

    output_page_file.finish().await?;

    // Create lookup file with per-column statistics
    let lookup_batch = compound_stats_as_batch(encoded_batches, &column_names, &data_types)?;

    let mut file_schema = lookup_batch.schema().as_ref().clone();
    file_schema.metadata.insert(
        COMPOUND_BATCH_SIZE_META_KEY.to_string(),
        batch_size.to_string(),
    );

    let mut lookup_file = store
        .new_index_file(COMPOUND_LOOKUP_NAME, Arc::new(file_schema))
        .await?;

    lookup_file.write_record_batch(lookup_batch).await?;
    lookup_file.finish().await?;

    // Clean up partition files
    cleanup_compound_partition_files(&store, part_lookup_files, part_page_files).await;

    Ok(())
}

/// Merge compound pages using SortPreservingMergeExec.
#[allow(clippy::too_many_arguments)]
async fn merge_compound_pages(
    part_lookup_files: &[String],
    page_files_map: &HashMap<u64, &String>,
    store: &Arc<dyn IndexStore>,
    batch_size: u64,
    page_file: &mut Box<dyn IndexWriter>,
    arrow_schema: Arc<Schema>,
    column_names: &[String],
    batch_readhead: Option<usize>,
) -> Result<Vec<EncodedCompoundBatch>> {
    let mut encoded_batches = Vec::new();
    let mut page_idx = 0u32;

    debug!(
        "Starting compound SortPreservingMerge with {} partitions",
        part_lookup_files.len()
    );

    let num_value_cols = column_names.len();

    // Build stream schema with generic column names
    let mut stream_fields = Vec::with_capacity(num_value_cols + 1);
    for i in 0..num_value_cols {
        stream_fields.push(arrow_schema.field(i).clone().with_name(format!("col{}", i)));
    }
    stream_fields.push(Field::new(COMPOUND_IDS_COLUMN, DataType::UInt64, false));
    let stream_schema = Arc::new(Schema::new(stream_fields));

    // Create execution plans for each partition
    let mut inputs: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
    for lookup_file in part_lookup_files {
        let partition_id = extract_compound_partition_id(lookup_file)?;
        let page_file_name = (*page_files_map.get(&partition_id).ok_or_else(|| {
            Error::Internal {
                message: format!("Page file not found for partition ID: {}", partition_id),
                location: location!(),
            }
        })?)
        .clone();

        let reader = store.open_index_file(&page_file_name).await?;
        let num_batches = reader.num_batches(batch_size).await;
        let reader = Arc::new(reader);

        let page_stream = stream::iter((0..num_batches).map({
            let reader = reader.clone();
            move |page_num| {
                let reader = reader.clone();
                async move { reader.read_record_batch(page_num as u64, batch_size).await }
            }
        }));

        let stream_schema_clone = stream_schema.clone();
        let stream = page_stream
            .buffered(batch_readhead.unwrap_or(1))
            .map_err(DataFusionError::from)
            .map_ok(move |batch| {
                // Rename columns to generic names
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(num_value_cols + 1);
                for i in 0..num_value_cols {
                    columns.push(batch.column(i).clone());
                }
                columns.push(batch.column(num_value_cols).clone());
                RecordBatch::try_new(stream_schema_clone.clone(), columns).unwrap()
            })
            .boxed();

        let sendable_stream = Box::pin(RecordBatchStreamAdapter::new(stream_schema.clone(), stream));
        inputs.push(Arc::new(OneShotExec::new(sendable_stream)));
    }

    // Create Union and SortPreservingMerge
    let union_inputs = Arc::new(UnionExec::new(inputs));

    // Build multi-column sort expressions
    let mut sort_exprs = Vec::with_capacity(num_value_cols);
    for i in 0..num_value_cols {
        let col_name = format!("col{}", i);
        sort_exprs.push(PhysicalSortExpr {
            expr: Arc::new(Column::new(&col_name, i)),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        });
    }

    let ordering = datafusion_physical_expr::LexOrdering::new(sort_exprs).ok_or_else(|| {
        Error::Internal {
            message: "Failed to create LexOrdering for compound merge".to_string(),
            location: location!(),
        }
    })?;

    let merge_exec = Arc::new(SortPreservingMergeExec::new(ordering, union_inputs));

    let unchunked = execute_plan(
        merge_exec,
        LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        },
    )?;

    // Chunk and process
    let mut chunked_stream = chunk_concat_stream(unchunked, batch_size as usize);

    while let Some(batch) = chunked_stream.try_next().await? {
        // Write batch with original column names
        let mut writer_columns: Vec<ArrayRef> = Vec::with_capacity(num_value_cols + 1);
        for i in 0..=num_value_cols {
            writer_columns.push(batch.column(i).clone());
        }
        let writer_batch = RecordBatch::try_new(arrow_schema.clone(), writer_columns)?;
        page_file.write_record_batch(writer_batch).await?;

        // Compute statistics for this batch
        let stats = analyze_compound_batch(&batch, num_value_cols)?;

        encoded_batches.push(EncodedCompoundBatch {
            stats,
            page_number: page_idx,
        });

        page_idx += 1;
    }

    Ok(encoded_batches)
}

/// Clean up compound partition files after successful merge.
async fn cleanup_compound_partition_files(
    store: &Arc<dyn IndexStore>,
    part_lookup_files: &[String],
    part_page_files: &[String],
) {
    for file_name in part_lookup_files {
        if file_name.starts_with("part_") && file_name.ends_with("_compound_page_lookup.lance") {
            if let Err(e) = store.delete_index_file(file_name).await {
                log::warn!("Failed to delete partition lookup file {}: {}", file_name, e);
            }
        }
    }

    for file_name in part_page_files {
        if file_name.starts_with("part_") && file_name.ends_with("_compound_page_data.lance") {
            if let Err(e) = store.delete_index_file(file_name).await {
                log::warn!("Failed to delete partition page file {}: {}", file_name, e);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use std::sync::Arc;

    #[test]
    fn test_compound_flat_metadata_schema() {
        let metadata = CompoundFlatIndexMetadata::new(
            vec!["tenant_id".to_string(), "timestamp".to_string()],
            vec![DataType::Utf8, DataType::Int64],
        );

        assert_eq!(metadata.num_columns(), 2);
        assert_eq!(metadata.column_names(), &["tenant_id", "timestamp"]);

        let schema = metadata.schema();
        assert_eq!(schema.fields().len(), 3); // 2 value cols + _rowid
        assert_eq!(schema.field(0).name(), "tenant_id");
        assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(1).name(), "timestamp");
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert_eq!(schema.field(2).name(), "_rowid");
        assert_eq!(schema.field(2).data_type(), &DataType::UInt64);
    }

    #[tokio::test]
    async fn test_compound_flat_train() {
        let metadata = CompoundFlatIndexMetadata::new(
            vec!["name".to_string(), "value".to_string()],
            vec![DataType::Utf8, DataType::Int64],
        );

        // Create a batch with the expected column names + _rowid
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("value", DataType::Int64, true),
                Field::new(ROW_ID, DataType::UInt64, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![100, 200, 300])) as ArrayRef,
            ],
        )
        .unwrap();

        let trained = metadata.train(batch).await.unwrap();

        assert_eq!(trained.num_rows(), 3);
        assert_eq!(trained.num_columns(), 3);
        assert_eq!(trained.schema().field(0).name(), "name");
        assert_eq!(trained.schema().field(1).name(), "value");
        assert_eq!(trained.schema().field(2).name(), "_rowid");
    }

    #[test]
    fn test_analyze_compound_batch() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tenant", DataType::Utf8, true),
                Field::new("count", DataType::Int64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
            ],
        )
        .unwrap();

        let stats = analyze_compound_batch(&batch, 2).unwrap();

        assert_eq!(stats.len(), 2);

        // First column: tenant (sorted, so min="a", max="c")
        assert_eq!(
            stats[0].min,
            ScalarValue::Utf8(Some("a".to_string()))
        );
        assert_eq!(
            stats[0].max,
            ScalarValue::Utf8(Some("c".to_string()))
        );
        assert_eq!(stats[0].null_count, 0);

        // Second column: count
        assert_eq!(stats[1].min, ScalarValue::Int64(Some(10)));
        assert_eq!(stats[1].max, ScalarValue::Int64(Some(30)));
        assert_eq!(stats[1].null_count, 0);
    }

    #[test]
    fn test_analyze_compound_batch_with_nulls() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("tenant", DataType::Utf8, true),
                Field::new("count", DataType::Int64, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![None, Some("b"), Some("c")])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(10), None, Some(30)])) as ArrayRef,
            ],
        )
        .unwrap();

        let stats = analyze_compound_batch(&batch, 2).unwrap();

        assert_eq!(stats[0].null_count, 1);
        assert_eq!(stats[1].null_count, 1);
    }

    #[test]
    fn test_compound_stats_as_batch() {
        let stats = vec![
            EncodedCompoundBatch {
                stats: vec![
                    ColumnStats {
                        min: ScalarValue::Utf8(Some("a".to_string())),
                        max: ScalarValue::Utf8(Some("c".to_string())),
                        null_count: 0,
                    },
                    ColumnStats {
                        min: ScalarValue::Int64(Some(1)),
                        max: ScalarValue::Int64(Some(100)),
                        null_count: 2,
                    },
                ],
                page_number: 0,
            },
            EncodedCompoundBatch {
                stats: vec![
                    ColumnStats {
                        min: ScalarValue::Utf8(Some("d".to_string())),
                        max: ScalarValue::Utf8(Some("f".to_string())),
                        null_count: 1,
                    },
                    ColumnStats {
                        min: ScalarValue::Int64(Some(101)),
                        max: ScalarValue::Int64(Some(200)),
                        null_count: 0,
                    },
                ],
                page_number: 1,
            },
        ];

        let column_names = vec!["tenant".to_string(), "count".to_string()];
        let data_types = vec![DataType::Utf8, DataType::Int64];

        let batch = compound_stats_as_batch(stats, &column_names, &data_types).unwrap();

        // Schema: min_tenant, max_tenant, null_count_tenant, min_count, max_count, null_count_count, page_idx
        assert_eq!(batch.num_columns(), 7);
        assert_eq!(batch.num_rows(), 2);

        // Verify column names
        assert_eq!(batch.schema().field(0).name(), "min_tenant");
        assert_eq!(batch.schema().field(1).name(), "max_tenant");
        assert_eq!(batch.schema().field(2).name(), "null_count_tenant");
        assert_eq!(batch.schema().field(3).name(), "min_count");
        assert_eq!(batch.schema().field(4).name(), "max_count");
        assert_eq!(batch.schema().field(5).name(), "null_count_count");
        assert_eq!(batch.schema().field(6).name(), "page_idx");
    }

    #[test]
    fn test_compound_stats_empty() {
        let stats: Vec<EncodedCompoundBatch> = vec![];
        let column_names = vec!["a".to_string(), "b".to_string()];
        let data_types = vec![DataType::Utf8, DataType::Int64];

        let batch = compound_stats_as_batch(stats, &column_names, &data_types).unwrap();

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 7); // Still has the expected columns
    }

    #[tokio::test]
    async fn test_compound_flat_remap() {
        let metadata = CompoundFlatIndexMetadata::new(
            vec!["name".to_string(), "value".to_string()],
            vec![DataType::Utf8, DataType::Int64],
        );

        let batch = RecordBatch::try_new(
            metadata.schema().clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![100, 200, 300, 400])) as ArrayRef,
            ],
        )
        .unwrap();

        // Remap: 100 -> 1000, 200 -> delete, 300 -> 3000, 400 stays
        let mapping: HashMap<u64, Option<u64>> =
            HashMap::from_iter(vec![(100, Some(1000)), (200, None), (300, Some(3000))]);

        let remapped = metadata.remap_subindex(batch, &mapping).await.unwrap();

        assert_eq!(remapped.num_rows(), 3); // Row 200 was deleted

        let row_ids = remapped
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 1000);
        assert_eq!(row_ids.value(1), 3000);
        assert_eq!(row_ids.value(2), 400); // Unchanged
    }

    #[test]
    fn test_create_compound_row_converter() {
        let data_types = vec![DataType::Utf8, DataType::Int64, DataType::Float64];
        let converter = create_compound_row_converter(&data_types).unwrap();

        // The converter should have the correct number of fields
        // We can verify it was created successfully by creating empty rows
        let empty_rows = converter.empty_rows(0, 0);
        assert_eq!(empty_rows.num_rows(), 0);
    }

    // ========================================================================
    // CompoundBTreeLookup Tests
    // ========================================================================

    #[test]
    fn test_compound_btree_lookup_new() {
        let page_stats = vec![
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(1)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("c".to_string())),
                    ScalarValue::Int64(Some(100)),
                ],
                null_counts: vec![0, 0],
                page_number: 0,
            },
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("d".to_string())),
                    ScalarValue::Int64(Some(101)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("f".to_string())),
                    ScalarValue::Int64(Some(200)),
                ],
                null_counts: vec![1, 2],
                page_number: 1,
            },
        ];

        let data_types = vec![DataType::Utf8, DataType::Int64];
        let lookup = CompoundBTreeLookup::new(page_stats, data_types.clone());

        assert_eq!(lookup.num_pages(), 2);
        assert_eq!(lookup.data_types(), &data_types);
    }

    #[test]
    fn test_compound_btree_lookup_from_serialized() {
        // Create a lookup batch that matches the schema from compound_stats_as_batch
        let column_names = vec!["tenant".to_string(), "count".to_string()];
        let data_types = vec![DataType::Utf8, DataType::Int64];

        // Schema: min_tenant, max_tenant, null_count_tenant, min_count, max_count, null_count_count, page_idx
        let lookup_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("min_tenant", DataType::Utf8, true),
                Field::new("max_tenant", DataType::Utf8, true),
                Field::new("null_count_tenant", DataType::UInt32, false),
                Field::new("min_count", DataType::Int64, true),
                Field::new("max_count", DataType::Int64, true),
                Field::new("null_count_count", DataType::UInt32, false),
                Field::new("page_idx", DataType::UInt32, false),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["a", "d"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["c", "f"])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0, 1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 101])) as ArrayRef,
                Arc::new(Int64Array::from(vec![100, 200])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0, 2])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0, 1])) as ArrayRef,
            ],
        )
        .unwrap();

        let lookup = CompoundBTreeLookup::try_from_serialized(lookup_batch, &column_names).unwrap();

        assert_eq!(lookup.num_pages(), 2);
        assert_eq!(lookup.data_types(), &data_types);
    }

    #[test]
    fn test_compound_btree_lookup_pruning_equality() {
        use super::super::compound::CompoundSargableQuery;

        let page_stats = vec![
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(1)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("c".to_string())),
                    ScalarValue::Int64(Some(100)),
                ],
                null_counts: vec![0, 0],
                page_number: 0,
            },
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("d".to_string())),
                    ScalarValue::Int64(Some(101)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("f".to_string())),
                    ScalarValue::Int64(Some(200)),
                ],
                null_counts: vec![0, 0],
                page_number: 1,
            },
        ];

        let lookup = CompoundBTreeLookup::new(page_stats, vec![DataType::Utf8, DataType::Int64]);

        // Query for tenant_id = "b" - should match page 0 only
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("b".to_string())),
        ]);
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![0]);

        // Query for tenant_id = "e" - should match page 1 only
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("e".to_string())),
        ]);
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![1]);

        // Query for tenant_id = "z" - should match no pages
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("z".to_string())),
        ]);
        let pages = lookup.find_candidate_pages(&query);
        assert!(pages.is_empty());

        // Query for tenant_id = "a" - should match page 0 (boundary case)
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("a".to_string())),
        ]);
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![0]);
    }

    #[test]
    fn test_compound_btree_lookup_pruning_range() {
        use super::super::compound::CompoundSargableQuery;
        use std::ops::Bound;

        let page_stats = vec![
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(1)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(100)),
                ],
                null_counts: vec![0, 0],
                page_number: 0,
            },
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(101)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(200)),
                ],
                null_counts: vec![0, 0],
                page_number: 1,
            },
        ];

        let lookup = CompoundBTreeLookup::new(page_stats, vec![DataType::Utf8, DataType::Int64]);

        // Query: tenant_id = "a" AND timestamp > 50
        // Should match page 0 (50 < 100) and page 1 (timestamp range 101-200 > 50)
        let query = CompoundSargableQuery::prefix_lookup_with_range(
            vec![ScalarValue::Utf8(Some("a".to_string()))],
            (
                Bound::Excluded(ScalarValue::Int64(Some(50))),
                Bound::Unbounded,
            ),
        );
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![0, 1]);

        // Query: tenant_id = "a" AND timestamp > 150
        // Should match only page 1 (101-200 includes values > 150)
        let query = CompoundSargableQuery::prefix_lookup_with_range(
            vec![ScalarValue::Utf8(Some("a".to_string()))],
            (
                Bound::Excluded(ScalarValue::Int64(Some(150))),
                Bound::Unbounded,
            ),
        );
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![1]);

        // Query: tenant_id = "a" AND timestamp < 50
        // Should match only page 0 (1-100 includes values < 50)
        let query = CompoundSargableQuery::prefix_lookup_with_range(
            vec![ScalarValue::Utf8(Some("a".to_string()))],
            (
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Int64(Some(50))),
            ),
        );
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![0]);
    }

    #[test]
    fn test_compound_btree_lookup_null_handling() {
        use super::super::compound::CompoundSargableQuery;

        let page_stats = vec![
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("a".to_string())),
                    ScalarValue::Int64(Some(1)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("c".to_string())),
                    ScalarValue::Int64(Some(100)),
                ],
                null_counts: vec![5, 0], // 5 nulls in tenant column
                page_number: 0,
            },
            CompoundPageStats {
                mins: vec![
                    ScalarValue::Utf8(Some("d".to_string())),
                    ScalarValue::Int64(Some(101)),
                ],
                maxs: vec![
                    ScalarValue::Utf8(Some("f".to_string())),
                    ScalarValue::Int64(Some(200)),
                ],
                null_counts: vec![0, 0], // No nulls
                page_number: 1,
            },
        ];

        let lookup = CompoundBTreeLookup::new(page_stats, vec![DataType::Utf8, DataType::Int64]);

        // Pages with nulls in column 0
        let null_pages = lookup.pages_with_nulls(0);
        assert_eq!(null_pages, vec![0]);

        // Pages with nulls in column 1
        let null_pages = lookup.pages_with_nulls(1);
        assert!(null_pages.is_empty());

        // Query for NULL in first column - should match page 0 only
        let query = CompoundSargableQuery::prefix_lookup(vec![ScalarValue::Utf8(None)]);
        let pages = lookup.find_candidate_pages(&query);
        assert_eq!(pages, vec![0]);
    }

    #[test]
    fn test_compound_btree_lookup_empty() {
        let lookup = CompoundBTreeLookup::new(vec![], vec![DataType::Utf8, DataType::Int64]);

        assert_eq!(lookup.num_pages(), 0);

        // Any query should return empty pages
        use super::super::compound::CompoundSargableQuery;
        let query = CompoundSargableQuery::prefix_lookup(vec![
            ScalarValue::Utf8(Some("test".to_string())),
        ]);
        let pages = lookup.find_candidate_pages(&query);
        assert!(pages.is_empty());
    }

    // ========================================================================
    // CompoundQueryParser Tests
    // ========================================================================

    #[test]
    fn test_compound_query_parser_new() {
        let parser = CompoundQueryParser::new(
            "test_index".to_string(),
            vec!["tenant_id".to_string(), "status".to_string(), "timestamp".to_string()],
            vec![DataType::Utf8, DataType::Utf8, DataType::Int64],
        );

        assert_eq!(parser.index_name(), "test_index");
        assert_eq!(parser.columns().len(), 3);
        assert_eq!(parser.data_types().len(), 3);
    }

    #[test]
    fn test_compound_query_parser_column_position() {
        let parser = CompoundQueryParser::new(
            "test_index".to_string(),
            vec!["tenant_id".to_string(), "status".to_string()],
            vec![DataType::Utf8, DataType::Utf8],
        );

        assert!(parser.is_first_column("tenant_id"));
        assert!(!parser.is_first_column("status"));
        assert!(!parser.is_first_column("unknown"));

        assert!(parser.contains_column("tenant_id"));
        assert!(parser.contains_column("status"));
        assert!(!parser.contains_column("unknown"));

        assert_eq!(parser.column_position("tenant_id"), Some(0));
        assert_eq!(parser.column_position("status"), Some(1));
        assert_eq!(parser.column_position("unknown"), None);
    }

    #[test]
    fn test_compound_query_parser_visit_comparison_first_column() {
        use super::super::expression::ScalarQueryParser;
        use datafusion_expr::Operator;

        let parser = CompoundQueryParser::new(
            "test_index".to_string(),
            vec!["tenant_id".to_string(), "status".to_string()],
            vec![DataType::Utf8, DataType::Utf8],
        );

        // Equality on first column should work
        let result = parser.visit_comparison(
            "tenant_id",
            &ScalarValue::Utf8(Some("acme".to_string())),
            &Operator::Eq,
        );
        assert!(result.is_some());

        // Range on first column should work
        let result = parser.visit_comparison(
            "tenant_id",
            &ScalarValue::Utf8(Some("acme".to_string())),
            &Operator::Gt,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_compound_query_parser_visit_comparison_non_first_column() {
        use super::super::expression::ScalarQueryParser;
        use datafusion_expr::Operator;

        let parser = CompoundQueryParser::new(
            "test_index".to_string(),
            vec!["tenant_id".to_string(), "status".to_string()],
            vec![DataType::Utf8, DataType::Utf8],
        );

        // Comparison on non-first column should NOT work (leftmost prefix rule)
        let result = parser.visit_comparison(
            "status",
            &ScalarValue::Utf8(Some("active".to_string())),
            &Operator::Eq,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_compound_query_parser_visit_is_null() {
        use super::super::expression::ScalarQueryParser;

        let parser = CompoundQueryParser::new(
            "test_index".to_string(),
            vec!["tenant_id".to_string(), "status".to_string()],
            vec![DataType::Utf8, DataType::Utf8],
        );

        // IS NULL on first column should work
        let result = parser.visit_is_null("tenant_id");
        assert!(result.is_some());

        // IS NULL on non-first column should NOT work
        let result = parser.visit_is_null("status");
        assert!(result.is_none());
    }

    // ========================================================================
    // Integration Tests for Update/Remap
    // ========================================================================

    use datafusion::physical_plan::stream::RecordBatchStreamAdapter as DFRecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;

    use crate::scalar::lance_format::LanceIndexStore;

    /// Helper to create a sorted test batch for compound index training.
    fn create_sorted_test_batch(
        tenant_ids: Vec<&str>,
        statuses: Vec<&str>,
        row_ids: Vec<u64>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("col0", DataType::Utf8, true),
                Field::new("col1", DataType::Utf8, true),
                Field::new(COMPOUND_IDS_COLUMN, DataType::UInt64, false),
            ])),
            vec![
                Arc::new(StringArray::from(tenant_ids)) as ArrayRef,
                Arc::new(StringArray::from(statuses)) as ArrayRef,
                Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Helper to create a stream from batches.
    fn batches_to_stream(
        batches: Vec<RecordBatch>,
        schema: Arc<Schema>,
    ) -> SendableRecordBatchStream {
        let stream = stream::iter(batches.into_iter().map(Ok::<_, datafusion_common::DataFusionError>));
        Box::pin(DFRecordBatchStreamAdapter::new(schema, stream))
    }

    #[tokio::test]
    async fn test_compound_index_update_basic() {
        // Create initial index with some data
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Initial data: tenant a-c
        let initial_batch = create_sorted_test_batch(
            vec!["a", "b", "c"],
            vec!["active", "active", "inactive"],
            vec![1, 2, 3],
        );

        let column_names = vec!["tenant".to_string(), "status".to_string()];
        let data_types = vec![DataType::Utf8, DataType::Utf8];
        let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone()).unwrap();
        let sub_index = CompoundFlatIndexMetadata::new(column_names.clone(), data_types.clone());

        let stream = batches_to_stream(vec![initial_batch], sub_index.schema().clone());

        train_compound_btree_index(
            stream,
            &sub_index,
            store.as_ref(),
            &compound_schema,
            100,
            None,
        )
        .await
        .unwrap();

        // Load the index
        let index = CompoundBTreeIndex::load(
            store.clone(),
            column_names.clone(),
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();

        assert_eq!(index.num_pages(), 1);

        // Now update with new data: tenant d-f
        let new_batch = create_sorted_test_batch(
            vec!["d", "e", "f"],
            vec!["active", "inactive", "active"],
            vec![4, 5, 6],
        );

        let new_stream = batches_to_stream(vec![new_batch], sub_index.schema().clone());

        // Create a new store for the updated index
        let update_dir = TempObjDir::default();
        let update_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            update_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Perform the update
        index
            .update(new_stream, update_store.as_ref())
            .await
            .unwrap();

        // Load the updated index
        let updated_index = CompoundBTreeIndex::load(
            update_store.clone(),
            column_names.clone(),
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();

        // Should still have 1 page (6 rows < batch_size)
        assert_eq!(updated_index.num_pages(), 1);

        // Verify all data is there by checking page reader
        let page_reader = update_store.open_index_file(COMPOUND_PAGES_NAME).await.unwrap();
        let all_data = page_reader.read_record_batch(0, 100).await.unwrap();
        assert_eq!(all_data.num_rows(), 6);
    }

    #[tokio::test]
    async fn test_compound_index_remap_basic() {
        // Create index with data
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let initial_batch = create_sorted_test_batch(
            vec!["a", "b", "c", "d"],
            vec!["active", "active", "inactive", "active"],
            vec![1, 2, 3, 4],
        );

        let column_names = vec!["tenant".to_string(), "status".to_string()];
        let data_types = vec![DataType::Utf8, DataType::Utf8];
        let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone()).unwrap();
        let sub_index = CompoundFlatIndexMetadata::new(column_names.clone(), data_types.clone());

        let stream = batches_to_stream(vec![initial_batch], sub_index.schema().clone());

        train_compound_btree_index(
            stream,
            &sub_index,
            store.as_ref(),
            &compound_schema,
            100,
            None,
        )
        .await
        .unwrap();

        let index = CompoundBTreeIndex::load(
            store.clone(),
            column_names.clone(),
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();

        // Create remap that deletes row 2 and 3
        let mut mapping = HashMap::new();
        mapping.insert(2u64, None);  // Delete row 2
        mapping.insert(3u64, None);  // Delete row 3

        // Create new store for remapped index
        let remap_dir = TempObjDir::default();
        let remap_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            remap_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Perform remap
        index.remap(&mapping, remap_store.as_ref()).await.unwrap();

        // Load remapped index and verify
        let remapped_index = CompoundBTreeIndex::load(
            remap_store.clone(),
            column_names.clone(),
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();

        // Page count should be same (stats are preserved)
        assert_eq!(remapped_index.num_pages(), 1);

        // Verify page data - should have 2 rows (1 and 4)
        let page_reader = remap_store.open_index_file(COMPOUND_PAGES_NAME).await.unwrap();
        let all_data = page_reader.read_record_batch(0, 100).await.unwrap();
        assert_eq!(all_data.num_rows(), 2);

        // Verify row IDs
        let row_ids = all_data
            .column_by_name(COMPOUND_IDS_COLUMN)
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>();
        assert_eq!(row_ids.value(0), 1);
        assert_eq!(row_ids.value(1), 4);
    }

    #[tokio::test]
    async fn test_compound_index_remap_no_op() {
        // Test that empty/no-op remap produces identical results
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let initial_batch = create_sorted_test_batch(
            vec!["a", "b", "c"],
            vec!["x", "y", "z"],
            vec![10, 20, 30],
        );

        let column_names = vec!["col1".to_string(), "col2".to_string()];
        let data_types = vec![DataType::Utf8, DataType::Utf8];
        let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone()).unwrap();
        let sub_index = CompoundFlatIndexMetadata::new(column_names.clone(), data_types.clone());

        let stream = batches_to_stream(vec![initial_batch], sub_index.schema().clone());

        train_compound_btree_index(
            stream,
            &sub_index,
            store.as_ref(),
            &compound_schema,
            100,
            None,
        )
        .await
        .unwrap();

        let index = CompoundBTreeIndex::load(
            store.clone(),
            column_names.clone(),
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();

        // Empty mapping = no-op
        let mapping = HashMap::new();

        let remap_dir = TempObjDir::default();
        let remap_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            remap_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        index.remap(&mapping, remap_store.as_ref()).await.unwrap();

        // Verify data is identical
        let original_reader = store.open_index_file(COMPOUND_PAGES_NAME).await.unwrap();
        let remapped_reader = remap_store.open_index_file(COMPOUND_PAGES_NAME).await.unwrap();

        assert_eq!(original_reader.num_rows(), remapped_reader.num_rows());

        let original_data = original_reader.read_record_batch(0, 100).await.unwrap();
        let remapped_data = remapped_reader.read_record_batch(0, 100).await.unwrap();

        assert_eq!(original_data, remapped_data);
    }
}
