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
        // Extract value columns by name and the row ID column
        let mut columns = Vec::with_capacity(self.num_columns + 1);

        for col_name in &self.column_names {
            let col = batch.column_by_name(col_name).ok_or_else(|| Error::Index {
                message: format!("Missing column '{}' in training batch", col_name),
                location: location!(),
            })?;
            columns.push(col.clone());
        }

        // Add row ID column
        let row_id_col = batch.column_by_name(ROW_ID).ok_or_else(|| Error::Index {
            message: format!("Missing '{}' column in training batch", ROW_ID),
            location: location!(),
        })?;
        columns.push(row_id_col.clone());

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
fn analyze_compound_batch(batch: &RecordBatch, column_names: &[String]) -> Result<Vec<ColumnStats>> {
    if batch.num_rows() == 0 {
        return Err(Error::Internal {
            message: "Received an empty batch in compound btree training".to_string(),
            location: location!(),
        });
    }

    let mut stats = Vec::with_capacity(column_names.len());

    for col_name in column_names {
        let col = batch.column_by_name(col_name).ok_or_else(|| Error::Internal {
            message: format!("Missing column '{}' in batch", col_name),
            location: location!(),
        })?;

        // For sorted data: min is first row, max is last row
        let min = ScalarValue::try_from_array(col, 0).map_err(|e| Error::Internal {
            message: format!("Failed to get min value for column '{}': {}", col_name, e),
            location: location!(),
        })?;

        let max = ScalarValue::try_from_array(col, col.len() - 1).map_err(|e| Error::Internal {
            message: format!("Failed to get max value for column '{}': {}", col_name, e),
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
    column_names: &[String],
    sub_index_trainer: &dyn CompoundBTreeSubIndex,
    writer: &mut dyn IndexWriter,
) -> Result<EncodedCompoundBatch> {
    let stats = analyze_compound_batch(&batch, column_names)?;
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
            &column_names,
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
use futures::stream::{self, StreamExt};
use lance_core::cache::{LanceCache, WeakLanceCache};
use lance_core::utils::mask::RowAddrTreeMap;
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
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // Update is deferred to M4
        Err(Error::NotSupported {
            source: "Compound index update not yet implemented".into(),
            location: location!(),
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

        let column_names = vec!["tenant".to_string(), "count".to_string()];
        let stats = analyze_compound_batch(&batch, &column_names).unwrap();

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

        let column_names = vec!["tenant".to_string(), "count".to_string()];
        let stats = analyze_compound_batch(&batch, &column_names).unwrap();

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
}
