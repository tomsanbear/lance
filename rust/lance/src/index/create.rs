// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    Error, Result,
    dataset::{
        Dataset,
        transaction::{Operation, TransactionBuilder},
    },
    index::{
        DatasetIndexExt, DatasetIndexInternalExt,
        api::{IndexSegment, IndexSegmentPlan},
        build_index_metadata_from_segments,
        scalar::{build_compound_btree_index, build_scalar_index},
        vector::{
            LANCE_VECTOR_INDEX, VectorIndexParams, build_distributed_vector_index,
            build_empty_vector_index, build_vector_index,
        },
        vector_index_details, vector_index_details_default,
    },
};
use futures::future::{BoxFuture, try_join_all};
use lance_core::datatypes::format_field_path;
use lance_index::progress::{IndexBuildProgress, NoopIndexBuildProgress};
use lance_index::{IndexParams, IndexType, scalar::CreatedIndex};
use lance_index::{
    metrics::NoOpMetricsCollector,
    scalar::{LANCE_SCALAR_INDEX, ScalarIndexParams, inverted::tokenizer::InvertedIndexParams},
};
use lance_table::format::{IndexMetadata, list_index_files_with_sizes};
use std::{
    collections::{HashMap, HashSet},
    future::IntoFuture,
    sync::Arc,
};
use tracing::instrument;
use uuid::Uuid;

use arrow_array::RecordBatchReader;
/// Generate default index name from field path.
///
/// Joins field names with `.` to create the base index name.
/// For example: `["meta-data", "user-id"]` -> `"meta-data.user-id"`
fn default_index_name(fields: &[&str]) -> String {
    if fields.iter().any(|f| f.contains('.')) {
        format_field_path(fields)
    } else {
        fields.join(".")
    }
}

/// Validate index columns and resolve them against the schema.
///
/// Returns the resolved fields, corrected column names (with proper casing), and field IDs.
/// Supports 1-8 columns for compound indices.
fn validate_index_columns(
    schema: &lance_core::datatypes::Schema,
    columns: &[String],
) -> Result<(Vec<lance_core::datatypes::Field>, Vec<String>, Vec<i32>)> {
    if columns.is_empty() {
        return Err(Error::index(
            "Index must have at least one column".to_string(),
        ));
    }
    if columns.len() > 8 {
        return Err(Error::index(format!(
            "Compound index exceeds maximum of 8 columns (got {})",
            columns.len()
        )));
    }

    let mut fields = Vec::with_capacity(columns.len());
    let mut corrected_columns = Vec::with_capacity(columns.len());

    for column_input in columns {
        let Some(field_path) = schema.resolve_case_insensitive(column_input) else {
            return Err(Error::index(format!(
                "CreateIndex: column '{}' does not exist",
                column_input
            )));
        };
        let field = *field_path.last().unwrap();
        fields.push(field.clone());

        let names: Vec<&str> = field_path.iter().map(|f| f.name.as_str()).collect();
        corrected_columns.push(format_field_path(&names));
    }

    let field_ids: Vec<i32> = fields.iter().map(|f| f.id).collect();
    Ok((fields, corrected_columns, field_ids))
}

/// Generate a default index name with collision handling.
///
/// Uses the user-supplied column names (from `columns`) to build the base name so
/// the generated index name preserves whatever characters the caller passed.
fn generate_index_name(
    columns: &[String],
    field_ids: &[i32],
    indices: &[IndexMetadata],
) -> String {
    let column_path = if columns.len() == 1 {
        default_index_name(&[columns[0].as_str()])
    } else {
        columns.join("_")
    };
    let base_name = format!("{column_path}_idx");
    let mut candidate = base_name.clone();
    let mut counter = 2;
    while indices
        .iter()
        .any(|idx| idx.name == candidate && idx.fields != field_ids)
    {
        candidate = format!("{base_name}_{counter}");
        counter += 1;
    }
    candidate
}

pub struct CreateIndexBuilder<'a> {
    dataset: &'a mut Dataset,
    columns: Vec<String>,
    index_type: IndexType,
    params: &'a dyn IndexParams,
    name: Option<String>,
    replace: bool,
    train: bool,
    fragments: Option<Vec<u32>>,
    index_uuid: Option<String>,
    preprocessed_data: Option<Box<dyn RecordBatchReader + Send + 'static>>,
    progress: Arc<dyn IndexBuildProgress>,
    /// Transaction properties to store with this commit.
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    /// Optional sort/aggregate spill pool size in bytes for this index build.
    mem_pool_size: Option<u64>,
}

impl<'a> CreateIndexBuilder<'a> {
    pub fn new(
        dataset: &'a mut Dataset,
        columns: &[&str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> Self {
        Self {
            dataset,
            columns: columns.iter().map(|s| s.to_string()).collect(),
            index_type,
            params,
            name: None,
            replace: false,
            train: true,
            fragments: None,
            index_uuid: None,
            preprocessed_data: None,
            progress: Arc::new(NoopIndexBuildProgress),
            transaction_properties: None,
            mem_pool_size: None,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }

    pub fn train(mut self, train: bool) -> Self {
        self.train = train;
        self
    }

    pub fn fragments(mut self, fragment_ids: Vec<u32>) -> Self {
        self.fragments = Some(fragment_ids);
        self
    }

    pub fn index_uuid(mut self, uuid: String) -> Self {
        self.index_uuid = Some(uuid);
        self
    }

    pub fn preprocessed_data(
        mut self,
        stream: Box<dyn RecordBatchReader + Send + 'static>,
    ) -> Self {
        self.preprocessed_data = Some(stream);
        self
    }

    pub fn progress(mut self, p: Arc<dyn IndexBuildProgress>) -> Self {
        self.progress = p;
        self
    }

    /// Set transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed index jobs).
    pub fn transaction_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.transaction_properties = Some(Arc::new(properties));
        self
    }

    /// Set the sort/aggregate spill pool size in bytes for this index build.
    ///
    /// Passed as `mem_pool_size` to `LanceExecutionOptions` when the build
    /// reads training data via DataFusion. Defaults to Lance's global pool
    /// (controlled by `LANCE_MEM_POOL_SIZE` env var, 100 MiB if unset).
    /// On memory-constrained hosts, set this from operator config rather than
    /// relying on the env var.
    pub fn mem_pool_size(mut self, size: u64) -> Self {
        self.mem_pool_size = Some(size);
        self
    }

    #[instrument(skip_all)]
    pub async fn execute_uncommitted(&mut self) -> Result<IndexMetadata> {
        let (fields, corrected_columns, field_ids) =
            validate_index_columns(self.dataset.schema(), &self.columns)?;

        let column = corrected_columns[0].as_str();
        #[allow(unused_variables)]
        let field = &fields[0];

        // If train is true but dataset is empty, automatically set train to false
        let train = if self.train {
            self.dataset.count_rows(None).await? > 0
        } else {
            false
        };

        // Load indices from the disk.
        let indices = self.dataset.load_indices().await?;
        let fri = self
            .dataset
            .open_frag_reuse_index(&NoOpMetricsCollector)
            .await?;

        let index_name = if let Some(name) = self.name.take() {
            name
        } else {
            generate_index_name(&self.columns, &field_ids, &indices)
        };
        let existing_named_indices = indices
            .iter()
            .filter(|idx| idx.name == index_name)
            .collect::<Vec<_>>();
        if existing_named_indices
            .iter()
            .any(|idx| idx.fields != field_ids)
        {
            return Err(Error::index(format!(
                "Index name '{index_name}' already exists with different fields, \
                please specify a different name"
            )));
        }
        if !existing_named_indices.is_empty() && !self.replace {
            return Err(Error::index(format!(
                "Index name '{index_name}' already exists, \
                please specify a different name or use replace=True"
            )));
        }

        let index_id = match &self.index_uuid {
            Some(uuid_str) => Uuid::parse_str(uuid_str)
                .map_err(|e| Error::index(format!("Invalid UUID string provided: {}", e)))?,
            None => Uuid::new_v4(),
        };
        let mut output_index_uuid = index_id;

        // Handle multi-column (compound) indices
        let created_index = if self.columns.len() > 1 {
            match (self.index_type, self.params.index_name()) {
                (IndexType::Scalar | IndexType::BTree, LANCE_SCALAR_INDEX) => {
                    let params = self
                        .params
                        .as_any()
                        .downcast_ref::<ScalarIndexParams>()
                        .cloned()
                        .unwrap_or_else(|| ScalarIndexParams::new("compoundbtree".to_string()));

                    build_compound_btree_index(
                        self.dataset,
                        &corrected_columns
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        &index_id.to_string(),
                        &params,
                        train,
                        self.fragments.clone(),
                        self.mem_pool_size,
                    )
                    .await?
                }
                (index_type, _) => {
                    return Err(Error::index(format!(
                        "Index type {:?} does not support multiple columns. \
                         Use BTree or Scalar for compound indices.",
                        index_type
                    )));
                }
            }
        } else {
            match (self.index_type, self.params.index_name()) {
            (
                IndexType::Bitmap
                | IndexType::BTree
                | IndexType::Inverted
                | IndexType::NGram
                | IndexType::ZoneMap
                | IndexType::BloomFilter
                | IndexType::LabelList
                | IndexType::RTree,
                LANCE_SCALAR_INDEX,
            ) => {
                assert!(
                    self.preprocessed_data.is_none() || self.index_type.eq(&IndexType::BTree),
                    "Preprocessed data stream can only be provided for B-Tree index type at the moment."
                );
                let base_params = ScalarIndexParams::for_builtin(self.index_type.try_into()?);

                // If custom params were provided, extract the params JSON and apply it
                let params = if let Some(provided_params) =
                    self.params.as_any().downcast_ref::<ScalarIndexParams>()
                {
                    if let Some(params_json) = &provided_params.params {
                        // Parse and apply the custom parameters
                        if let Ok(json_value) =
                            serde_json::from_str::<serde_json::Value>(params_json)
                        {
                            base_params.with_params(&json_value)
                        } else {
                            base_params
                        }
                    } else {
                        base_params
                    }
                } else {
                    base_params
                };

                let preprocesssed_data = self
                    .preprocessed_data
                    .take()
                    .map(|reader| lance_datafusion::utils::reader_to_stream(Box::new(reader)));
                build_scalar_index(
                    self.dataset,
                    column,
                    &index_id.to_string(),
                    &params,
                    train,
                    self.fragments.clone(),
                    preprocesssed_data,
                    self.progress.clone(),
                    self.mem_pool_size,
                )
                .await?
            }
            (IndexType::Scalar, LANCE_SCALAR_INDEX) => {
                // Guess the index type
                let params = self
                    .params
                    .as_any()
                    .downcast_ref::<ScalarIndexParams>()
                    .ok_or_else(|| {
                        Error::index("Scalar index type must take a ScalarIndexParams".to_string())
                    })?;
                build_scalar_index(
                    self.dataset,
                    column,
                    &index_id.to_string(),
                    params,
                    train,
                    self.fragments.clone(),
                    None,
                    self.progress.clone(),
                    self.mem_pool_size,
                )
                .await?
            }
            (IndexType::Inverted, _) => {
                // Inverted index params.
                let inverted_params = self
                    .params
                    .as_any()
                    .downcast_ref::<InvertedIndexParams>()
                    .ok_or_else(|| {
                        Error::index(
                            "Inverted index type must take a InvertedIndexParams".to_string(),
                        )
                    })?;

                let params = ScalarIndexParams::new("inverted".to_string())
                    .with_params(&inverted_params.to_training_json()?);
                build_scalar_index(
                    self.dataset,
                    column,
                    &index_id.to_string(),
                    &params,
                    train,
                    self.fragments.clone(),
                    None,
                    self.progress.clone(),
                    self.mem_pool_size,
                )
                .await?
            }
            (
                IndexType::Vector
                | IndexType::IvfPq
                | IndexType::IvfSq
                | IndexType::IvfFlat
                | IndexType::IvfRq
                | IndexType::IvfHnswFlat
                | IndexType::IvfHnswPq
                | IndexType::IvfHnswSq,
                LANCE_VECTOR_INDEX,
            ) => {
                // Vector index params.
                let vec_params = self
                    .params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| {
                        Error::index("Vector index type must take a VectorIndexParams".to_string())
                    })?;
                let index_version = vec_params.index_type().version() as u32;

                if train {
                    // Check if this is distributed indexing (fragment-level)
                    if let Some(fragments) = &self.fragments {
                        // For distributed indexing, build only on specified fragments
                        // This creates temporary index metadata without committing
                        let segment_uuid = Box::pin(build_distributed_vector_index(
                            self.dataset,
                            column,
                            &index_name,
                            &index_id.to_string(),
                            vec_params,
                            fri,
                            fragments,
                            self.progress.clone(),
                            self.mem_pool_size,
                        ))
                        .await?;
                        output_index_uuid = segment_uuid;
                    } else {
                        // Standard full dataset indexing
                        Box::pin(build_vector_index(
                            self.dataset,
                            column,
                            &index_name,
                            &index_id.to_string(),
                            vec_params,
                            fri,
                            self.progress.clone(),
                            self.mem_pool_size,
                        ))
                        .await?;
                    }
                } else {
                    // Create empty vector index
                    build_empty_vector_index(
                        self.dataset,
                        column,
                        &index_name,
                        &index_id.to_string(),
                        vec_params,
                        self.mem_pool_size,
                    )
                    .await?;
                }
                // Capture file sizes after vector index creation
                let index_dir = self
                    .dataset
                    .indices_dir()
                    .join(output_index_uuid.to_string());
                let files =
                    list_index_files_with_sizes(&self.dataset.object_store, &index_dir).await?;
                CreatedIndex {
                    index_details: vector_index_details(vec_params),
                    index_version,
                    files: Some(files),
                }
            }
            // Can't use if let Some(...) here because it's not stable yet.
            // TODO: fix after https://github.com/rust-lang/rust/issues/51114
            (IndexType::Vector, name)
                if self
                    .dataset
                    .session
                    .index_extensions
                    .contains_key(&(IndexType::Vector, name.to_string())) =>
            {
                let ext = self
                    .dataset
                    .session
                    .index_extensions
                    .get(&(IndexType::Vector, name.to_string()))
                    .expect("already checked")
                    .clone()
                    .to_vector()
                    // this should never happen because we control the registration
                    // if this fails, the registration logic has a bug
                    .ok_or(Error::internal(
                        "unable to cast index extension to vector".to_string(),
                    ))?;

                if train {
                    ext.create_index(self.dataset, column, &index_id.to_string(), self.params)
                        .await?;
                } else {
                    todo!("create empty vector index when train=false");
                }
                // Capture file sizes after vector index creation
                let index_dir = self.dataset.indices_dir().join(index_id.to_string());
                let files =
                    list_index_files_with_sizes(&self.dataset.object_store, &index_dir).await?;
                CreatedIndex {
                    index_details: vector_index_details_default(),
                    index_version: self.index_type.version() as u32,
                    files: Some(files),
                }
            }
            (IndexType::FragmentReuse, _) => {
                return Err(Error::index(
                    "Fragment reuse index can only be created through compaction".to_string(),
                ));
            }
            (index_type, index_name) => {
                return Err(Error::index(format!(
                    "Index type {index_type} with name {index_name} is not supported"
                )));
            }
            }
        };

        Ok(IndexMetadata {
            uuid: output_index_uuid,
            name: index_name,
            fields: field_ids,
            dataset_version: self.dataset.manifest.version,
            fragment_bitmap: if train {
                match &self.fragments {
                    Some(fragment_ids) => Some(fragment_ids.iter().collect()),
                    None => Some(self.dataset.fragment_bitmap.as_ref().clone()),
                }
            } else {
                // Empty bitmap for untrained indices
                Some(roaring::RoaringBitmap::new())
            },
            index_details: Some(Arc::new(created_index.index_details)),
            index_version: created_index.index_version as i32,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: created_index.files,
        })
    }

    #[instrument(skip_all)]
    async fn execute(mut self) -> Result<IndexMetadata> {
        let new_idx = self.execute_uncommitted().await?;
        let index_uuid = new_idx.uuid;
        let removed_indices = if self.replace {
            self.dataset
                .load_indices()
                .await?
                .iter()
                .filter(|idx| idx.name == new_idx.name)
                .cloned()
                .collect()
        } else {
            vec![]
        };
        let transaction = if uses_segment_commit_path(self.index_type, &new_idx.name, self.params) {
            let field_id = *new_idx.fields.first().ok_or_else(|| {
                Error::internal(format!(
                    "Index '{}' is missing field ids after build",
                    new_idx.name
                ))
            })?;
            let segment_index_type = match self.index_type {
                IndexType::Vector
                | IndexType::IvfPq
                | IndexType::IvfSq
                | IndexType::IvfFlat
                | IndexType::IvfRq
                | IndexType::IvfHnswFlat
                | IndexType::IvfHnswPq
                | IndexType::IvfHnswSq => self
                    .params
                    .as_any()
                    .downcast_ref::<VectorIndexParams>()
                    .ok_or_else(|| {
                        Error::index("Vector index type must take a VectorIndexParams".to_string())
                    })?
                    .index_type(),
                unsupported => {
                    return Err(Error::internal(format!(
                        "Segment commit path does not support index type {}",
                        unsupported
                    )));
                }
            };
            let segments = self
                .dataset
                .create_index_segment_builder()
                .with_index_type(segment_index_type)
                .with_segments(vec![new_idx.clone()])
                .build_all()
                .await?;
            let new_indices =
                build_index_metadata_from_segments(self.dataset, &new_idx.name, field_id, segments)
                    .await?;
            TransactionBuilder::new(
                new_idx.dataset_version,
                Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                },
            )
            .transaction_properties(self.transaction_properties.clone())
            .build()
        } else {
            TransactionBuilder::new(
                new_idx.dataset_version,
                Operation::CreateIndex {
                    new_indices: vec![new_idx],
                    removed_indices,
                },
            )
            .transaction_properties(self.transaction_properties.clone())
            .build()
        };

        self.dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        // Fetch the committed index metadata from the dataset.
        // This ensures we return the version that may have been modified by the commit.
        let indices = self.dataset.load_indices().await?;
        indices
            .iter()
            .find(|idx| idx.uuid == index_uuid)
            .cloned()
            .ok_or_else(|| {
                Error::internal(format!(
                    "Index with UUID {} not found after commit",
                    index_uuid
                ))
            })
    }
}

fn uses_segment_commit_path(
    index_type: IndexType,
    index_name: &str,
    params: &dyn IndexParams,
) -> bool {
    if index_name != LANCE_VECTOR_INDEX {
        return false;
    }

    matches!(
        index_type,
        IndexType::Vector
            | IndexType::IvfPq
            | IndexType::IvfSq
            | IndexType::IvfFlat
            | IndexType::IvfRq
            | IndexType::IvfHnswFlat
            | IndexType::IvfHnswPq
            | IndexType::IvfHnswSq
    ) && params.as_any().is::<VectorIndexParams>()
}

impl<'a> IntoFuture for CreateIndexBuilder<'a> {
    type Output = Result<IndexMetadata>;
    type IntoFuture = BoxFuture<'a, Result<IndexMetadata>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

/// Build physical index segments from previously-written uncommitted index outputs.
///
/// Use [`DatasetIndexExt::create_index_segment_builder`] and then either:
///
/// - call [`Self::with_index_type`] with the concrete segment type first, then
/// - call [`Self::plan`] and orchestrate individual segment builds externally, or
/// - call [`Self::build_all`] to build all segments on the current node.
///
/// This builder only builds physical segments. Publishing those segments as
/// a logical index still requires [`DatasetIndexExt::commit_existing_index_segments`].
/// Together these two APIs form the canonical segment-based index build workflow.
#[derive(Clone)]
pub struct IndexSegmentBuilder<'a> {
    dataset: &'a Dataset,
    index_type: Option<IndexType>,
    segments: Vec<IndexMetadata>,
    target_segment_bytes: Option<u64>,
}

impl<'a> IndexSegmentBuilder<'a> {
    pub(crate) fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            index_type: None,
            segments: Vec::new(),
            target_segment_bytes: None,
        }
    }

    /// Declare the concrete index type of the staged segments.
    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = Some(index_type);
        self
    }

    /// Provide the segment metadata returned by `execute_uncommitted()`.
    ///
    /// These segments must already exist in storage and must not have been
    /// published into a logical index yet.
    pub fn with_segments(mut self, segments: Vec<IndexMetadata>) -> Self {
        self.segments = segments;
        self
    }

    /// Set the target size, in bytes, for merged physical segments.
    ///
    /// When set, input segments will be grouped into larger physical segments
    /// up to approximately this size. When unset, each input segment becomes
    /// one physical segment.
    pub fn with_target_segment_bytes(mut self, bytes: u64) -> Self {
        self.target_segment_bytes = Some(bytes);
        self
    }

    /// Plan how input segments should be grouped into physical segments.
    pub async fn plan(&self) -> Result<Vec<IndexSegmentPlan>> {
        if self.segments.is_empty() {
            return Err(Error::invalid_input(
                "IndexSegmentBuilder requires at least one segment; \
                 call with_segments(...) with execute_uncommitted() outputs"
                    .to_string(),
            ));
        }
        let index_type = self.index_type.ok_or_else(|| {
            Error::invalid_input(
                "IndexSegmentBuilder requires an explicit index type; call with_index_type(...)"
                    .to_string(),
            )
        })?;
        let mut seen_segment_ids = HashSet::with_capacity(self.segments.len());
        for segment in &self.segments {
            if !seen_segment_ids.insert(segment.uuid) {
                return Err(Error::invalid_input(format!(
                    "IndexSegmentBuilder received duplicate segment uuid {}",
                    segment.uuid
                )));
            }
        }

        match index_type {
            IndexType::Inverted => crate::index::scalar::inverted::plan_segments(
                &self.segments,
                self.target_segment_bytes,
            ),
            IndexType::Vector => {
                crate::index::vector::ivf::plan_segments(
                    &self.segments,
                    Some(index_type),
                    self.target_segment_bytes,
                )
                .await
            }
            IndexType::IvfFlat
            | IndexType::IvfPq
            | IndexType::IvfSq
            | IndexType::IvfRq
            | IndexType::IvfHnswFlat
            | IndexType::IvfHnswPq
            | IndexType::IvfHnswSq => {
                crate::index::vector::ivf::plan_segments(
                    &self.segments,
                    Some(index_type),
                    self.target_segment_bytes,
                )
                .await
            }
            unsupported => Err(Error::invalid_input(format!(
                "IndexSegmentBuilder does not support planning segments for index type {}",
                unsupported
            ))),
        }
    }

    /// Build one segment from a previously-generated plan.
    pub async fn build(&self, plan: &IndexSegmentPlan) -> Result<IndexSegment> {
        match plan.requested_index_type().ok_or_else(|| {
            Error::invalid_input(
                "IndexSegmentBuilder requires planned segments to declare an index type"
                    .to_string(),
            )
        })? {
            IndexType::Inverted => {
                crate::index::scalar::inverted::build_segment(self.dataset, plan).await
            }
            IndexType::Vector
            | IndexType::IvfFlat
            | IndexType::IvfPq
            | IndexType::IvfSq
            | IndexType::IvfRq
            | IndexType::IvfHnswFlat
            | IndexType::IvfHnswPq
            | IndexType::IvfHnswSq => {
                crate::index::vector::ivf::build_segment(
                    self.dataset.object_store.as_ref(),
                    &self.dataset.indices_dir(),
                    plan,
                )
                .await
            }
            unsupported => Err(Error::invalid_input(format!(
                "IndexSegmentBuilder does not support building segments for index type {}",
                unsupported
            ))),
        }
    }

    /// Plan and build all segments from the provided inputs.
    pub async fn build_all(&self) -> Result<Vec<IndexSegment>> {
        let plans = self.plan().await?;
        try_join_all(plans.iter().map(|plan| self.build(plan))).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::datatypes::{Float32Type, Int32Type};
    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatchIterator};
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{self, gen_batch};
    use lance_index::optimize::OptimizeOptions;
    use lance_index::progress::IndexBuildProgress;
    use lance_index::scalar::{FullTextSearchQuery, inverted::tokenizer::InvertedIndexParams};
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
    use lance_linalg::distance::{DistanceType, MetricType};
    use serde_json::json;
    use std::sync::Arc;
    use uuid::Uuid;

    lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

    #[test]
    fn test_inverted_training_params_include_build_only_fields() {
        let params = InvertedIndexParams::default()
            .memory_limit_mb(4096)
            .num_workers(7);
        let scalar_params = ScalarIndexParams::new("inverted".to_string())
            .with_params(&params.to_training_json().unwrap());
        let json: serde_json::Value =
            serde_json::from_str(scalar_params.params.as_ref().unwrap()).unwrap();
        assert_eq!(
            json.get("memory_limit"),
            Some(&serde_json::Value::from(4096))
        );
        assert_eq!(json.get("num_workers"), Some(&serde_json::Value::from(7)));
    }

    #[test]
    fn test_default_index_name() {
        // Single field - preserved as-is
        assert_eq!(default_index_name(&["user-id"]), "user-id");
        assert_eq!(default_index_name(&["user:id"]), "user:id");
        assert_eq!(default_index_name(&["userId"]), "userId");

        // Nested paths - joined with dot
        assert_eq!(
            default_index_name(&["meta-data", "user-id"]),
            "meta-data.user-id"
        );
        assert_eq!(
            default_index_name(&["MetaData", "userId"]),
            "MetaData.userId"
        );

        // Path with dots in field names - escape
        assert_eq!(
            default_index_name(&["meta.data", "user.id"]),
            "`meta.data`.`user.id`"
        );

        // Empty input
        assert_eq!(default_index_name(&[]), "");
    }

    #[tokio::test]
    async fn test_default_index_name_with_special_chars() {
        // Verify default index names preserve special characters in column names.
        let mut dataset = gen_batch()
            .col("user-id", lance_datagen::array::step::<Int32Type>())
            .col("user:id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);

        // Create index on column with hyphen
        let idx1 = CreateIndexBuilder::new(&mut dataset, &["user-id"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx1.name, "user-id_idx");

        // Create index on column with colon
        let idx2 = CreateIndexBuilder::new(&mut dataset, &["user:id"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx2.name, "user:id_idx");

        // Verify both indices exist
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
    }

    #[tokio::test]
    async fn test_index_name_collision_with_explicit_name() {
        // Test collision handling when explicit name conflicts with default name.
        let mut dataset = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .col("b", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);

        // (a) Explicit name on first index, default on second that would collide
        // Create index on "a" with explicit name "b_idx"
        let idx1 = CreateIndexBuilder::new(&mut dataset, &["a"], IndexType::BTree, &params)
            .name("b_idx".to_string())
            .execute()
            .await
            .unwrap();
        assert_eq!(idx1.name, "b_idx");

        // Create index on "b" with default name - would be "b_idx" but that's taken
        // so it should get "b_idx_2"
        let idx2 = CreateIndexBuilder::new(&mut dataset, &["b"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx2.name, "b_idx_2");

        // Verify both indices exist
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
    }

    #[tokio::test]
    async fn test_index_name_collision_explicit_errors() {
        // Test that explicit name collision with existing index errors.
        let mut dataset = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .col("b", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);

        // (b) Default name on first, explicit same name on second should error
        // Create index on "a" with default name "a_idx"
        let idx1 = CreateIndexBuilder::new(&mut dataset, &["a"], IndexType::BTree, &params)
            .execute()
            .await
            .unwrap();
        assert_eq!(idx1.name, "a_idx");

        // Try to create index on "b" with explicit name "a_idx" - should error
        let result = CreateIndexBuilder::new(&mut dataset, &["b"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn test_concurrent_create_index_same_name_returns_retryable_conflict() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());
        let reader = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(100),
                lance_datagen::BatchCount::from(1),
            );
        let dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let read_version = dataset.manifest.version;
        let mut reader1 = dataset.checkout_version(read_version).await.unwrap();
        let mut reader2 = dataset.checkout_version(read_version).await.unwrap();

        let first = CreateIndexBuilder::new(&mut reader1, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await;
        assert!(
            first.is_ok(),
            "first create_index should succeed: {first:?}"
        );

        let second = CreateIndexBuilder::new(&mut reader2, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await;
        assert!(
            matches!(second, Err(Error::RetryableCommitConflict { .. })),
            "second concurrent create_index should be retryable, got {second:?}"
        );

        let latest_indices = reader1.load_indices_by_name("a_idx").await.unwrap();
        assert_eq!(latest_indices.len(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_replace_index_same_name_returns_retryable_conflict() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());
        let reader = gen_batch()
            .col("a", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(100),
                lance_datagen::BatchCount::from(1),
            );
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let original = CreateIndexBuilder::new(&mut dataset, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .execute()
            .await
            .unwrap();

        let read_version = dataset.manifest.version;
        let mut reader1 = dataset.checkout_version(read_version).await.unwrap();
        let mut reader2 = dataset.checkout_version(read_version).await.unwrap();

        let replacement = CreateIndexBuilder::new(&mut reader1, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .replace(true)
            .execute()
            .await
            .unwrap();
        assert_ne!(replacement.uuid, original.uuid);

        let second = CreateIndexBuilder::new(&mut reader2, &["a"], IndexType::BTree, &params)
            .name("a_idx".to_string())
            .replace(true)
            .execute()
            .await;
        assert!(
            matches!(second, Err(Error::RetryableCommitConflict { .. })),
            "second concurrent replace should be retryable, got {second:?}"
        );

        let latest_indices = reader1.load_indices_by_name("a_idx").await.unwrap();
        assert_eq!(latest_indices.len(), 1);
        assert_eq!(latest_indices[0].uuid, replacement.uuid);
        assert_ne!(latest_indices[0].uuid, original.uuid);
    }

    // Helper function to create test data with text field suitable for inverted index
    fn create_text_batch(start: i32, end: i32) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Utf8, false),
        ]));
        let texts = (start..end)
            .map(|i| match i % 3 {
                0 => format!("document {} with some text content", i),
                1 => format!("another document {} containing different words", i),
                _ => format!("text sample {} for testing inverted index", i),
            })
            .collect::<Vec<_>>();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(start..end)),
                Arc::new(StringArray::from_iter_values(texts)),
            ],
        )
        .unwrap()
    }

    async fn prepare_vector_ivf(dataset: &Dataset, vector_column: &str) -> IvfBuildParams {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();
        let dim = vectors.value_length() as usize;
        let values = vectors.values().as_primitive::<Float32Type>();

        let kmeans = train_kmeans::<Float32Type>(
            values,
            KMeansParams::new(None, 10, 1, DistanceType::L2),
            dim,
            4,
            3,
        )
        .unwrap();
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                kmeans.centroids.as_primitive::<Float32Type>().clone(),
                dim as i32,
            )
            .unwrap(),
        );
        IvfBuildParams::try_with_centroids(4, centroids).unwrap()
    }

    #[tokio::test]
    async fn test_execute_uncommitted() {
        // Test the complete workflow that covers the user's specified code pattern:
        // 1. Create dataset with multiple fragments
        // 2. Get fragment IDs from dataset using dataset.get_fragments()
        // 3. Create CreateIndexBuilder with fragments() method
        // 4. Call execute_uncommitted() to get IndexMetadata
        // 5. Verify IndexMetadata contains correct fragment_bitmap

        // Create temporary directory for dataset
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // Create test data with multiple fragments
        let batch1 = create_text_batch(0, 10);
        let batch2 = create_text_batch(10, 20);
        let batch3 = create_text_batch(20, 30);

        let write_params = WriteParams {
            max_rows_per_file: 10, // Force multiple fragments
            max_rows_per_group: 5,
            ..Default::default()
        };

        // Write dataset with multiple batches to create multiple fragments
        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params))
            .await
            .unwrap();

        let params = InvertedIndexParams::default();

        // Get fragment IDs from the dataset
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        assert!(
            fragment_ids.len() >= 2,
            "Should have multiple fragments for testing"
        );

        // Test fragments() method with specific fragment IDs and ensure duplicate/out-of-order fragments are handled properly
        let selected_fragments = vec![
            fragment_ids[1],
            fragment_ids[0],
            fragment_ids[1],
            fragment_ids[2],
        ];
        let selected_fragments_expected = vec![fragment_ids[0], fragment_ids[1], fragment_ids[2]];

        let mut builder =
            CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                .name("fragment_index".to_string())
                .fragments(selected_fragments.clone());

        // Execute uncommitted to get index metadata
        let index_metadata = builder.execute_uncommitted().await.unwrap();

        // Verify the IndexMetadata contains the correct fragment_bitmap
        let fragment_bitmap = index_metadata.fragment_bitmap.unwrap();
        let indexed_fragments: Vec<u32> = fragment_bitmap.iter().collect();
        assert_eq!(
            indexed_fragments, selected_fragments_expected,
            "Index should only cover the selected fragments"
        );

        // Verify other metadata fields
        assert_eq!(index_metadata.name, "fragment_index");
        assert!(!index_metadata.uuid.is_nil());
        assert!(index_metadata.created_at.is_some());
    }

    #[tokio::test]
    async fn test_merge_index_metadata_inverted_reports_progress() {
        // This exercises the public distributed inverted-index workflow end to end:
        // 1. build one uncommitted shard per fragment with CreateIndexBuilder.progress(...)
        // 2. merge those shards with Dataset::merge_index_metadata(...)
        //
        // Expected outcomes:
        // - the build callback should surface public build stages such as load_data,
        //   tokenize_docs, copy_partitions, and write_metadata
        // - the merge callback should surface public merge stages such as
        //   read_partition_metadata, remap_partition_files, and write_merged_metadata
        // - merge stages should be reported in execution order
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batch1 = create_text_batch(0, 15);
        let batch2 = create_text_batch(15, 30);
        let batch3 = create_text_batch(30, 45);

        let write_params = WriteParams {
            max_rows_per_file: 15,
            max_rows_per_group: 5,
            ..Default::default()
        };

        // Write dataset with multiple batches to create multiple fragments
        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params))
            .await
            .unwrap();

        let params = InvertedIndexParams::default();
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        let shared_uuid = Uuid::new_v4().to_string();
        let build_progress = Arc::new(RecordingProgress::default());

        for &fragment_id in &fragment_ids {
            let mut builder =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("distributed_index".to_string())
                    .fragments(vec![fragment_id])
                    .index_uuid(shared_uuid.clone())
                    .progress(build_progress.clone());

            let index_metadata = builder.execute_uncommitted().await.unwrap();
            assert_eq!(index_metadata.uuid.to_string(), shared_uuid);
            assert_eq!(index_metadata.name, "distributed_index");

            let fragment_bitmap = index_metadata.fragment_bitmap.as_ref().unwrap();
            let indexed_fragments: Vec<u32> = fragment_bitmap.iter().collect();
            assert_eq!(indexed_fragments, vec![fragment_id]);
        }

        let merge_progress = Arc::new(RecordingProgress::default());
        dataset
            .merge_index_metadata(
                &shared_uuid,
                IndexType::Inverted,
                None,
                merge_progress.clone(),
            )
            .await
            .unwrap();

        let build_tags = build_progress
            .recorded_events()
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        assert!(
            build_tags.iter().any(|e| e == "start:load_data"),
            "expected load_data progress during public distributed build"
        );
        assert!(
            build_tags.iter().any(|e| e == "start:tokenize_docs"),
            "expected tokenize_docs progress during public distributed build"
        );
        assert!(
            build_tags.iter().any(|e| e == "start:copy_partitions"),
            "expected copy_partitions progress during public distributed build"
        );
        assert!(
            build_tags.iter().any(|e| e == "start:write_metadata"),
            "expected write_metadata progress during public distributed build"
        );

        let merge_events = merge_progress.recorded_events();
        let merge_tags = merge_events
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        let read_start = merge_tags
            .iter()
            .position(|e| e == "start:read_partition_metadata")
            .expect("missing read_partition_metadata start");
        let read_complete = merge_tags
            .iter()
            .position(|e| e == "complete:read_partition_metadata")
            .expect("missing read_partition_metadata complete");
        let remap_start = merge_tags
            .iter()
            .position(|e| e == "start:remap_partition_files")
            .expect("missing remap_partition_files start");
        let remap_complete = merge_tags
            .iter()
            .position(|e| e == "complete:remap_partition_files")
            .expect("missing remap_partition_files complete");
        let metadata_start = merge_tags
            .iter()
            .position(|e| e == "start:write_merged_metadata")
            .expect("missing write_merged_metadata start");
        let metadata_complete = merge_tags
            .iter()
            .position(|e| e == "complete:write_merged_metadata")
            .expect("missing write_merged_metadata complete");
        assert!(read_start < read_complete);
        assert!(read_complete < remap_start);
        assert!(remap_start < remap_complete);
        assert!(remap_complete < metadata_start);
        assert!(metadata_start < metadata_complete);
        assert!(
            merge_tags
                .iter()
                .any(|e| e == "progress:read_partition_metadata"),
            "expected read_partition_metadata progress during public merge"
        );
        assert!(
            merge_tags
                .iter()
                .any(|e| e == "progress:remap_partition_files"),
            "expected remap_partition_files progress during public merge"
        );
        assert!(
            merge_tags
                .iter()
                .any(|e| e == "progress:write_merged_metadata"),
            "expected write_merged_metadata progress during public merge"
        );
    }

    #[tokio::test]
    async fn test_merge_index_metadata_btree_reports_progress() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        let shared_uuid = Uuid::new_v4().to_string();
        let build_progress = Arc::new(RecordingProgress::default());

        for &fragment_id in &fragment_ids {
            CreateIndexBuilder::new(&mut dataset, &["id"], IndexType::BTree, &params)
                .name("distributed_btree".to_string())
                .fragments(vec![fragment_id])
                .index_uuid(shared_uuid.clone())
                .progress(build_progress.clone())
                .execute_uncommitted()
                .await
                .unwrap();
        }

        let merge_progress = Arc::new(RecordingProgress::default());
        dataset
            .merge_index_metadata(
                &shared_uuid,
                IndexType::BTree,
                Some(1),
                merge_progress.clone(),
            )
            .await
            .unwrap();

        let build_tags = build_progress
            .recorded_events()
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        assert!(
            build_tags.iter().any(|e| e == "start:load_data"),
            "expected load_data progress during public distributed build"
        );

        let merge_tags = merge_progress
            .recorded_events()
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        let pages_start = merge_tags
            .iter()
            .position(|e| e == "start:merge_pages")
            .expect("missing merge_pages start");
        let pages_complete = merge_tags
            .iter()
            .position(|e| e == "complete:merge_pages")
            .expect("missing merge_pages complete");
        let write_start = merge_tags
            .iter()
            .position(|e| e == "start:write_lookup_file")
            .expect("missing write_lookup_file start");
        let write_complete = merge_tags
            .iter()
            .position(|e| e == "complete:write_lookup_file")
            .expect("missing write_lookup_file complete");
        assert!(pages_start < pages_complete);
        assert!(pages_complete < write_start);
        assert!(write_start < write_complete);
        assert!(
            merge_tags.iter().any(|e| e == "progress:merge_pages"),
            "expected merge_pages progress during public merge"
        );
        assert!(
            merge_tags.iter().any(|e| e == "progress:write_lookup_file"),
            "expected write_lookup_file progress during public merge"
        );
        assert!(
            !merge_tags.iter().any(|e| e == "start:merge_lookups"),
            "fragment-based distributed BTREE merge should not use merge_lookups"
        );
    }

    #[tokio::test]
    async fn test_distributed_build_bitmap() {
        use datafusion::common::ScalarValue;
        use lance_index::scalar::{SargableQuery, SearchResult, bitmap::BITMAP_LOOKUP_NAME};
        use lance_select::RowSetOps;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "category",
            DataType::Int32,
            false,
        )]));
        let batches = (0..4)
            .map(
                |fragment_id| -> std::result::Result<_, arrow_schema::ArrowError> {
                    let values = vec![fragment_id, fragment_id, fragment_id + 10, fragment_id + 10];
                    Ok(RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(values))],
                    )
                    .unwrap())
                },
            )
            .collect::<Vec<_>>();
        let reader = RecordBatchIterator::new(batches.into_iter(), schema);

        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let base_params =
            ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::Bitmap);
        let fragments = dataset.get_fragments();
        let fragment_ids: Vec<u32> = fragments.iter().map(|f| f.id() as u32).collect();
        let shared_uuid = Uuid::new_v4().to_string();
        let mut shard_metadata = None;
        let shard_groups = fragment_ids.chunks(2).collect::<Vec<_>>();

        for (shard_id, fragment_group) in shard_groups.iter().enumerate() {
            let params = base_params
                .clone()
                .with_params(&json!({ "shard_id": shard_id as u32 }));
            let index_metadata =
                CreateIndexBuilder::new(&mut dataset, &["category"], IndexType::Bitmap, &params)
                    .name("distributed_bitmap".to_string())
                    .fragments(fragment_group.to_vec())
                    .index_uuid(shared_uuid.clone())
                    .execute_uncommitted()
                    .await
                    .unwrap();
            if shard_metadata.is_none() {
                shard_metadata = Some(index_metadata);
            }
        }

        dataset
            .merge_index_metadata(
                &shared_uuid,
                IndexType::Bitmap,
                None,
                Arc::new(NoopIndexBuildProgress),
            )
            .await
            .unwrap();

        let mut committed_index_metadata = shard_metadata.unwrap();
        committed_index_metadata.fragment_bitmap = Some(fragment_ids.iter().copied().collect());
        committed_index_metadata.files = Some(
            list_index_files_with_sizes(
                dataset.object_store.as_ref(),
                &dataset.indices_dir().clone().join(shared_uuid.clone()),
            )
            .await
            .unwrap(),
        );
        committed_index_metadata.dataset_version = dataset.manifest.version;

        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![committed_index_metadata],
                removed_indices: vec![],
            },
        )
        .build();
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let dataset = Dataset::open(&dataset_uri).await.unwrap();
        let indices = dataset
            .load_indices_by_name("distributed_bitmap")
            .await
            .unwrap();
        assert_eq!(indices.len(), 1);
        let index = &indices[0];
        assert_eq!(
            index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            fragment_ids
        );

        let files = index.files.as_ref().unwrap();
        assert!(files.iter().any(|file| file.path == BITMAP_LOOKUP_NAME));
        assert!(
            files.iter().all(|file| !file.path.starts_with("part_")),
            "committed bitmap index should only reference merged files"
        );

        let scalar_index = crate::index::scalar::open_scalar_index(
            &dataset,
            "category",
            index,
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();
        assert_eq!(scalar_index.index_type(), IndexType::Bitmap);

        let query_result = scalar_index
            .search(
                &SargableQuery::Equals(ScalarValue::Int32(Some(2))),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let SearchResult::Exact(query_rows) = query_result else {
            panic!("expected exact bitmap result");
        };
        assert_eq!(query_rows.true_rows().len(), Some(2));
    }

    #[tokio::test]
    async fn test_vector_execute_uncommitted_segments_commit_without_staging() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );
        let mut input_segments = Vec::new();

        for fragment in &fragments {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                    .name("vector_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            let segment_index = dataset
                .indices_dir()
                .clone()
                .join(segment.uuid.to_string())
                .join(crate::index::INDEX_FILE_NAME);
            assert!(
                dataset
                    .object_store
                    .as_ref()
                    .exists(&segment_index)
                    .await
                    .unwrap()
            );
            input_segments.push(segment);
        }

        let segments = dataset
            .create_index_segment_builder()
            .with_index_type(params.index_type())
            .with_segments(input_segments.clone())
            .build_all()
            .await
            .unwrap();
        assert_eq!(segments.len(), fragments.len());
        let mut built_segment_ids = segments
            .iter()
            .map(|segment| segment.uuid())
            .collect::<Vec<_>>();
        built_segment_ids.sort();
        let mut input_segment_ids = input_segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        input_segment_ids.sort();
        assert_eq!(built_segment_ids, input_segment_ids);

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), fragments.len());

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_index_segment_builder_vector_commits_multi_segment_logical_index() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );
        let mut input_segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                    .name("vector_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            input_segments.push(segment);
        }

        let segments = dataset
            .create_index_segment_builder()
            .with_index_type(params.index_type())
            .with_segments(input_segments)
            .build_all()
            .await
            .unwrap();
        assert_eq!(segments.len(), 2);

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 2);
        let mut committed_fragment_sets = indices
            .iter()
            .map(|metadata| {
                metadata
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        committed_fragment_sets.sort();
        assert_eq!(committed_fragment_sets, vec![vec![0], vec![1]]);

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_index_segment_builder_vector_segments_without_index_details() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(256),
                lance_datagen::BatchCount::from(4),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );
        let mut input_segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let mut segment =
                CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                    .name("vector_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            segment.index_details = None;
            input_segments.push(segment);
        }

        let segments = dataset
            .create_index_segment_builder()
            .with_index_type(params.index_type())
            .with_segments(input_segments)
            .build_all()
            .await
            .unwrap();
        assert_eq!(segments.len(), 2);
    }

    #[tokio::test]
    async fn test_index_segment_builder_fts_commits_multi_segment_logical_index() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batch1 = create_text_batch(0, 10);
        let batch2 = create_text_batch(10, 20);
        let batch3 = create_text_batch(20, 30);

        let batches = RecordBatchIterator::new(
            vec![Ok(batch1), Ok(batch2), Ok(batch3)],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 5,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let mut input_segments = Vec::new();
        for fragment in dataset.get_fragments() {
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("text_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            input_segments.push(segment);
        }

        let segments = dataset
            .create_index_segment_builder()
            .with_index_type(IndexType::Inverted)
            .with_segments(input_segments.clone())
            .build_all()
            .await
            .unwrap();
        assert_eq!(segments.len(), input_segments.len());

        for segment in &segments {
            let metadata_path = dataset
                .indices_dir()
                .clone()
                .join(segment.uuid().to_string())
                .join(lance_index::scalar::inverted::METADATA_FILE);
            assert!(
                dataset
                    .object_store
                    .as_ref()
                    .exists(&metadata_path)
                    .await
                    .unwrap()
            );
        }

        dataset
            .commit_existing_index_segments("text_idx", "text", segments)
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(indices.len(), input_segments.len());
    }

    #[tokio::test]
    async fn test_merge_existing_index_segments_supports_fts_segments() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batches = RecordBatchIterator::new(
            vec![
                Ok(create_text_batch(0, 10)),
                Ok(create_text_batch(10, 20)),
                Ok(create_text_batch(20, 30)),
            ],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                max_rows_per_group: 5,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let mut input_segments = Vec::new();
        let mut expected_fragments = roaring::RoaringBitmap::new();
        for fragment in dataset.get_fragments() {
            expected_fragments.insert(fragment.id() as u32);
            let segment =
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                    .name("text_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap();
            input_segments.push(segment);
        }

        let merged = dataset
            .merge_existing_index_segments(input_segments)
            .await
            .unwrap();
        assert_eq!(
            merged
                .fragment_bitmap
                .as_ref()
                .expect("merged FTS segment should have fragment coverage"),
            &expected_fragments
        );
        assert!(
            merged
                .index_details
                .as_ref()
                .expect("merged FTS segment should have index details")
                .type_url
                .ends_with("InvertedIndexDetails")
        );

        dataset
            .commit_existing_index_segments("text_idx", "text", vec![merged])
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(indices.len(), 1);

        let results = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("document".to_string()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 20);
    }

    #[tokio::test]
    async fn test_index_segment_builder_rejects_duplicate_segment_uuids() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batches = RecordBatchIterator::new(
            vec![Ok(create_text_batch(0, 10))],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let segment =
            CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                .name("text_idx".to_string())
                .fragments(vec![0])
                .execute_uncommitted()
                .await
                .unwrap();

        let err = dataset
            .create_index_segment_builder()
            .with_index_type(IndexType::Inverted)
            .with_segments(vec![segment.clone(), segment])
            .build_all()
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("duplicate segment uuid"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_index_segment_builder_requires_explicit_index_type() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batches = RecordBatchIterator::new(
            vec![Ok(create_text_batch(0, 10))],
            create_text_batch(0, 1).schema(),
        );
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let segment =
            CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Inverted, &params)
                .name("text_idx".to_string())
                .fragments(vec![0])
                .execute_uncommitted()
                .await
                .unwrap();

        let err = dataset
            .create_index_segment_builder()
            .with_segments(vec![segment])
            .plan()
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("requires an explicit index type"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_index_segment_builder_requires_requested_index_type() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let batches = RecordBatchIterator::new(
            vec![Ok(create_text_batch(0, 10))],
            create_text_batch(0, 1).schema(),
        );
        let dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 10,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let segment = IndexSegment::new(
            Uuid::new_v4(),
            [0_u32],
            Arc::new(prost_types::Any::default()),
            0,
        );
        let plan = IndexSegmentPlan::new(segment, Vec::new(), 0, None);
        let err = dataset
            .create_index_segment_builder()
            .with_index_type(IndexType::Inverted)
            .build(&plan)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("declare an index type"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_commit_existing_index_supports_local_hnsw_segments() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(128),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(
            reader,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let uuid = Uuid::new_v4();
        let params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
            HnswBuildParams::default(),
        );

        CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .index_uuid(uuid.to_string())
            .execute_uncommitted()
            .await
            .unwrap();

        dataset
            .commit_existing_index_segments(
                "vector_idx",
                "vector",
                vec![IndexSegment::new(
                    uuid,
                    dataset.fragment_bitmap.as_ref().clone(),
                    Arc::new(vector_index_details(&params)),
                    IndexType::IvfHnswFlat.version(),
                )],
            )
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, uuid);
        assert_eq!(
            indices[0].fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
    }

    #[tokio::test]
    async fn test_create_index_vector_commits_with_segment_metadata() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(128),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_vector_ivf(&dataset, "vector").await,
        );

        let committed = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        assert!(
            committed
                .files
                .as_ref()
                .is_some_and(|files| !files.is_empty()),
            "single-machine vector create_index should preserve committed file info"
        );

        let loaded = dataset.load_indices_by_name(&committed.name).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].uuid, committed.uuid);
        assert!(
            loaded[0]
                .files
                .as_ref()
                .is_some_and(|files| !files.is_empty()),
            "committed metadata loaded from the manifest should include file info"
        );
    }

    #[tokio::test]
    async fn test_create_index_ivf_rq_preserves_index_version_on_segment_commit_path() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let reader = gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(128),
                lance_datagen::BatchCount::from(2),
            );
        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_rq(4, 1, DistanceType::L2);

        let committed = dataset
            .create_index(&["vector"], IndexType::IvfRq, None, &params, false)
            .await
            .unwrap();

        assert_eq!(committed.index_version, IndexType::IvfRq.version());

        let loaded = dataset.load_indices_by_name(&committed.name).await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].index_version, IndexType::IvfRq.version());
    }

    #[tokio::test]
    async fn test_optimize_should_not_removes_delta_indices() {
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let num_rows = 256;
        let reader = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(num_rows),
                lance_datagen::BatchCount::from(1),
            );

        let mut dataset = Dataset::write(reader, &dataset_uri, None).await.unwrap();

        let vector_params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None, // Will auto-generate name "vector_idx"
                &vector_params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "Should have 1 index");
        assert_eq!(indices[0].name, "vector_idx");
        assert_eq!(indices[0].fragment_bitmap.as_ref().unwrap().len(), 1);
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().contains(0));

        // create again with replace=false
        let res = dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None, // Will auto-generate name "vector_idx"
                &vector_params,
                false,
            )
            .await;
        assert!(res.is_err());

        // create again with replace=true
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None, // Will auto-generate name "vector_idx"
                &vector_params,
                true,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "Should have 1 index");
        assert_eq!(indices[0].name, "vector_idx");
        assert_eq!(indices[0].fragment_bitmap.as_ref().unwrap().len(), 1);
        assert!(indices[0].fragment_bitmap.as_ref().unwrap().contains(0));

        let scalar_params =
            ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                None, // Will auto-generate name "id_idx"
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2, "Should have 2 indices");

        let num_new_rows = 32;
        let new_reader = lance_datagen::gen_batch()
            .col(
                "id",
                lance_datagen::array::step_custom::<Int32Type>(num_rows as i32, 1),
            )
            .col(
                "vector",
                lance_datagen::array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(16)),
            )
            .into_reader_rows(
                lance_datagen::RowCount::from(num_new_rows),
                lance_datagen::BatchCount::from(1),
            );

        dataset = Dataset::write(
            new_reader,
            &dataset_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Load indices before optimization
        let indices_before = dataset.load_indices().await.unwrap();
        assert_eq!(indices_before.len(), 2, "Should still have 2 indices");

        // Optimize with num_indices_to_merge=0
        let optimize_options = OptimizeOptions::append();
        dataset.optimize_indices(&optimize_options).await.unwrap();

        // Load indices after optimization
        let indices_after = dataset.load_indices().await.unwrap();

        // There should be 3 indices:
        // 1. one scalar index with name "id_idx", and the bitmap is [0,1]
        // 2. one delta vector index with name "vector_idx", and the bitmap is [0]
        // 3. one delta vector index with name "vector_idx", and the bitmap is [1]
        assert_eq!(indices_after.len(), 3, "{:?}", indices_after);
        let id_idx = indices_after
            .iter()
            .find(|idx| idx.name == "id_idx")
            .unwrap();
        let vector_indices = indices_after
            .iter()
            .filter(|idx| idx.name == "vector_idx")
            .collect::<Vec<_>>();
        assert!(
            id_idx
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .contains_range(0..2)
                && id_idx.fragment_bitmap.as_ref().unwrap().len() == 2
        );
        assert_eq!(vector_indices.len(), 2);
        assert!(
            vector_indices
                .iter()
                .any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(0)
                    && idx.fragment_bitmap.as_ref().unwrap().len() == 1)
        );
        assert!(
            vector_indices
                .iter()
                .any(|idx| idx.fragment_bitmap.as_ref().unwrap().contains(1)
                    && idx.fragment_bitmap.as_ref().unwrap().len() == 1)
        );
    }

    // ========================================================================
    // Compound (multi-column) scalar index integration tests
    //
    // These exercise the integration layer:
    //   CreateIndexBuilder::execute_uncommitted (multi-column branch)
    //     -> build_compound_btree_index
    //     -> CompoundBTreeIndexPlugin
    //     -> manifest persistence
    //     -> infer_scalar_index_details (CompoundBTreeIndexDetails)
    //     -> ScalarIndexInfo::get_compound_index (planner path)
    //     -> remap_index path (compaction + frag-reuse)
    // ========================================================================

    #[tokio::test]
    async fn test_create_compound_index() {
        use crate::Dataset;
        use crate::index::DatasetIndexInternalExt;
        use lance_index::scalar::ScalarIndexParams;

        // Use a fresh temp dir; file:// URI form so writer + reader both round-trip.
        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let tenant_ids: Vec<&str> = (0..100)
            .map(|i| match i % 3 {
                0 => "acme",
                1 => "globex",
                _ => "initech",
            })
            .collect();
        let statuses: Vec<&str> = (0..100)
            .map(|i| if i % 2 == 0 { "active" } else { "inactive" })
            .collect();
        let values: Vec<i32> = (0..100).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(tenant_ids)),
                Arc::new(StringArray::from(statuses)),
                Arc::new(Int32Array::from(values)),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 50,
            max_rows_per_group: 25,
            ..Default::default()
        };

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params))
            .await
            .unwrap();

        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("compound_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);

        let compound_idx = &indices[0];
        assert_eq!(compound_idx.name, "compound_idx");
        assert_eq!(
            compound_idx.fields.len(),
            2,
            "Compound index should have 2 fields"
        );

        // Verify the details type URL — confirms execute_uncommitted dispatched
        // to build_compound_btree_index and that infer_scalar_index_details
        // returned CompoundBTreeIndexDetails.
        let index_details = compound_idx
            .index_details
            .as_ref()
            .expect("should have details");
        assert!(
            index_details.type_url.contains("CompoundBTreeIndexDetails"),
            "Index details should be CompoundBTreeIndexDetails, got: {}",
            index_details.type_url
        );

        // Smoke-test open_scalar_index_by_name: it should resolve to the same
        // compound index we just created. Returns Some(index) on hit, None on
        // miss; this verifies the trait method on DatasetIndexExt wires
        // through to the internal open path.
        let by_name = dataset
            .open_scalar_index_by_name("tenant_id", "compound_idx")
            .await
            .unwrap();
        assert!(by_name.is_some(), "open_scalar_index_by_name should find compound_idx");

        let missing = dataset
            .open_scalar_index_by_name("tenant_id", "no_such_index")
            .await
            .unwrap();
        assert!(missing.is_none(), "open_scalar_index_by_name should return None for unknown name");

        // Cross-check that the same uuid reads correctly through the internal
        // open_scalar_index path used by query execution, and pin the
        // reported `index_type()` so this contract can't drift silently.
        //
        // NOTE: compound BTree currently reports `IndexType::Scalar` (the
        // legacy generic alias, see `crate::IndexType` enum: `Scalar = 0`
        // commented as "Legacy scalar index, alias to BTree"). Every other
        // scalar index plugin returns its specific variant (BTree → BTree,
        // Bitmap → Bitmap, ZoneMap → ZoneMap, …), so this is an asymmetry
        // worth knowing about: a downstream `match` on `IndexType::BTree`
        // will skip the compound index. There is no `IndexType::CompoundBTree`
        // variant today; introducing one would affect the on-disk wire
        // format. Until that's addressed, callers should treat `Scalar` from
        // an opened multi-column index as "compound BTree".
        let scalar_index = dataset
            .open_scalar_index(
                "tenant_id",
                &compound_idx.uuid.to_string(),
                &lance_index::metrics::NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert_eq!(
            scalar_index.index_type(),
            IndexType::Scalar,
            "compound BTree reports `Scalar` today; if this changes (e.g. \
             a new `IndexType::CompoundBTree` variant is added), update the \
             callers that pattern-match on the index type"
        );
    }

    #[tokio::test]
    async fn test_compound_index_rejects_unsupported_types() {
        use crate::Dataset;
        use crate::index::vector::VectorIndexParams;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("col1", DataType::Utf8, false),
            ArrowField::new("col2", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, &dataset_uri, None).await.unwrap();

        // Vector index type with multi-column input must error out before
        // hitting the compound branch (vector indices are inherently single-column).
        let params = VectorIndexParams::ivf_flat(8, MetricType::Cosine);
        let result = dataset
            .create_index(
                &["col1", "col2"],
                IndexType::Vector,
                None,
                &params,
                false,
            )
            .await;

        assert!(
            result.is_err(),
            "Vector index should not support multiple columns"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("does not support multiple columns"),
            "Error should mention multi-column not supported: {}",
            err
        );
    }

    /// End-to-end search correctness via CompoundSargableQuery::PrefixLookup.
    ///
    /// This exercises the search path that compound_btree.rs unit tests cannot
    /// reach — going through create_index, manifest persistence, and reopening
    /// via open_scalar_index. Any divergence between the index data layout the
    /// trainer writes and the layout the reader expects shows up here.
    #[tokio::test]
    async fn test_compound_index_search_cartesian_product() {
        use crate::Dataset;
        use crate::index::DatasetIndexInternalExt;
        use datafusion::common::ScalarValue;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::scalar::compound::CompoundSargableQuery;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        // 6 rows: cartesian product of {acme, beta, gamma} × {active, inactive}
        // with row index inferred from row order:
        //   row 0: (acme, active, 100)
        //   row 1: (acme, active, 200)
        //   row 2: (acme, inactive, 300)
        //   row 3: (beta, active, 400)
        //   row 4: (beta, inactive, 500)
        //   row 5: (gamma, active, 600)
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    "acme", "acme", "acme", "beta", "beta", "gamma",
                ])),
                Arc::new(StringArray::from(vec![
                    "active", "active", "inactive", "active", "inactive", "active",
                ])),
                Arc::new(Int32Array::from(vec![100, 200, 300, 400, 500, 600])),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut dataset = Dataset::write(batches, &dataset_uri, None).await.unwrap();

        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("idx_tenant_status".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let compound_idx = &indices[0];
        let scalar_index = dataset
            .open_scalar_index(
                "tenant_id",
                &compound_idx.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Helper: extract returned row_ids from a SearchResult.
        let collect_row_ids = |result: lance_index::scalar::SearchResult| -> Vec<u64> {
            result
                .row_addrs()
                .true_rows()
                .row_addrs()
                .map(|iter| iter.map(u64::from).collect())
                .unwrap_or_default()
        };

        // (acme, active) -> rows 0, 1
        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("acme".into())),
                ScalarValue::Utf8(Some("active".into())),
            ],
            range: None,
        };
        let r = scalar_index.search(&q, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(collect_row_ids(r).len(), 2, "(acme, active) -> 2 rows");

        // (acme, inactive) -> row 2
        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("acme".into())),
                ScalarValue::Utf8(Some("inactive".into())),
            ],
            range: None,
        };
        let r = scalar_index.search(&q, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(collect_row_ids(r).len(), 1, "(acme, inactive) -> 1 row");

        // (beta, active) -> row 3
        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".into())),
                ScalarValue::Utf8(Some("active".into())),
            ],
            range: None,
        };
        let r = scalar_index.search(&q, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(collect_row_ids(r).len(), 1, "(beta, active) -> 1 row");

        // (beta, inactive) -> row 4
        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".into())),
                ScalarValue::Utf8(Some("inactive".into())),
            ],
            range: None,
        };
        let r = scalar_index.search(&q, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(collect_row_ids(r).len(), 1, "(beta, inactive) -> 1 row");

        // Prefix-only on first column: acme -> rows 0, 1, 2
        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![ScalarValue::Utf8(Some("acme".into()))],
            range: None,
        };
        let r = scalar_index.search(&q, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(collect_row_ids(r).len(), 3, "prefix (acme) -> 3 rows");
    }

    /// Compaction with defer_index_remap=true uses the fragment-reuse path.
    /// This is the path I changed when generalising
    /// `remap_row_ids_record_batch` from 2 columns to N columns. Without this
    /// test we have no end-to-end coverage that the compound index correctly
    /// follows row identity through a compaction.
    #[tokio::test]
    async fn test_compound_index_with_compaction_and_fragment_reuse() {
        use crate::Dataset;
        use crate::dataset::optimize::{CompactionOptions, compact_files};
        use crate::index::DatasetIndexInternalExt;
        use datafusion::common::ScalarValue;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::scalar::compound::CompoundSargableQuery;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["acme", "acme", "beta"])),
                Arc::new(StringArray::from(vec!["active", "inactive", "active"])),
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 3,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        Dataset::write(batches, &dataset_uri, Some(write_params.clone()))
            .await
            .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["beta", "gamma", "gamma"])),
                Arc::new(StringArray::from(vec!["inactive", "active", "inactive"])),
                Arc::new(Int32Array::from(vec![400, 500, 600])),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch2)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.fragments().len(), 2, "two fragments before compaction");

        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("idx_compound".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let compound_idx = indices.iter().find(|i| i.name == "idx_compound").unwrap();
        assert_eq!(compound_idx.fields.len(), 2);

        // Query before compaction.
        let scalar_index = dataset
            .open_scalar_index(
                "tenant_id",
                &compound_idx.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".into())),
                ScalarValue::Utf8(Some("active".into())),
            ],
            range: None,
        };
        let result_before = scalar_index.search(&q, &NoOpMetricsCollector).await.unwrap();
        let row_ids_before: Vec<u64> = result_before
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        assert_eq!(
            row_ids_before.len(),
            1,
            "(beta, active) -> 1 row before compaction"
        );

        // Compact with defer_index_remap=true; this triggers the fragment-reuse
        // remap path that calls remap_row_ids_record_batch on the compound
        // index's (col0, col1, ..., row_id) schema. Pre-fix this would have
        // asserted the schema had exactly 2 columns; post-fix it must work for
        // any number of columns.
        let compaction_options = CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        };
        compact_files(&mut dataset, compaction_options, None)
            .await
            .unwrap();

        dataset = Dataset::open(&dataset_uri).await.unwrap();
        assert_eq!(dataset.fragments().len(), 1, "one fragment after compaction");

        let indices_after = dataset.load_indices().await.unwrap();
        let compound_idx_after = indices_after
            .iter()
            .find(|idx| idx.name == "idx_compound")
            .expect("compound index should survive compaction");
        let scalar_index_after = dataset
            .open_scalar_index(
                "tenant_id",
                &compound_idx_after.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let q = CompoundSargableQuery::PrefixLookup {
            prefix: vec![
                ScalarValue::Utf8(Some("beta".into())),
                ScalarValue::Utf8(Some("active".into())),
            ],
            range: None,
        };
        let result_after = scalar_index_after
            .search(&q, &NoOpMetricsCollector)
            .await
            .unwrap();
        let row_ids_after: Vec<u64> = result_after
            .row_addrs()
            .true_rows()
            .row_addrs()
            .map(|iter| iter.map(u64::from).collect())
            .unwrap_or_default();
        assert_eq!(
            row_ids_after.len(),
            1,
            "(beta, active) -> still 1 row after compaction via fragment reuse"
        );

        // Final verify: row identity preserved through compaction.
        let projection = crate::dataset::ProjectionRequest::from_columns(
            ["tenant_id", "status", "value"],
            dataset.schema(),
        );
        let fetched = dataset.take_rows(&row_ids_after, projection).await.unwrap();
        assert_eq!(fetched.num_rows(), 1);
        let tenant_col = fetched
            .column_by_name("tenant_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(tenant_col.value(0), "beta");
        let status_col = fetched
            .column_by_name("status")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(status_col.value(0), "active");
    }

    /// Compound index with partial fragment coverage exercises the planner's
    /// compound path: `IndexInformationProvider::get_compound_index` lookup,
    /// `maybe_compound_prefix` matcher, and row-id resolution for the indexed
    /// fragment + scan fallback for the unindexed fragment.
    #[tokio::test]
    async fn test_compound_index_mixed_fragment_coverage() {
        use crate::Dataset;
        use lance_index::scalar::ScalarIndexParams;

        let tmpdir = TempStrDir::default();
        let dataset_uri = format!("file://{}", tmpdir.as_str());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("tenant_id", DataType::Utf8, false),
            ArrowField::new("status", DataType::Utf8, false),
            ArrowField::new("value", DataType::Int32, false),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["acme", "acme", "beta"])),
                Arc::new(StringArray::from(vec!["active", "inactive", "active"])),
                Arc::new(Int32Array::from(vec![100, 200, 300])),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 3,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new(vec![Ok(batch1)], schema.clone());
        let mut dataset = Dataset::write(batches, &dataset_uri, Some(write_params))
            .await
            .unwrap();

        // Index covers fragment 0 only.
        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["tenant_id", "status"],
                IndexType::BTree,
                Some("idx_compound".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let frag_bitmap = indices[0].fragment_bitmap.as_ref().unwrap();
        assert!(frag_bitmap.contains(0), "index covers fragment 0");
        assert_eq!(frag_bitmap.len(), 1, "index covers only 1 fragment");

        // Append a second fragment without re-indexing.
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["beta", "gamma", "gamma"])),
                Arc::new(StringArray::from(vec!["inactive", "active", "inactive"])),
                Arc::new(Int32Array::from(vec![400, 500, 600])),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch2)], schema.clone());
        Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let dataset = Dataset::open(&dataset_uri).await.unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        let indices_after = dataset.load_indices().await.unwrap();
        let frag_bitmap_after = indices_after[0].fragment_bitmap.as_ref().unwrap();
        assert_eq!(
            frag_bitmap_after.len(),
            1,
            "index still covers exactly 1 fragment after the append"
        );

        // (beta, active) is in fragment 0 (indexed). Query must find it via
        // the compound prefix path.
        let result = dataset
            .scan()
            .filter("tenant_id = 'beta' AND status = 'active'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 1, "(beta, active) -> 1 row");
        let value_col = result
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(value_col.value(0), 300, "row from indexed fragment");

        // (gamma, active) lives only in the unindexed fragment. The scanner
        // must fall back to a full scan for that fragment.
        let result2 = dataset
            .scan()
            .filter("tenant_id = 'gamma' AND status = 'active'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result2.num_rows(), 1, "(gamma, active) -> 1 row from unindexed fragment");
        let value_col2 = result2
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(value_col2.value(0), 500);

        // tenant_id alone spans both fragments — covers prefix-only path on
        // the indexed fragment plus full scan on the unindexed one.
        let result3 = dataset
            .scan()
            .filter("tenant_id = 'beta'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result3.num_rows(), 2, "beta spans both fragments");
        let value_col3 = result3
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut values: Vec<i32> = (0..value_col3.len()).map(|i| value_col3.value(i)).collect();
        values.sort();
        assert_eq!(values, vec![300, 400]);

        // Predicate-order independence: status before tenant_id must hit the
        // same compound index.
        let result_reversed = dataset
            .scan()
            .filter("status = 'active' AND tenant_id = 'beta'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result_reversed.num_rows(), 1);
        let value_col_reversed = result_reversed
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(value_col_reversed.value(0), 300);
    }
}
