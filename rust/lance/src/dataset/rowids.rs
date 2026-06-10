// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::Dataset;
use crate::session::caches::{RowIdIndexKey, RowIdSequenceKey};
use crate::{Error, Result};
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use lance_core::utils::deletion::DeletionVector;
use lance_core::utils::path::LancePathExt;
use lance_io::object_store::ObjectStore;
use lance_table::{
    format::{
        ExternalFile, Fragment, RowDatasetVersionMeta, RowDatasetVersionSequence, RowIdMeta,
        row_meta_inline_threshold_bytes,
    },
    rowids::{FragmentRowIdIndex, RowIdIndex, RowIdSequence, read_row_ids, version},
};
use object_store::path::Path;
use std::collections::HashMap;
use std::sync::Arc;

/// Directory under the dataset root holding externalized row-meta sequences
/// (row ids and created_at / last_updated_at version sequences too large to
/// store inline in the manifest).
pub const ROWIDS_DIR: &str = "_rowids";

/// Load a row id sequence from the given dataset and fragment.
pub async fn load_row_id_sequence(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Arc<RowIdSequence>> {
    // Virtual path to prevent collisions in the cache.
    match &fragment.row_id_meta {
        None => Err(Error::internal("Missing row id meta")),
        Some(RowIdMeta::Inline(data)) => {
            let data = data.clone();
            let key = RowIdSequenceKey {
                fragment_id: fragment.id,
            };
            dataset
                .metadata_cache
                .get_or_insert_with_key(key, || async move { read_row_ids(&data) })
                .await
        }
        Some(RowIdMeta::External(file_slice)) => {
            let file_slice = file_slice.clone();
            let dataset_clone = dataset.clone();
            let key = RowIdSequenceKey {
                fragment_id: fragment.id,
            };
            dataset
                .metadata_cache
                .get_or_insert_with_key(key, || async move {
                    let data = read_external_meta_bytes(
                        &dataset_clone.object_store,
                        &dataset_clone.base,
                        &file_slice,
                    )
                    .await?;
                    read_row_ids(&data)
                })
                .await
        }
    }
}

/// Load row id sequences from the given dataset and fragments.
///
/// Returned as a vector of (fragment_id, sequence) pairs. These are not
/// guaranteed to be in the same order as the input fragments.
pub fn load_row_id_sequences<'a>(
    dataset: &'a Dataset,
    fragments: &'a [Fragment],
) -> impl Stream<Item = Result<(u32, Arc<RowIdSequence>)>> + 'a {
    futures::stream::iter(fragments)
        .map(|fragment| {
            load_row_id_sequence(dataset, fragment).map_ok(move |seq| (fragment.id as u32, seq))
        })
        .buffer_unordered(dataset.object_store.io_parallelism())
}

pub async fn get_row_id_index(
    dataset: &Dataset,
) -> Result<Option<Arc<lance_table::rowids::RowIdIndex>>> {
    if dataset.manifest.uses_stable_row_ids() {
        let key = RowIdIndexKey {
            version: dataset.manifest.version,
        };
        let index = dataset
            .metadata_cache
            .get_or_insert_with_key(key, || load_row_id_index(dataset))
            .await?;
        Ok(Some(index))
    } else {
        Ok(None)
    }
}

async fn load_row_id_index(dataset: &Dataset) -> Result<lance_table::rowids::RowIdIndex> {
    let sequences = load_row_id_sequences(dataset, &dataset.manifest.fragments)
        .try_collect::<Vec<_>>()
        .await?;

    let fragments = dataset.get_fragments();
    let fragment_map: std::collections::HashMap<u32, &crate::dataset::fragment::FileFragment> =
        fragments.iter().map(|f| (f.id() as u32, f)).collect();

    let fragment_indices: Vec<_> =
        futures::stream::iter(sequences.into_iter().map(|(fragment_id, sequence)| {
            let fragment = fragment_map
                .get(&fragment_id)
                .expect("Fragment should exist");
            let has_deletion_file = fragment.metadata().deletion_file.is_some();
            let fragment_clone = (*fragment).clone();
            async move {
                let deletion_vector = if has_deletion_file {
                    match fragment_clone.get_deletion_vector().await {
                        Ok(Some(dv)) => dv,
                        Ok(None) | Err(_) => Arc::new(DeletionVector::default()),
                    }
                } else {
                    Arc::new(DeletionVector::default())
                };

                Ok::<FragmentRowIdIndex, Error>(FragmentRowIdIndex {
                    fragment_id,
                    row_id_sequence: sequence,
                    deletion_vector,
                })
            }
        }))
        .buffer_unordered(dataset.object_store.io_parallelism())
        .try_collect()
        .await?;

    let index = RowIdIndex::new(&fragment_indices)?;

    Ok(index)
}

/// Which of the three per-fragment row-meta sequences a map entry refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RowMetaKind {
    RowId,
    CreatedAtVersion,
    LastUpdatedAtVersion,
}

const ALL_ROW_META_KINDS: [RowMetaKind; 3] = [
    RowMetaKind::RowId,
    RowMetaKind::CreatedAtVersion,
    RowMetaKind::LastUpdatedAtVersion,
];

/// Bytes and original location of row metas that were pulled inline for a
/// transaction's sync apply logic. Lets the spill pass reuse the original
/// external reference for sequences the transaction left untouched instead
/// of rewriting identical bytes on every commit.
#[derive(Debug, Default)]
pub(crate) struct HydratedRowMetas {
    entries: HashMap<(u64, RowMetaKind), (ExternalFile, Vec<u8>)>,
}

/// Whether any of the fragment's three row-meta sequences is stored
/// externally. Cheap pre-check so commit paths only clone the manifest for
/// hydration when there is actually something to hydrate.
pub(crate) fn fragment_has_external_row_meta(fragment: &Fragment) -> bool {
    ALL_ROW_META_KINDS
        .iter()
        .any(|kind| external_file(fragment, *kind).is_some())
}

fn inline_bytes(fragment: &Fragment, kind: RowMetaKind) -> Option<&[u8]> {
    match kind {
        RowMetaKind::RowId => match &fragment.row_id_meta {
            Some(RowIdMeta::Inline(data)) => Some(data.as_slice()),
            _ => None,
        },
        RowMetaKind::CreatedAtVersion => match &fragment.created_at_version_meta {
            Some(RowDatasetVersionMeta::Inline(data)) => Some(data.as_ref()),
            _ => None,
        },
        RowMetaKind::LastUpdatedAtVersion => match &fragment.last_updated_at_version_meta {
            Some(RowDatasetVersionMeta::Inline(data)) => Some(data.as_ref()),
            _ => None,
        },
    }
}

fn external_file(fragment: &Fragment, kind: RowMetaKind) -> Option<&ExternalFile> {
    match kind {
        RowMetaKind::RowId => match &fragment.row_id_meta {
            Some(RowIdMeta::External(file)) => Some(file),
            _ => None,
        },
        RowMetaKind::CreatedAtVersion => match &fragment.created_at_version_meta {
            Some(RowDatasetVersionMeta::External(file)) => Some(file),
            _ => None,
        },
        RowMetaKind::LastUpdatedAtVersion => match &fragment.last_updated_at_version_meta {
            Some(RowDatasetVersionMeta::External(file)) => Some(file),
            _ => None,
        },
    }
}

fn set_external(fragment: &mut Fragment, kind: RowMetaKind, file: ExternalFile) {
    match kind {
        RowMetaKind::RowId => fragment.row_id_meta = Some(RowIdMeta::External(file)),
        RowMetaKind::CreatedAtVersion => {
            fragment.created_at_version_meta = Some(RowDatasetVersionMeta::External(file));
        }
        RowMetaKind::LastUpdatedAtVersion => {
            fragment.last_updated_at_version_meta = Some(RowDatasetVersionMeta::External(file));
        }
    }
}

fn set_inline(fragment: &mut Fragment, kind: RowMetaKind, bytes: Vec<u8>) {
    match kind {
        RowMetaKind::RowId => fragment.row_id_meta = Some(RowIdMeta::Inline(bytes)),
        RowMetaKind::CreatedAtVersion => {
            fragment.created_at_version_meta =
                Some(RowDatasetVersionMeta::Inline(Arc::from(bytes)));
        }
        RowMetaKind::LastUpdatedAtVersion => {
            fragment.last_updated_at_version_meta =
                Some(RowDatasetVersionMeta::Inline(Arc::from(bytes)));
        }
    }
}

async fn read_external_meta_bytes(
    object_store: &ObjectStore,
    base_path: &Path,
    file: &ExternalFile,
) -> Result<Vec<u8>> {
    // `Path::join` percent-encodes a separator inside its argument, so the
    // multi-segment relative path must be joined part-by-part.
    let path = base_path.child_path(&Path::parse(file.path.as_str())?);
    let range = file.offset as usize..(file.offset as usize + file.size as usize);
    let data = object_store.open(&path).await?.get_range(range).await?;
    Ok(data.to_vec())
}

/// Load a created_at / last_updated_at version sequence, resolving external
/// storage through the dataset's object store.
///
/// No sequence cache: version sequences are only read on compaction rewrites
/// and on scans that explicitly project `_row_created_at_version` /
/// `_row_last_updated_at_version` — both cold paths, unlike row-id sequences
/// which back the per-query row-id index.
pub async fn load_version_sequence(
    dataset: &Dataset,
    meta: &RowDatasetVersionMeta,
) -> Result<RowDatasetVersionSequence> {
    match meta {
        RowDatasetVersionMeta::Inline(_) => Ok(meta.load_sequence()?),
        RowDatasetVersionMeta::External(file) => {
            let bytes =
                read_external_meta_bytes(&dataset.object_store, &dataset.base, file).await?;
            Ok(version::read_dataset_versions(&bytes)?)
        }
    }
}

/// Pull externally-stored row metas inline (in memory only) so a
/// transaction's sync apply logic — which can only decode `Inline` metas —
/// sees every sequence. Returns the original external references so
/// [`externalize_large_row_metas`] can restore them for sequences the
/// transaction did not change.
pub(crate) async fn hydrate_external_row_metas(
    object_store: &ObjectStore,
    base_path: &Path,
    fragments: &mut [Fragment],
) -> Result<HydratedRowMetas> {
    let mut hydrated = HydratedRowMetas::default();
    for fragment in fragments.iter_mut() {
        for kind in ALL_ROW_META_KINDS {
            let Some(file) = external_file(fragment, kind).cloned() else {
                continue;
            };
            let bytes = read_external_meta_bytes(object_store, base_path, &file).await?;
            hydrated
                .entries
                .insert((fragment.id, kind), (file, bytes.clone()));
            set_inline(fragment, kind, bytes);
        }
    }
    Ok(hydrated)
}

/// Move every inline row-meta sequence larger than
/// [`row_meta_inline_threshold_bytes`] out of the manifest into a shared
/// external file under [`ROWIDS_DIR`], so the manifest re-pays only a
/// ~50-byte reference per commit instead of the full sequence.
///
/// Sequences whose bytes match a [`HydratedRowMetas`] entry get their
/// original external reference back without any write. All sequences spilled
/// by one call share a single object (`file_name`) at distinct offsets.
/// Returns the number of sequences written to the external file.
pub(crate) async fn externalize_large_row_metas(
    object_store: &ObjectStore,
    base_path: &Path,
    fragments: &mut [Fragment],
    file_name: &str,
    hydrated: &HydratedRowMetas,
) -> Result<usize> {
    // Pass 1 (read-only): restore unchanged hydrated sequences and collect
    // the (fragment index, kind, offset, size) layout of genuinely new
    // oversized sequences into one buffer.
    let mut restores: Vec<(usize, RowMetaKind, ExternalFile)> = Vec::new();
    let mut spills: Vec<(usize, RowMetaKind, u64, u64)> = Vec::new();
    let mut buffer: Vec<u8> = Vec::new();
    let relative_path = format!("{ROWIDS_DIR}/{file_name}");

    let threshold_bytes = row_meta_inline_threshold_bytes();
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        for kind in ALL_ROW_META_KINDS {
            let Some(bytes) = inline_bytes(fragment, kind) else {
                continue;
            };
            if bytes.len() <= threshold_bytes {
                continue;
            }
            if let Some((file, original_bytes)) = hydrated.entries.get(&(fragment.id, kind))
                && original_bytes.as_slice() == bytes
            {
                restores.push((fragment_index, kind, file.clone()));
                continue;
            }
            let offset = buffer.len() as u64;
            buffer.extend_from_slice(bytes);
            spills.push((fragment_index, kind, offset, bytes.len() as u64));
        }
    }

    if !buffer.is_empty() {
        let full_path = base_path.clone().join(ROWIDS_DIR).join(file_name);
        object_store.put(&full_path, &buffer).await?;
    }

    let spilled_count = spills.len();
    for (fragment_index, kind, file) in restores {
        set_external(&mut fragments[fragment_index], kind, file);
    }
    for (fragment_index, kind, offset, size) in spills {
        set_external(
            &mut fragments[fragment_index],
            kind,
            ExternalFile {
                path: relative_path.clone(),
                offset,
                size,
            },
        );
    }
    Ok(spilled_count)
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use crate::dataset::{UpdateBuilder, WriteMode, WriteParams, builder::DatasetBuilder};

    use super::*;

    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::Future;
    use lance_core::datatypes::Schema;
    use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
    use lance_datagen::Dimension;
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use std::collections::HashMap;
    use std::collections::HashSet;

    fn sequence_batch(values: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(values))]).unwrap()
    }

    #[tokio::test]
    async fn test_empty_dataset_rowids() {
        let schema = sequence_batch(0..0).schema();
        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert!(dataset.manifest.uses_stable_row_ids());

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_none());

        assert_eq!(dataset.manifest().next_row_id, 0);
    }

    #[tokio::test]
    async fn test_must_set_on_creation() {
        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &tmp_dir;

        let batch = sequence_batch(0..10);
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: false,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_stable_row_ids());

        // Trying to append without stable row ids should pass (a warning is emitted) but should not
        // affect the stable_row_ids setting.
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            mode: WriteMode::Append,
            ..Default::default()
        };
        let reader =
            RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema());
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();
        assert!(!dataset.manifest().uses_stable_row_ids());
    }

    #[tokio::test]
    async fn test_new_row_ids() {
        let num_rows = 25u64;
        let batch = sequence_batch(0..num_rows as i32);
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 10,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();

        let found_addresses = (0..num_rows)
            .map(|i| index.get(i).unwrap())
            .collect::<Vec<_>>();
        let expected_addresses = (0..num_rows)
            .map(|i| {
                let fragment_id = i / 10;
                RowAddress::new_from_parts(fragment_id as u32, (i % 10) as u32)
            })
            .collect::<Vec<_>>();
        assert_eq!(found_addresses, expected_addresses);

        assert_eq!(dataset.manifest().next_row_id, num_rows);
    }

    #[tokio::test]
    async fn test_row_ids_overwrite() {
        // Validate we don't re-use after overwriting
        let num_rows = 10u64;
        let batch = sequence_batch(0..num_rows as i32);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmp_path, Some(write_params))
            .await
            .unwrap();

        // Overwriting should NOT reset the row id counter.
        assert_eq!(dataset.manifest().next_row_id, 2 * num_rows);

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_none());
        assert!(index.get(num_rows).is_some());
    }

    #[tokio::test]
    async fn test_row_ids_append() {
        // Validate we handle row ids well when appending concurrently.
        fn write_batch(uri: &str, start: i32) -> impl Future<Output = Result<()>> + '_ {
            let batch = sequence_batch(start..(start + 10));
            let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
            let write_params = WriteParams {
                enable_stable_row_ids: true,
                mode: WriteMode::Append,
                ..Default::default()
            };
            async move {
                let _ = Dataset::write(reader, uri, Some(write_params)).await?;
                Ok(())
            }
        }

        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let tmp_path = &temp_dir;
        let mut start = 0;
        // Just do one first to create the dataset.
        write_batch(tmp_path, start).await.unwrap();
        start += 10;
        // Now do the rest concurrently.
        let futures = (0..5)
            .map(|offset| write_batch(tmp_path, start + offset * 10))
            .collect::<Vec<_>>();
        futures::future::try_join_all(futures).await.unwrap();

        let dataset = DatasetBuilder::from_uri(tmp_path).load().await.unwrap();

        assert_eq!(dataset.manifest().next_row_id, 60);

        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_some());
        assert!(index.get(60).is_none());
    }

    #[tokio::test]
    async fn test_scan_row_ids() {
        // Write dataset with multiple files -> _rowid != _rowaddr
        // Scan with and without each.;
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        for with_row_id in [true, false] {
            for with_row_address in &[true, false] {
                for projection in &[vec![], vec!["id"]] {
                    if !with_row_id && !with_row_address && projection.is_empty() {
                        continue;
                    }

                    let mut scan = dataset.scan();
                    if with_row_id {
                        scan.with_row_id();
                    }
                    if *with_row_address {
                        scan.with_row_address();
                    }
                    let scan = scan.project(projection).unwrap();
                    let result = scan.try_into_batch().await.unwrap();

                    if with_row_id {
                        let row_ids = result[ROW_ID]
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let expected = vec![0, 1, 2, 3, 4, 5].into();
                        assert_eq!(row_ids, &expected);
                    }

                    if *with_row_address {
                        let row_addrs = result[ROW_ADDR]
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let expected =
                            vec![0, 1, 1 << 32, (1 << 32) + 1, 2 << 32, (2 << 32) + 1].into();
                        assert_eq!(row_addrs, &expected);
                    }

                    if !projection.is_empty() {
                        let ids = result["id"].as_any().downcast_ref::<Int32Array>().unwrap();
                        let expected = vec![0, 1, 2, 3, 4, 5].into();
                        assert_eq!(ids, &expected);
                    }
                }
            }
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_delete_with_row_ids(#[values(true, false)] with_scalar_index: bool) {
        let batch = sequence_batch(0..6);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 2,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        if with_scalar_index {
            dataset
                .create_index(
                    &["id"],
                    IndexType::Scalar,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }

        dataset.delete("id = 3 or id = 4").await.unwrap();

        let mut scan = dataset.scan();
        scan.with_row_id().with_row_address();
        let result = scan.try_into_batch().await.unwrap();

        let row_ids = result[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let expected = vec![0, 1, 2, 5].into();
        assert_eq!(row_ids, &expected);

        let row_addrs = result[ROW_ADDR]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let expected = vec![0, 1, 1 << 32, (2 << 32) + 1].into();
        assert_eq!(row_addrs, &expected);
    }

    #[tokio::test]
    async fn test_row_ids_update() {
        // Updated fragments get fresh row ids.
        let num_rows = 5u64;
        let batch = sequence_batch(0..num_rows as i32);

        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.manifest().next_row_id, num_rows);

        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 3")
            .unwrap()
            .set("id", "100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_some());
        // the updated row ids mapping to new address
        assert_eq!(index.get(3), Some(RowAddress::new_from_parts(1, 0)));
        // there is no new row id
        assert_eq!(index.get(5), None);
    }

    fn build_rowid_to_i_map(row_ids: &UInt64Array, i_array: &Int32Array) -> HashMap<u64, i32> {
        row_ids
            .values()
            .iter()
            .zip(i_array.values().iter())
            .map(|(&row_id, &i)| (row_id, i))
            .collect()
    }

    async fn scan_rowid_map(dataset: &Dataset) -> HashMap<u64, i32> {
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let row_ids = result[ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        build_rowid_to_i_map(row_ids, i)
    }

    async fn compact(dataset: &mut Dataset, target_rows: usize) {
        let options = CompactionOptions {
            target_rows_per_fragment: target_rows,
            ..Default::default()
        };
        let _ = compact_files(dataset, options, None).await.unwrap();
    }

    async fn delete(dataset: &mut Dataset, expr: &str) {
        dataset.delete(expr).await.unwrap();
    }

    #[tokio::test]
    async fn test_stable_row_id_after_multiple_deletion_and_compaction() {
        async fn delete(dataset: &mut Dataset, expr: &str) {
            dataset.delete(expr).await.unwrap();
        }

        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // first delete and compact
        delete(&mut dataset, "i = 2 or i = 3 or i = 5").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 20).await;
        let map_after = scan_rowid_map(&dataset).await;

        // verify row id
        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // second delete
        delete(&mut dataset, "i = 9").await;
        let mut scan = dataset.scan();
        let result = scan
            .filter("i >= 0")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let id_set = ids.values().iter().cloned().collect::<HashSet<_>>();
        let expected: Vec<i32> = (0..60)
            .filter(|&i| i != 2 && i != 3 && i != 5 && i != 9)
            .collect();
        assert_eq!(id_set, expected.iter().cloned().collect::<HashSet<_>>());

        // get the row_id where i == 15
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan
            .filter("i == 15")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let row_id_vec = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();

        // third delete and compact
        delete(&mut dataset, "i = 15 or i = 25").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 30).await;
        let map_after = scan_rowid_map(&dataset).await;

        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // verify the rowid represent i == 15 has been deleted
        let result = dataset
            .take_rows(&row_id_vec, Schema::try_from(dataset.schema()).unwrap())
            .await
            .unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn test_stable_row_id_after_deletion_update_and_compaction() {
        // gen dataset
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "i",
                lance_datagen::array::step::<arrow_array::types::Int32Type>(),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // delete some rows
        delete(&mut dataset, "i = 2 or i = 3 or i = 5").await;
        let map_before = scan_rowid_map(&dataset).await;

        // update some rows
        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i >= 15")
            .unwrap()
            .set("category", "999")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // compact the dataset
        let mut dataset = Arc::try_unwrap(updated_dataset).expect("no other Arc references");
        compact(&mut dataset, 20).await;
        let map_after = scan_rowid_map(&dataset).await;

        // verify row id
        assert_eq!(
            map_before.keys().collect::<HashSet<_>>(),
            map_after.keys().collect::<HashSet<_>>()
        );
        for row_id in map_before.keys() {
            assert_eq!(map_before[row_id], map_after[row_id]);
        }

        // verify category filed
        let mut scan = dataset.scan();
        scan.with_row_id();
        scan.scan_in_order(true);
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let category = result["category"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for idx in 0..i.len() {
            if i.value(idx) >= 15 {
                assert_eq!(category.value(idx), 999);
            }
        }
    }

    /// Force every row-meta sequence to spill to external storage so the
    /// full external pipeline is exercised at toy scale.
    fn force_tiny_row_meta_threshold() {
        // SAFETY: nextest runs each test in its own process and the
        // threshold LazyLock has not been read before this call.
        unsafe { std::env::set_var("LANCE_ROW_META_INLINE_THRESHOLD_BYTES", "8") };
    }

    fn external_row_meta_files(fragments: &[Fragment]) -> HashSet<String> {
        let mut paths = HashSet::new();
        for fragment in fragments {
            if let Some(RowIdMeta::External(file)) = &fragment.row_id_meta {
                paths.insert(file.path.clone());
            }
            if let Some(lance_table::format::RowDatasetVersionMeta::External(file)) =
                &fragment.created_at_version_meta
            {
                paths.insert(file.path.clone());
            }
            if let Some(lance_table::format::RowDatasetVersionMeta::External(file)) =
                &fragment.last_updated_at_version_meta
            {
                paths.insert(file.path.clone());
            }
        }
        paths
    }

    async fn list_rowids_dir(dataset: &Dataset) -> HashSet<String> {
        let base = dataset.base.clone();
        dataset
            .object_store
            .read_dir_all(&dataset.rowids_dir(), None)
            .map_ok(|meta| {
                meta.location
                    .prefix_match(&base)
                    .map(|parts| object_store::path::Path::from_iter(parts).to_string())
                    .unwrap_or_else(|| meta.location.to_string())
            })
            .try_collect::<HashSet<_>>()
            .await
            .unwrap_or_default()
    }

    /// Dataset for the externalization tests: 6 fragments x 10 rows with
    /// stable row ids, on a real tempdir (cleanup needs a listable store).
    async fn external_meta_test_dataset(
        tmp_path: &str,
    ) -> (Dataset, HashMap<u64, i32>) {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_dataset_with_params(
                tmp_path,
                FragmentCount::from(6),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        delete(&mut dataset, "i = 2 or i = 17").await;
        let map_before = scan_rowid_map(&dataset).await;
        compact(&mut dataset, 30).await;
        (dataset, map_before)
    }

    #[tokio::test]
    async fn test_row_metas_externalize_on_commit_and_read_back() {
        force_tiny_row_meta_threshold();
        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let (dataset, map_before) = external_meta_test_dataset(&tmp_dir).await;

        // The compaction rewrite re-assigned every sequence inline; the
        // commit spill must have moved them out of the manifest.
        let externals = external_row_meta_files(&dataset.manifest.fragments);
        assert!(
            !externals.is_empty(),
            "compaction commit must externalize row metas over the threshold, fragments: {:?}",
            dataset.manifest.fragments,
        );
        let on_disk = list_rowids_dir(&dataset).await;
        for path in &externals {
            assert!(on_disk.contains(path), "{path} referenced but missing from _rowids: {on_disk:?}");
        }

        // Row-id index loads through the external read path and row ids are
        // unchanged by the spill.
        let map_after = scan_rowid_map(&dataset).await;
        assert_eq!(map_before, map_after);
        let index = get_row_id_index(&dataset).await.unwrap().unwrap();
        assert!(index.get(0).is_some());

        // Version columns read through the external version-sequence loader.
        let mut scan = dataset.scan();
        scan.project(&["i", lance_core::ROW_CREATED_AT_VERSION]).unwrap();
        let result = scan.try_into_batch().await.unwrap();
        let created = result[lance_core::ROW_CREATED_AT_VERSION]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(
            created.values().iter().all(|v| *v == 1),
            "all rows were created at version 1, got {created:?}",
        );
    }

    #[tokio::test]
    async fn test_update_after_externalization_preserves_created_at() {
        force_tiny_row_meta_threshold();
        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let (dataset, _) = external_meta_test_dataset(&tmp_dir).await;

        let externals_before = external_row_meta_files(&dataset.manifest.fragments);
        assert!(!externals_before.is_empty());

        // The update's sync apply logic can only read Inline metas; the
        // commit-path hydration must surface the externalized sequences so
        // the rewritten row keeps its original created_at instead of being
        // misclassified as an insert.
        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 5")
            .unwrap()
            .set("category", "999")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let dataset = update_result.new_dataset;
        let updated_version = dataset.manifest.version;

        let mut scan = dataset.scan();
        scan.project(&[
            "i",
            lance_core::ROW_CREATED_AT_VERSION,
            lance_core::ROW_LAST_UPDATED_AT_VERSION,
        ])
        .unwrap();
        let result = scan.try_into_batch().await.unwrap();
        let i = result["i"].as_any().downcast_ref::<Int32Array>().unwrap();
        let created = result[lance_core::ROW_CREATED_AT_VERSION]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let updated = result[lance_core::ROW_LAST_UPDATED_AT_VERSION]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        for idx in 0..i.len() {
            assert_eq!(
                created.value(idx),
                1,
                "row i={} must keep created_at=1 across the update",
                i.value(idx),
            );
            if i.value(idx) == 5 {
                assert_eq!(updated.value(idx), updated_version);
            }
        }

        // Untouched fragments must reuse their original external references
        // rather than rewriting identical bytes on every update commit.
        let externals_after = external_row_meta_files(&dataset.manifest.fragments);
        assert!(
            externals_after.intersection(&externals_before).next().is_some(),
            "unchanged fragments must keep their original external files; before: \
             {externals_before:?}, after: {externals_after:?}",
        );
    }

    #[tokio::test]
    async fn test_manifest_byte_size_regression_canary() {
        // No tiny-threshold override here: this pins the DEFAULT manifest
        // footprint. A production table re-paid 4.9 MB of manifest per
        // commit before row metas were externalized, and nothing in the
        // suite would have caught that regression. 100 fragments of
        // appended rows must stay comfortably small — fragment metadata
        // ~130 B each, Range-encoded row ids ~16 B, uniform version runs
        // ~20 B. The 256 KiB bound is ~10x slack over the expected ~25 KiB
        // so layout changes don't flake it, while a sequence-inlining
        // regression (MBs) still trips it.
        const MANIFEST_SIZE_BOUND_BYTES: u64 = 256 * 1024;

        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_dataset_with_params(
                &tmp_dir,
                FragmentCount::from(100),
                FragmentRowCount::from(1000),
                Some(WriteParams {
                    max_rows_per_file: 1000,
                    enable_stable_row_ids: true,
                    enable_v2_manifest_paths: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // One more commit so the measured manifest carries version metadata
        // assigned through the transaction path, not just initial creation.
        let batch = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_reader_rows(lance_datagen::RowCount::from(1000), lance_datagen::BatchCount::from(1));
        dataset.append(batch, None).await.unwrap();

        let manifest_sizes: Vec<u64> = std::fs::read_dir(format!("{}/_versions", &*tmp_dir))
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .collect();
        let largest = manifest_sizes.iter().copied().max().unwrap();
        assert!(
            largest <= MANIFEST_SIZE_BOUND_BYTES,
            "largest manifest is {largest} bytes (bound {MANIFEST_SIZE_BOUND_BYTES}); a \
             per-commit metadata growth regression has crept into the manifest",
        );
    }

    #[tokio::test]
    async fn test_cleanup_retains_referenced_external_row_metas() {
        force_tiny_row_meta_threshold();
        // Manifest timestamps come from the mocked clock while the tempdir's
        // file mtimes come from the real one, and cleanup's listing only
        // yields files older than the earliest retained manifest. Park the
        // mocked clock far past any real mtime so the listing sees the
        // tempdir's files at all.
        mock_instant::thread_local::MockClock::set_system_time(
            std::time::Duration::from_secs(50_000 * 24 * 60 * 60),
        );
        let tmp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let (mut dataset, _) = external_meta_test_dataset(&tmp_dir).await;

        // A second compaction supersedes the first pass's external files so
        // the cleanup has genuine orphans-to-be once old versions age out.
        delete(&mut dataset, "i = 30").await;
        compact(&mut dataset, 60).await;
        let map_before = scan_rowid_map(&dataset).await;
        let referenced = external_row_meta_files(&dataset.manifest.fragments);
        assert!(!referenced.is_empty());
        let on_disk_before = list_rowids_dir(&dataset).await;
        assert!(
            on_disk_before.len() > referenced.len(),
            "expected superseded external files on disk before cleanup; on disk: \
             {on_disk_before:?}, referenced: {referenced:?}",
        );

        // Advance past every write above so all old manifests qualify
        // (`timestamp < before_timestamp` is strict).
        mock_instant::thread_local::MockClock::set_system_time(
            std::time::Duration::from_secs(50_001 * 24 * 60 * 60),
        );
        let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
            .before_timestamp(crate::utils::temporal::utc_now())
            .delete_unverified(true)
            .build();
        let stats = dataset.cleanup_with_policy(policy).await.unwrap();
        assert!(
            stats.row_meta_files_removed >= 1,
            "cleanup must reclaim superseded external row-meta files, stats: {stats:?}",
        );

        let on_disk_after = list_rowids_dir(&dataset).await;
        assert_eq!(
            on_disk_after, referenced,
            "after cleanup, _rowids must hold exactly the files the current manifest references",
        );

        // The surviving external files still serve reads.
        let map_after = scan_rowid_map(&dataset).await;
        assert_eq!(map_before, map_after);
    }
}
