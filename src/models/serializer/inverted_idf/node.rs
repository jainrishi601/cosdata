use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::InvertedIndexIDFCache,
    inverted_index_idf::{InvertedIndexIDFNode, InvertedIndexIDFNodeData},
    prob_lazy_load::lazy_item::ProbLazyItem,
    types::FileOffset,
};

use super::{InvertedIndexIDFSerialize, INVERTED_INDEX_DATA_CHUNK_SIZE};

// @SERIALIZED_SIZE:
//
//   4 byte for dim index +                          | 4
//   1 byte for implicit flag & quantization         | 5
//   2 bytes for data map len +                      | 7
//   INVERTED_INDEX_DATA_CHUNK_SIZE * (              |
//     2 bytes for quotient +                        |
//     4 bytes of pagepool                           |
//   ) +                                             | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 7
//   4 byte for next data chunk                      | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 11
//   16 * 4 bytes for dimension offsets +            | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 75
impl InvertedIndexIDFSerialize for InvertedIndexIDFNode {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        _: u8,
        _: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let data_file_idx = (self.dim_index % data_file_parts as u32) as u8;
        if !self.is_serialized.swap(true, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64)?;
            dim_bufman.update_u32_with_cursor(cursor, self.dim_index)?;
            let mut quantization_and_implicit = self.quantization_bits;
            if self.implicit {
                quantization_and_implicit |= 1u8 << 7;
            }
            dim_bufman.update_u8_with_cursor(cursor, quantization_and_implicit)?;
            self.data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                self.quantization_bits,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + INVERTED_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 11,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                self.quantization_bits,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        } else if self.is_dirty.swap(false, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64 + 5)?;
            self.data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                self.quantization_bits,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + INVERTED_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 11,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                self.quantization_bits,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        } else {
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + INVERTED_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 11,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                self.quantization_bits,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        };
        Ok(self.file_offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        _: u8,
        _: u8,
        data_file_parts: u8,
        cache: &InvertedIndexIDFCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let dim_index = dim_bufman.read_u32_with_cursor(cursor)?;
        let quantization_and_implicit = dim_bufman.read_u8_with_cursor(cursor)?;
        let implicit = (quantization_and_implicit & (1u8 << 7)) != 0;
        let quantization_bits = (quantization_and_implicit << 1) >> 1;
        let data_file_idx = (dim_index % data_file_parts as u32) as u8;
        let data = <*mut ProbLazyItem<InvertedIndexIDFNodeData>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 5),
            quantization_bits,
            data_file_idx,
            data_file_parts,
            cache,
        )?;
        let children = AtomicArray::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + INVERTED_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 11),
            quantization_bits,
            data_file_idx,
            data_file_parts,
            cache,
        )?;

        Ok(Self {
            is_serialized: AtomicBool::new(true),
            is_dirty: AtomicBool::new(false),
            dim_index,
            implicit,
            file_offset,
            quantization_bits,
            data,
            children,
        })
    }
}
