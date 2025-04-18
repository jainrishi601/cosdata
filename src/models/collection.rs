use super::common::WaCustomError;
use super::paths::get_data_path;
use crate::metadata::MetadataSchema;
use lmdb::{Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use serde_cbor::to_vec;
use siphasher::sip::SipHasher24;
use std::fs::create_dir_all;
use std::{fs, hash::Hasher, path::Path, sync::Arc};

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct DenseVectorOptions {
    pub enabled: bool,
    pub dimension: usize,
}

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct SparseVectorOptions {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFIDFOptions {
    pub enabled: bool,
}

#[derive(Deserialize, Clone, Serialize, Debug)]
pub struct CollectionConfig {
    pub max_vectors: Option<i32>,
    pub replication_factor: Option<i32>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Collection {
    pub name: String,
    pub description: Option<String>,
    pub dense_vector: DenseVectorOptions,
    pub sparse_vector: SparseVectorOptions,
    pub tf_idf_options: TFIDFOptions,
    pub metadata_schema: Option<MetadataSchema>,
    pub config: CollectionConfig,
}

impl Collection {
    pub fn new(
        name: String,
        description: Option<String>,
        dense_vector_options: DenseVectorOptions,
        sparse_vector_options: SparseVectorOptions,
        tf_idf_options: TFIDFOptions,
        metadata_schema: Option<MetadataSchema>,
        config: CollectionConfig,
    ) -> Result<Self, WaCustomError> {
        if name.is_empty() {
            return Err(WaCustomError::InvalidParams);
        }

        let collection = Collection {
            name,
            description,
            dense_vector: dense_vector_options,
            sparse_vector: sparse_vector_options,
            tf_idf_options,
            metadata_schema,
            config,
        };

        let collection_path = collection.get_path();
        fs::create_dir_all(&collection_path).map_err(|e| WaCustomError::FsError(e.to_string()))?;

        Ok(collection)
    }

    /// Computes the SipHash of the collection name
    pub fn get_hash(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        hasher.write(self.name.as_bytes());
        hasher.finish()
    }

    /// computes the key used to store the collection in the database
    pub fn get_key(&self) -> [u8; 8] {
        let hash = self.get_hash();
        hash.to_le_bytes()
    }

    /// creates a path out of the collection name
    pub fn get_path(&self) -> Arc<Path> {
        let collections_path = get_data_path().join("collections");
        create_dir_all(&collections_path).expect("Failed to create collections directory");
        collections_path.join(&self.name).into()
    }

    /// serializes the collection
    pub fn serialize(&self) -> Result<Vec<u8>, WaCustomError> {
        to_vec(self).map_err(|e| WaCustomError::SerializationError(e.to_string()))
    }

    /// perists the collection instance on disk (lmdb -> collections database)
    pub fn persist(&self, env: &Environment, db: Database) -> Result<(), WaCustomError> {
        let key = self.get_key();
        let value = self.serialize()?;

        let mut txn = env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.put(db, &key, &value, WriteFlags::empty())
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    /// deletes a collection instance from the disk (lmdb -> collections database)
    #[allow(dead_code)]
    pub fn delete(&self, env: &Environment, db: Database) -> Result<(), WaCustomError> {
        let key = self.get_key();

        let mut txn = env
            .begin_rw_txn()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        txn.del(db, &key, None)
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
        txn.commit()
            .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

        Ok(())
    }
}
