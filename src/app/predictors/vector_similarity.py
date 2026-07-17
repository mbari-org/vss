# fastapi-vss, Apache-2.0 license
# Filename: predictors/vector_similarity.py
# Description: Runs operations on Redis database with RediSearch on embedded vectors
import os
from concurrent.futures import ThreadPoolExecutor

import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.logger import err, info, debug

# Number of concurrent Redis KNN queries per batch (redis-py releases the GIL on socket I/O,
# so threads issue searches in parallel over the connection pool). Tunable via VSS_KNN_WORKERS.
KNN_SEARCH_WORKERS = int(os.getenv("VSS_KNN_WORKERS", "16"))


class VectorSimilarity:
    INDEX_NAME = "index"  # Vector Index Name
    DOC_PREFIX = "doc:"  # RediSearch Key Prefix for the Index

    def __init__(self, r: redis.Redis, vector_dimensions: int, reset: bool = False):
        self.r = r
        self.create_index(vector_dimensions, reset)

    def create_index(self, vector_dimensions: int, reset_db: bool = False):
        try:
            # check to see if index exists
            self.r.ft(self.INDEX_NAME).info()
            err(f"Index {self.INDEX_NAME} already exists!")
            if reset_db:
                info(f"Existing index found. Dropping and recreating the index with new dimensions {vector_dimensions}.")
                self.r.ft(self.INDEX_NAME).dropindex(delete_documents=True)
                self.r.flushall()
                self.reset(vector_dimensions)
        except redis.exceptions.ResponseError:
            if reset_db:
                self.reset(vector_dimensions)

    def reset(self, vector_dimensions: int):
        schema = (
            TagField("tag"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dimensions,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.DOC_PREFIX], index_type=IndexType.HASH)
        self.r.ft(self.INDEX_NAME).create_index(fields=schema, definition=definition)

    def add_vector(self, doc_id: str, vector: list, tag: str):
        doc_key = f"{self.DOC_PREFIX}{doc_id}"
        debug(f"Adding vector for {doc_key} {len(vector)} dimensions tagged as {tag}")
        self.r.hset(doc_key, mapping={"vector": vector, "tag": tag})

    def search_vector(self, vector: list, top_n: int):
        try:
            query = Query(f"*=>[KNN {top_n} @vector $vec as score]").sort_by("score").return_fields("id", "score").paging(0, top_n).dialect(2)
            query_params = {"vec": vector}
            return self.r.ft(self.INDEX_NAME).search(query, query_params).docs
        except Exception as e:
            err(f"Error searching vector: {e}")
            return []

    def search_vectors(self, vectors: list, top_n: int) -> list:
        """Run KNN search for a batch of vectors concurrently.

        Redis-py checks out a connection from the pool per command and releases the GIL
        during socket I/O, so issuing searches from a thread pool avoids the serial
        round-trip-per-vector cost. Results preserve input order.
        """
        if not vectors:
            return []
        if len(vectors) == 1:
            return [self.search_vector(vectors[0], top_n)]

        max_workers = min(KNN_SEARCH_WORKERS, len(vectors))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda v: self.search_vector(v, top_n), vectors))
