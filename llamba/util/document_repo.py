from typing import Optional

from qdrant_client import QdrantClient, models

from fastembed import TextEmbedding, SparseTextEmbedding


class DocumentRepoQdrant:
    def __init__(self, client: QdrantClient, collection_name: str, dense_vectorizer: TextEmbedding, dense_field: str,
                 sparse_vectorizer: Optional[SparseTextEmbedding] = None, sparse_field: Optional[str] = None, **kwargs):
        self.client = client
        self.collection_name = collection_name
        self.payload_return = kwargs.get("payload_return", "text")
        self.max_results = kwargs.get("max_results", 5)

        self.dense_vectorizer = dense_vectorizer
        self.dense_field = dense_field
        self.sparse_vectorizer = sparse_vectorizer
        self.sparse_field = sparse_field
        self.mode = "hybrid" if self.sparse_vectorizer else "dense"

    def check_connection(self):
        return True if self.client.get_collection(self.collection_name) else False

    def __dense_vectorize(self, text: str):
        raw_embedding = next(self.dense_vectorizer.query_embed(text))
        return raw_embedding

    def __sparse_vectorize(self, text: str):
        raw_embedding = next(self.sparse_vectorizer.query_embed(text))
        return models.SparseVector(
            indices=raw_embedding.indices,
            values=raw_embedding.values,
        )

    def __build_prefetch(self, text: str, limit: int):
        dense_prefetch = models.Prefetch(
            query=self.__dense_vectorize(text),
            using=self.dense_field,
            limit=limit
        )
        sparse_prefetch = models.Prefetch(
            query=self.__sparse_vectorize(text),
            using=self.sparse_field,
            limit=limit
        )
        return [dense_prefetch, sparse_prefetch]

    def __get_search_params(self, text: str, limit: int) -> dict:
        if self.sparse_vectorizer is None:
            return {
                "query": self.__dense_vectorize(text),
                "using": self.dense_field,
            }
        return {
            "prefetch": self.__build_prefetch(text, limit=limit),
            "query": models.FusionQuery(fusion=models.Fusion.RRF)
        }

    def search(self, prompt: str):
        search_params = self.__get_search_params(prompt, self.max_results)

        results = self.client.query_points(
            collection_name=self.collection_name,
            **search_params,
            limit=self.max_results,
            with_payload=True,
            with_vectors=False
        )
        result_data = [result.payload[self.payload_return] for result in results.points]
        return result_data
