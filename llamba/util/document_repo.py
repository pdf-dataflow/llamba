import abc

from qdrant_client import QdrantClient, models
from .vectorizer import Vectorizer, ModelSettings, ModelType, Vector


class IDocumentRepo(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def check_connection(self) -> bool:
        ...

    @abc.abstractmethod
    def search(self, prompt: str) -> list[str]:
        ...


class DocumentRepoQdrant(IDocumentRepo):
    def __init__(self, client: QdrantClient, collection_name: str, used_models: ModelSettings, **kwargs):
        self.client = client
        self.collection_name = collection_name
        self.payload_return = kwargs.get("payload_return", "text")
        self.max_results = kwargs.get("max_results", 5)
        self.vectorizer = Vectorizer(used_models=used_models)

    def check_connection(self):
        return True if self.client.get_collection(self.collection_name) else False

    def __to_qdrant_vector(self, vector: Vector):
        if vector.type == ModelType.dense:
            return vector.data
        elif vector.type == ModelType.sparse:
            return models.SparseVector(
                indices=vector.data.indices,
                values=vector.data.values,
            )

    def __build_prefetch(self, vectors: dict, limit: int):
        return [
            models.Prefetch(
                query=self.__to_qdrant_vector(vector),
                using=name,
                limit=limit,
            )
            for name, vector in vectors.items()
        ]

    def __get_search_params(self, vectors: dict[str, Vector], limit: int) -> dict:
        if len(vectors) == 1:
            name, vector = iter(vectors.items())
            return {
                "query": self.__to_qdrant_vector(vector),
                "using": name,
            }
        return {
            "prefetch": self.__build_prefetch(vectors, limit=limit),
            "query": models.FusionQuery(fusion=models.Fusion.RRF)
        }

    def search(self, prompt: str):
        vectors = self.vectorizer.vectorize(prompt)
        search_params = self.__get_search_params(vectors, self.max_results)

        results = self.client.query_points(
            collection_name=self.collection_name,
            **search_params,
            limit=self.max_results,
            with_payload=True,
            with_vectors=False
        )
        result_data = [result.payload[self.payload_return] for result in results.points]
        return result_data
