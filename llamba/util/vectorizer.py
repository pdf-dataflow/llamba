from typing import TypeAlias, Iterable

from fastembed import TextEmbedding, SparseTextEmbedding
from dataclasses import dataclass
from enum import StrEnum

CollectionField: TypeAlias = str


class ModelType(StrEnum):
    dense = "dense"
    sparse = "sparse"


@dataclass
class ModelCfg:
    name: str
    path: str
    type: ModelType


ModelSettings = dict[
    CollectionField,
    ModelCfg
]

import numpy as np
from fastembed import SparseTextEmbedding

@dataclass
class Vector:
    data: np.ndarray | SparseTextEmbedding
    type: ModelType


class Vectorizer:
    def __init__(self, used_models: ModelSettings):
        self.__used_models_settings = used_models
        self.used_models = self.__init_models(used_models)

    @staticmethod
    def __init_models(models_cfg: ModelSettings) -> dict[
        CollectionField, TextEmbedding | SparseTextEmbedding]:
        models = {}

        for field, model_cfg in models_cfg.items():
            if model_cfg.type == ModelType.sparse:
                model = SparseTextEmbedding(
                    model_name=model_cfg.name,
                    specific_model_path=model_cfg.path,
                )
            elif model_cfg.type == ModelType.dense:
                model = TextEmbedding(
                    model_name=model_cfg.name,
                    specific_model_path=model_cfg.path,
                )

            models[field] = model

        return models

    def vectorize(self, text: str) -> dict[CollectionField, Vector]:
        result = {}
        for field, model in self.used_models.items():
            vector = model.query_embed(text)
            result[field] = Vector(
                next(vector),
                self.__used_models_settings[field].type
            )
        return result
