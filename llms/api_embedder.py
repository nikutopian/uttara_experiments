import os
from enum import Enum
from typing import Any, List

from openai import OpenAI
from sentence_transformers import SentenceTransformer


class Embedder:
    def get_embedding(self, input_text: str):
        raise NotImplementedError("should be implemented in sub-class")

    def get_embeddings(self, input_texts: List[str]):
        raise NotImplementedError("should be implemented in sub-class")


class ApiEmbedder(Embedder):
    def __init__(self, client: Any, model: Enum):
        self.model = model
        self.client = client
        self.seed = 42
        self.temperature = 0.0

    def get_embedding(self, input_text: str):
        input_text = input_text.replace("\n", "")
        response = self.client.embeddings.create(
            model=self.model.value, input=[input_text]
        )
        return response.data[0].embedding

    def get_embeddings(self, input_texts: List[str]):
        # input_text = input_text.replace("\n", "")
        response = self.client.embeddings.create(
            model=self.model.value, input=input_texts
        )
        return [d.embedding for d in response.data]


class OpenAIEmbeddingModelType(Enum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class OpenAIEmbedder(ApiEmbedder):
    def __init__(self, model: OpenAIEmbeddingModelType):
        super().__init__(
            client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), model=model
        )

class SentenceTransformerEmbeddingModelType(Enum):
    MXBAI_EMBED_LARGE_V1 = "mixedbread-ai/mxbai-embed-large-v1"

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_type: SentenceTransformerEmbeddingModelType):
        self.model = SentenceTransformer(model_type.value)

    def get_embedding(self, input_text: str):
        input_text = input_text.replace("\n", "")
        embeddings = self.model.encode([input_text])
        return embeddings[0]

    def get_embeddings(self, input_texts: List[str]):
        input_texts = [x.replace("\n", "") for x in input_texts]
        embeddings = self.model.encode(input_texts)
        return embeddings
