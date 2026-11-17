"""API-based text embeddings; no local neural model download."""

import math
import os
import re

import requests


class EmbeddingError(Exception):
    """Safe, user-facing explanation without provider payloads or credentials."""


def model_name():
    name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", name):
        raise EmbeddingError("The embedding model configuration is invalid.")
    return name


def normalize(values):
    if not isinstance(values, list) or not values:
        raise EmbeddingError("The embedding service returned an invalid vector.")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in values):
        raise EmbeddingError("The embedding service returned an invalid vector.")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingError("The embedding service returned an empty vector.")
    return tuple(value / norm for value in values)


def embed_texts(texts, task_type, model=None):
    if not texts:
        return []
    model = model or model_name()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
        raise EmbeddingError("The embedding model configuration is invalid.")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise EmbeddingError("Semantic search needs a configured Gemini API key.")
    vectors = []
    # Bounded synchronous batches; each input gets its own embedding.
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
                headers={"x-goog-api-key": key},
                json={"requests": [{"model": f"models/{model}",
                                    "content": {"parts": [{"text": text}]},
                                    "taskType": task_type, "outputDimensionality": 768} for text in batch]},
                timeout=(5, 12),
            )
            if response.status_code == 429:
                raise EmbeddingError("Embedding quota reached; using keyword search.")
            if not response.ok:
                raise EmbeddingError("Embeddings are unavailable; using keyword search.")
            data = response.json().get("embeddings", [])
            if len(data) != len(batch):
                raise EmbeddingError("The embedding service returned an incomplete batch.")
            vectors.extend(normalize(item.get("values")) for item in data)
        except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
            raise EmbeddingError("Embeddings could not be retrieved; using keyword search.") from exc
    if len({len(vector) for vector in vectors}) != 1:
        raise EmbeddingError("The embedding dimensions do not match.")
    return vectors
