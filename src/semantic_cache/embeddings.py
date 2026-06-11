"""Embeddings behind a protocol: deterministic hashed char-3-gram vectors by
default (zero network, surprisingly decent for near-duplicate detection),
OpenAI text-embedding-3-small when an API key is configured."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

DIM = 512


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def normalize_query(text: str) -> str:
    """Strip filler so 'Dis-moi ce qu'est Python ?' ~ 'Qu'est-ce que Python'."""
    text = text.lower().strip()
    text = re.sub(
        r"^(dis[- ]moi|explique[- ]moi|peux[- ]tu me dire|please|s'il te plaît|stp)\s+",
        "", text)
    text = re.sub(r"[^\wà-ÿ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class HashingEmbedder:
    """Char 3-grams hashed into a fixed-dim count vector, L2-normalized."""

    def embed(self, text: str) -> list[float]:
        text = normalize_query(text)
        vec = [0.0] * DIM
        padded = f"  {text}  "
        for i in range(len(padded) - 2):
            gram = padded[i : i + 3]
            idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30)

    def embed(self, text: str) -> list[float]:
        resp = self._client.post("/embeddings",
                                 json={"model": self.model, "input": text})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def build_embedder(api_key: str = "") -> Embedder:
    return OpenAIEmbedder(api_key) if api_key else HashingEmbedder()
