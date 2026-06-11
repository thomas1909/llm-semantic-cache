"""The semantic cache itself: scoped nearest-neighbor lookup with TTL,
adaptive thresholds, invalidation, and near-miss tracking."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from .embeddings import Embedder, HashingEmbedder, cosine

_TIME_SENSITIVE = re.compile(
    r"\b(aujourd'hui|maintenant|actuel(le)?s?|cette semaine|ce mois|météo|news|"
    r"dernières nouvelles|today|now|current|latest|this week)\b", re.IGNORECASE)
_CREATIVE = re.compile(
    r"\b(écris|rédige|imagine|invente|compose|write|create a story|poem)\b",
    re.IGNORECASE)
_CONSTRAINED = re.compile(
    r"\b(classe|classifie|classify|oui ou non|yes or no|true or false|extrais|extract)\b",
    re.IGNORECASE)

DEFAULT_TTL = 24 * 3600
SHORT_TTL = 3600


def scope_key(model: str, system_prompt: str, temperature: float,
              max_tokens: int | None) -> str:
    """Two identical user prompts with different system prompts or params must
    NOT share cache entries — the scope hash prevents cross-contamination."""
    raw = json.dumps(
        {"model": model, "system": system_prompt, "temperature": temperature,
         "max_tokens": max_tokens}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def assign_ttl(prompt: str) -> int:
    return SHORT_TTL if _TIME_SENSITIVE.search(prompt) else DEFAULT_TTL


def adaptive_threshold(prompt: str, default: float = 0.95) -> float:
    """Constrained answer spaces tolerate looser matches; creative generation
    needs near-exactness (or no caching at all)."""
    if _CREATIVE.search(prompt):
        return 0.99
    if _CONSTRAINED.search(prompt):
        return 0.90
    return default


@dataclass
class CacheEntry:
    entry_id: str
    scope: str
    prompt: str
    embedding: list[float]
    response: dict
    created_at: float
    ttl: int
    hit_count: int = 0


@dataclass
class LookupResult:
    hit: bool
    entry: CacheEntry | None = None
    similarity: float = 0.0
    threshold: float = 0.95
    near_miss: bool = False


@dataclass
class Metrics:
    hits: int = 0
    misses: int = 0
    near_misses: int = 0
    invalidations: int = 0
    saved_cost_usd: float = 0.0
    hit_latencies_ms: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SemanticCache:
    def __init__(self, embedder: Embedder | None = None,
                 default_threshold: float = 0.95,
                 cost_per_request_usd: float = 0.002):
        self.embedder = embedder or HashingEmbedder()
        self.default_threshold = default_threshold
        self.cost_per_request_usd = cost_per_request_usd
        self._entries: dict[str, CacheEntry] = {}
        self.metrics = Metrics()

    def lookup(self, prompt: str, scope: str,
               threshold: float | None = None) -> LookupResult:
        start = time.perf_counter()
        threshold = threshold if threshold is not None else adaptive_threshold(
            prompt, self.default_threshold)
        query = self.embedder.embed(prompt)
        now = time.time()
        best: CacheEntry | None = None
        best_sim = 0.0
        for entry in self._entries.values():
            if entry.scope != scope:
                continue
            if now - entry.created_at > entry.ttl:
                continue
            sim = cosine(query, entry.embedding)
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is not None and best_sim >= threshold:
            best.hit_count += 1
            self.metrics.hits += 1
            self.metrics.saved_cost_usd += self.cost_per_request_usd
            self.metrics.hit_latencies_ms.append(
                (time.perf_counter() - start) * 1000)
            return LookupResult(hit=True, entry=best, similarity=best_sim,
                                threshold=threshold)
        self.metrics.misses += 1
        near = best is not None and best_sim >= threshold - 0.05
        if near:
            self.metrics.near_misses += 1
        return LookupResult(hit=False, entry=best, similarity=best_sim,
                            threshold=threshold, near_miss=near)

    def store(self, prompt: str, scope: str, response: dict,
              ttl: int | None = None) -> CacheEntry:
        entry = CacheEntry(
            entry_id=uuid.uuid4().hex[:12],
            scope=scope,
            prompt=prompt,
            embedding=self.embedder.embed(prompt),
            response=response,
            created_at=time.time(),
            ttl=ttl if ttl is not None else assign_ttl(prompt),
        )
        self._entries[entry.entry_id] = entry
        return entry

    def invalidate_scope(self, scope: str) -> int:
        """When a system prompt or model version changes, wipe its entries."""
        doomed = [k for k, e in self._entries.items() if e.scope == scope]
        for k in doomed:
            del self._entries[k]
        self.metrics.invalidations += len(doomed)
        return len(doomed)

    def purge_expired(self) -> int:
        now = time.time()
        doomed = [k for k, e in self._entries.items()
                  if now - e.created_at > e.ttl]
        for k in doomed:
            del self._entries[k]
        return len(doomed)

    def size(self) -> int:
        return len(self._entries)

    def threshold_sweep(self, pairs: list[tuple[str, str]],
                        thresholds: list[float]) -> dict[float, float]:
        """For tuning: given (query, cached_prompt) pairs, what fraction would
        hit at each threshold? Run on near-miss history to pick the tradeoff."""
        sims = [cosine(self.embedder.embed(a), self.embedder.embed(b))
                for a, b in pairs]
        return {t: sum(s >= t for s in sims) / len(sims) if sims else 0.0
                for t in thresholds}
