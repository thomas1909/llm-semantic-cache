# CLAUDE.md — Semantic Cache (BASWE Project 7)

## Goal
Drop-in semantic caching proxy (OpenAI chat-completions contract). Scoped
nearest-neighbor lookup (system prompt + model + params hash), adaptive
similarity thresholds, TTL tiers, invalidation, near-miss analytics,
Prometheus metrics.

## Stack
Python 3.11 · `uv` · Pydantic v2 · FastAPI · httpx · pytest · ruff. Pure-python
vectors (dim 512) — no numpy needed offline.

## Modules (`src/semantic_cache/`)
- **embeddings.py** — Embedder protocol; HashingEmbedder (char-3-gram hashed,
  L2-normalized, + normalize_query filler stripping); OpenAIEmbedder; cosine().
- **cache.py** — scope_key(model, system, temperature, max_tokens) — prevents
  cross-contamination; assign_ttl (time-sensitive 1h / default 24h);
  adaptive_threshold (constrained 0.90 / default 0.95 / creative 0.99);
  SemanticCache.lookup/store/invalidate_scope/purge_expired/threshold_sweep;
  Metrics (hits, misses, near_misses within 0.05 below threshold, saved_cost).
- **proxy.py** — FastAPI: POST /v1/chat/completions (X-Cache headers),
  /v1/cache/invalidate, /v1/cache/threshold-sweep, /v1/stats, /metrics
  (Prometheus text). DeterministicUpstream (offline, simulated latency,
  call counter) vs OpenAIUpstream by env key.

## Commands
```bash
uv sync --extra dev --link-mode=copy
uv run --no-sync pytest -v
uv run --no-sync ruff check .
$env:SEMCACHE_AUTOSTART="1"; uv run --no-sync uvicorn semantic_cache.proxy:app --port 8500
```

## Hard rules
- Cache key MUST include system-prompt hash + params — never user text alone.
- Tests never hit a network (HashingEmbedder + DeterministicUpstream).
- Only complete upstream responses are cached (no partial/streaming bodies).
- ruff + pytest green before stopping.
