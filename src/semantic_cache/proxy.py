"""Drop-in proxy mirroring the OpenAI chat-completions contract.

Switch your app to this base URL and nothing else changes; the response gains
an `X-Cache: HIT|MISS` header. Providers are routed on the `model` field, with
a deterministic upstream when no API key is configured (offline mode)."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Protocol

import httpx
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from .cache import SemanticCache, scope_key


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = None
    user: str | None = None


class Upstream(Protocol):
    def complete(self, req: ChatRequest) -> dict: ...


class DeterministicUpstream:
    """Offline LLM stand-in with a visible per-prompt fingerprint and a
    simulated latency so cached-vs-uncached numbers mean something."""

    def __init__(self, latency_s: float = 0.05):
        self.latency_s = latency_s
        self.calls = 0

    def complete(self, req: ChatRequest) -> dict:
        self.calls += 1
        time.sleep(self.latency_s)
        user_text = " ".join(m.content for m in req.messages if m.role == "user")
        digest = hashlib.sha1(user_text.encode()).hexdigest()[:8]
        content = f"[{req.model}] réponse({digest}) : {user_text[:120]}"
        n_in = sum(len(m.content.split()) for m in req.messages)
        return {
            "id": f"chatcmpl-{digest}",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": n_in,
                      "completion_tokens": len(content.split()),
                      "total_tokens": n_in + len(content.split())},
        }


class OpenAIUpstream:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}, timeout=120)

    def complete(self, req: ChatRequest) -> dict:
        resp = self._client.post(
            "/chat/completions", json=req.model_dump(exclude_none=True))
        resp.raise_for_status()
        return resp.json()


class InvalidateRequest(BaseModel):
    model: str
    system_prompt: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None


class SweepRequest(BaseModel):
    pairs: list[tuple[str, str]]
    thresholds: list[float] = Field(
        default_factory=lambda: [0.90, 0.93, 0.95, 0.98])


def create_app(cache: SemanticCache | None = None,
               upstream: Upstream | None = None) -> FastAPI:
    app = FastAPI(title="Semantic Cache Proxy", version="0.1.0")
    cache = cache or SemanticCache()
    if upstream is None:
        key = os.getenv("OPENAI_API_KEY", "")
        upstream = OpenAIUpstream(key) if key else DeterministicUpstream()
    app.state.cache = cache
    app.state.upstream = upstream

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest, response: Response):
        system = " ".join(m.content for m in req.messages if m.role == "system")
        user_text = " ".join(m.content for m in req.messages if m.role == "user")
        scope = scope_key(req.model, system, req.temperature, req.max_tokens)
        result = cache.lookup(user_text, scope)
        if result.hit and result.entry is not None:
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Similarity"] = f"{result.similarity:.4f}"
            return result.entry.response
        data = upstream.complete(req)
        cache.store(user_text, scope, data)
        response.headers["X-Cache"] = "MISS"
        return data

    @app.post("/v1/cache/invalidate")
    def invalidate(req: InvalidateRequest):
        scope = scope_key(req.model, req.system_prompt, req.temperature,
                          req.max_tokens)
        return {"invalidated": cache.invalidate_scope(scope)}

    @app.post("/v1/cache/threshold-sweep")
    def threshold_sweep(req: SweepRequest):
        return {f"{t:.2f}": rate
                for t, rate in cache.threshold_sweep(req.pairs, req.thresholds).items()}

    @app.get("/v1/stats")
    def stats():
        m = cache.metrics
        lat = sorted(m.hit_latencies_ms)
        p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
        return {
            "hits": m.hits, "misses": m.misses, "hit_rate": round(m.hit_rate, 4),
            "near_misses": m.near_misses, "invalidations": m.invalidations,
            "saved_cost_usd": round(m.saved_cost_usd, 6),
            "cache_size": cache.size(), "hit_latency_p95_ms": round(p95, 3),
        }

    @app.get("/metrics")
    def metrics_prometheus():
        m = cache.metrics
        body = "\n".join([
            f"semcache_hits_total {m.hits}",
            f"semcache_misses_total {m.misses}",
            f"semcache_near_misses_total {m.near_misses}",
            f"semcache_saved_cost_usd {m.saved_cost_usd:.6f}",
            f"semcache_entries {cache.size()}",
        ]) + "\n"
        return Response(content=body, media_type="text/plain")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app() if os.getenv("SEMCACHE_AUTOSTART") else None
