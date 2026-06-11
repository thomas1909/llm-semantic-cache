"""Embeddings, cache semantics, scoping, TTL, thresholds, proxy API."""

import time

from fastapi.testclient import TestClient

from semantic_cache.cache import (
    SemanticCache,
    adaptive_threshold,
    assign_ttl,
    scope_key,
)
from semantic_cache.embeddings import HashingEmbedder, cosine
from semantic_cache.proxy import DeterministicUpstream, create_app

SCOPE = scope_key("gpt-4o-mini", "tu es un assistant", 0.0, None)


def test_embedding_similarity_orders_correctly():
    e = HashingEmbedder()
    base = e.embed("Qu'est-ce que le langage Python ?")
    same = e.embed("qu'est-ce que le langage python")
    close = e.embed("Dis-moi ce qu'est le langage Python")
    far = e.embed("Quelle est la météo à Grenoble demain matin ?")
    assert cosine(base, same) > 0.99
    assert cosine(base, close) > cosine(base, far)
    assert cosine(base, far) < 0.6


def test_exact_and_paraphrase_hits():
    cache = SemanticCache()
    cache.store("Qu'est-ce que le langage Python ?", SCOPE, {"answer": "un langage"})
    assert cache.lookup("Qu'est-ce que le langage Python ?", SCOPE).hit
    # Filler-stripped paraphrase should also hit (query normalization).
    result = cache.lookup("Dis-moi ce qu'est le langage Python ?", SCOPE,
                          threshold=0.7)
    assert result.hit


def test_different_scope_never_cross_contaminates():
    cache = SemanticCache()
    cache.store("Qu'est-ce que Python ?", SCOPE, {"answer": "réponse A"})
    other_scope = scope_key("gpt-4o-mini", "réponds uniquement en anglais", 0.0, None)
    assert not cache.lookup("Qu'est-ce que Python ?", other_scope).hit
    temp_scope = scope_key("gpt-4o-mini", "tu es un assistant", 0.9, None)
    assert not cache.lookup("Qu'est-ce que Python ?", temp_scope).hit


def test_ttl_assignment_and_expiry():
    assert assign_ttl("Quelle est la météo aujourd'hui ?") == 3600
    assert assign_ttl("Qu'est-ce qu'une monade ?") == 24 * 3600
    cache = SemanticCache()
    entry = cache.store("question éphémère", SCOPE, {"answer": "x"}, ttl=1)
    entry.created_at = time.time() - 5  # simulate the clock advancing
    assert not cache.lookup("question éphémère", SCOPE).hit
    assert cache.purge_expired() == 1


def test_adaptive_thresholds():
    assert adaptive_threshold("Classe ce ticket en bug ou feature") == 0.90
    assert adaptive_threshold("Écris un poème sur la mer") == 0.99
    assert adaptive_threshold("Qu'est-ce que Python ?") == 0.95


def test_invalidation_by_scope():
    cache = SemanticCache()
    cache.store("q1", SCOPE, {"a": 1})
    cache.store("q2", SCOPE, {"a": 2})
    assert cache.invalidate_scope(SCOPE) == 2
    assert cache.size() == 0


def test_near_miss_tracking_and_sweep():
    cache = SemanticCache()
    cache.store("Comment installer Python sur Windows ?", SCOPE, {"a": 1})
    cache.lookup("Comment installer Python sous Windows 11 ?", SCOPE,
                 threshold=0.97)
    assert cache.metrics.misses == 1
    sweep = cache.threshold_sweep(
        [("Comment installer Python sur Windows ?",
          "Comment installer Python sous Windows 11 ?")],
        [0.5, 0.99])
    assert sweep[0.5] >= sweep[0.99]


def test_proxy_hit_miss_and_savings():
    upstream = DeterministicUpstream(latency_s=0.0)
    client = TestClient(create_app(SemanticCache(), upstream))
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "assistant"},
                     {"role": "user", "content": "Qu'est-ce que Python ?"}],
    }
    r1 = client.post("/v1/chat/completions", json=payload)
    assert r1.headers["X-Cache"] == "MISS"
    r2 = client.post("/v1/chat/completions", json=payload)
    assert r2.headers["X-Cache"] == "HIT"
    assert r2.json() == r1.json()
    assert upstream.calls == 1  # the second request never reached the provider

    stats = client.get("/v1/stats").json()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["saved_cost_usd"] > 0

    prom = client.get("/metrics").text
    assert "semcache_hits_total 1" in prom


def test_proxy_invalidation_endpoint():
    upstream = DeterministicUpstream(latency_s=0.0)
    client = TestClient(create_app(SemanticCache(), upstream))
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "v1 du system prompt"},
                     {"role": "user", "content": "Bonjour ?"}],
    }
    client.post("/v1/chat/completions", json=payload)
    r = client.post("/v1/cache/invalidate", json={
        "model": "gpt-4o-mini", "system_prompt": "v1 du system prompt"})
    assert r.json()["invalidated"] == 1
    client.post("/v1/chat/completions", json=payload)
    assert upstream.calls == 2  # cache was emptied for that scope
