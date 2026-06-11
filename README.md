# Semantic Cache — proxy de cache sémantique pour APIs LLM

> **Projet 7 du guide BASWE.** Un middleware qui s'intercale entre ton application
> et n'importe quel provider LLM, détecte les requêtes **sémantiquement
> similaires** déjà répondues et les sert instantanément — latence quasi nulle,
> 30-60 % d'économies d'API sur des charges réalistes.

## Drop-in : zéro changement de code

Le proxy expose **exactement** le contrat `POST /v1/chat/completions` d'OpenAI.
Changer la base URL de ton client suffit. La réponse gagne deux headers :
`X-Cache: HIT|MISS` et `X-Cache-Similarity`.

```
app ──> proxy FastAPI ──┬─ HIT  (cosine ≥ seuil, même scope) ──> réponse cachée (~0 ms)
                        └─ MISS ──> provider réel ──> réponse + mise en cache
```

## Les pièges traités (ceux que les démos ignorent)

- **Pas de matching exact** : embeddings (hash 3-grammes déterministe offline,
  `text-embedding-3-small` si clé OpenAI) + normalisation de requête (strip des
  fillers "dis-moi", "explique-moi"…).
- **Pas de contamination croisée** : la clé de cache inclut le **hash du system
  prompt + modèle + temperature + max_tokens**. Deux apps avec le même prompt
  utilisateur mais des system prompts différents ne partagent jamais une entrée.
- **TTL par type de contenu** : les questions sensibles au temps ("météo
  aujourd'hui", "latest news") expirent en 1 h ; le stable en 24 h.
- **Seuils adaptatifs** : classification/extraction (espace de réponse contraint)
  → 0.90 ; génération créative → 0.99 ; défaut 0.95.
- **Invalidation** : `POST /v1/cache/invalidate` par scope (changement de version
  de system prompt ou de modèle).
- **Near-misses** : les requêtes juste sous le seuil sont comptées ;
  `POST /v1/cache/threshold-sweep` rejoue des paires sur plusieurs seuils pour
  visualiser le compromis hit-rate / fraîcheur — l'argument d'entretien clé.

## Démarrage

```bash
uv sync --extra dev --link-mode=copy
uv run --no-sync pytest -v          # 9 tests offline

$env:SEMCACHE_AUTOSTART="1"; uv run --no-sync uvicorn semantic_cache.proxy:app --port 8500
```

```bash
curl -s -D - -X POST localhost:8500/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Qu''est-ce que Python ?"}]}' | head -5
# 1er appel : X-Cache: MISS · 2e appel identique : X-Cache: HIT

curl localhost:8500/v1/stats     # hit_rate, saved_cost_usd, p95 des hits
curl localhost:8500/metrics      # format Prometheus (brancher Grafana dessus)
```

## Mode offline

Sans `OPENAI_API_KEY` : embeddings hash déterministes + upstream simulé (latence
artificielle 50 ms) — la démo HIT vs MISS et toutes les métriques fonctionnent
sans réseau.
