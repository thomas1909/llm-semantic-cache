FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV SEMCACHE_AUTOSTART=1
EXPOSE 8500
CMD ["uvicorn", "semantic_cache.proxy:app", "--host", "0.0.0.0", "--port", "8500"]
