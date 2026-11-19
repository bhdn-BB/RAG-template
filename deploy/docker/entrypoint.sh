#!/bin/sh
# API container entrypoint: run pgvector migrations (only for that backend), then start uvicorn.
set -e

if [ "${VECTOR_STORE}" = "pgvector" ]; then
    echo "Running Alembic migrations for pgvector..."
    attempt=1
    until alembic upgrade head; do
        if [ "$attempt" -ge 20 ]; then
            echo "Database is not reachable, giving up." >&2
            exit 1
        fi
        echo "Database not ready yet (attempt $attempt), retrying in 3s..."
        attempt=$((attempt + 1))
        sleep 3
    done
fi

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${WORKERS:-2}" \
    --proxy-headers \
    --forwarded-allow-ips "*"
