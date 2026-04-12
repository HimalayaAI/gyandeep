#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${ROOT_DIR}/core/services/storage/docker/docker-compose.yaml"
SCHEMA_FILE="${ROOT_DIR}/core/services/storage/schema.sql"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  . "${ROOT_DIR}/.env"
  set +a
fi

DB_NAME="${DB_NAME:-gyandeep}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
DB_PORT="${DB_PORT:-5432}"

compose() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

export DB_NAME DB_USER DB_PASSWORD DB_PORT

echo "Starting pgvector Postgres..."
compose -f "$COMPOSE_FILE" up -d

echo "Waiting for database to be ready..."
for _ in {1..30}; do
  if compose -f "$COMPOSE_FILE" exec -T db pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Ensuring database exists..."
compose -f "$COMPOSE_FILE" exec -T db psql -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
  compose -f "$COMPOSE_FILE" exec -T db psql -U "$DB_USER" -d postgres -c "CREATE DATABASE ${DB_NAME}"

echo "Applying schema..."
compose -f "$COMPOSE_FILE" exec -T db psql -U "$DB_USER" -d "$DB_NAME" < "$SCHEMA_FILE"

echo "Done. Database '${DB_NAME}' is ready with pgvector schema."

if [[ -n "${APP_CMD:-}" ]]; then
  echo "Starting app via APP_CMD..."
  exec bash -lc "$APP_CMD"
fi

if [[ -f "${ROOT_DIR}/dashboard/backend/app.py" ]]; then
  echo "Starting app: dashboard backend"
  exec python -m dashboard.backend.app
fi

if [[ -f "${ROOT_DIR}/app.py" ]]; then
  echo "Starting app: app.py"
  exec python "${ROOT_DIR}/app.py"
fi

if [[ -f "${ROOT_DIR}/main.py" ]]; then
  echo "Starting app: main.py"
  exec python "${ROOT_DIR}/main.py"
fi

if [[ -f "${ROOT_DIR}/server.py" ]]; then
  echo "Starting app: server.py"
  exec python "${ROOT_DIR}/server.py"
fi

echo "No app entrypoint found. Set APP_CMD to run the Gyandeep app."
