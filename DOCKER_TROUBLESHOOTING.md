# Docker Troubleshooting (Gyandeep)

This guide captures the exact Docker/Postgres issues seen in this project and the working fix path.

## Symptoms

1. `docker` command not found in VS Code terminal.
2. `failed to connect to the docker API at npipe:////./pipe/docker_engine`.
3. Backend warning: `password authentication failed for user "postgres"`.
4. Backend startup crash: `psycopg2.errors.UndefinedTable: relation "plugin_jobs" does not exist`.

## Root Causes

1. Docker Desktop daemon was not running.
2. `docker compose` was run without `--env-file .env`, so compose fell back to default values.
3. Host already had another Postgres on port `5432`, causing credential/instance mismatch.
4. Fresh DB volume was created without applying the project schema.

## Working Configuration

Use these values in `.env`:

```dotenv
DB_HOST=127.0.0.1
DB_PORT=55432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=gyandeep
```

Why `55432`:

1. Avoids collision with local Postgres typically running on `5432`.

## Recovery Steps (Clean and Repeatable)

Run from project root:

```powershell
docker compose --env-file .env -f core/services/storage/docker/docker-compose.yaml down -v
docker compose --env-file .env -f core/services/storage/docker/docker-compose.yaml up -d
docker compose --env-file .env -f core/services/storage/docker/docker-compose.yaml ps
docker compose --env-file .env -f core/services/storage/docker/docker-compose.yaml exec -T db pg_isready -U postgres -d gyandeep
```

Expected port mapping in `ps` output:

```text
0.0.0.0:55432->5432/tcp
```

## Apply Schema (Required on Fresh Volume)

```powershell
Get-Content -Raw core/services/storage/schema.sql |
  docker compose --env-file .env -f core/services/storage/docker/docker-compose.yaml exec -T db psql -U postgres -d gyandeep
```

## Start Backend

```powershell
cd dashboard/backend
uvicorn app:app --reload
```

## Verify End-to-End

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/books).Content
```

Expected:

```json
{"books":[]}
```

## Quick Checks

1. Docker installed: `docker --version`
2. Docker engine running: `docker info`
3. Compose available: `docker compose version`

## Important Notes

1. Always include `--env-file .env` when running compose with `core/services/storage/docker/docker-compose.yaml`.
2. If backend shows DB auth errors after changes, restart backend so it reloads environment variables.
3. If `plugin_jobs` table errors appear, re-apply `core/services/storage/schema.sql`.
4. Rotate any API key that was accidentally exposed in `.env`.
