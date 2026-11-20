# Running untrusted notebooks safely

LeakVerity executes uploaded code, so how it runs matters.

## Uploads are off by default

`LEAKVERITY_ALLOW_UPLOADS` defaults to `0` everywhere, in the Docker image and when you run `uvicorn` directly. To audit your own notebooks on your own machine, start the server with `LEAKVERITY_ALLOW_UPLOADS=1`. Only enable it on a public host together with the container sandbox.

## Jobs and retention

Each audit gets an unguessable 128-bit job id. Whoever holds the id can read the results, which include the uploaded code. Jobs are kept in memory for at most `LEAKVERITY_JOB_TTL` seconds (default 2 hours), at most `LEAKVERITY_MAX_JOBS` (default 50) are kept, and nothing is written to disk. There are no user accounts: do not upload confidential notebooks to a shared deployment.

## Public demo

`LEAKVERITY_ALLOW_UPLOADS=0`: the upload form is disabled and `/api/audit` returns 403. Visitors can open the saved audits and run both demo notebooks live, but no visitor-supplied code ever executes. Use this for Demo Day and any public link.

## Process sandbox (`LEAKVERITY_SANDBOX=process`, the default for local use)

Each experiment runs as a separate `python -I` process with:

- a fresh temporary working directory, deleted afterwards;
- network sockets disabled inside the interpreter;
- `RLIMIT_CPU`, `RLIMIT_AS` (6 GB) and `RLIMIT_FSIZE` (512 MB) limits, plus a wall-clock timeout (300 s);
- a minimal environment, so API keys and other secrets are not passed through (tested);
- one audit at a time (`LEAKVERITY_CONCURRENCY=1`).

This prevents accidents and casual abuse. It is **not** a hard security boundary against a determined attacker. Use it for your own notebooks.

## Container sandbox (`LEAKVERITY_SANDBOX=docker`), for accepting public uploads

Each run starts a disposable container from `Dockerfile.sandbox`:

```bash
docker build -f Dockerfile.sandbox -t leakverity-sandbox .
LEAKVERITY_SANDBOX=docker LEAKVERITY_ALLOW_UPLOADS=1 uvicorn app.main:app --port 7860
```

The container runs with `--network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 256`, a memory and CPU cap, a tmpfs `/tmp`, a non-root user, and only the job directory mounted.

> Status: this mode is implemented in `leakverity/sandbox.py` but was **not** exercised in the development environment, which had no Docker daemon. Run the test suite with `LEAKVERITY_SANDBOX=docker` on a Docker host before relying on it. The server needs access to the Docker socket, so run the server itself on a dedicated host.

## Data and AI

Datasets are never sent to an external AI provider. The AI reader receives column names, a few example values and univariate AUCs, plus the notebook's markdown text for claim reading. Leave the API key empty to keep everything local.
