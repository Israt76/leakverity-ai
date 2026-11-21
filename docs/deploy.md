# Deploying

## Hugging Face Spaces (Docker deployment)

1. Create a new Space, choose **Docker** as the SDK and **CPU basic** hardware.
2. Push this repository to the Space. Add this front matter to the top of the Space's `README.md`:

```yaml
---
title: LeakVerity
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

## Render / any container host

```bash
docker build -t leakverity .
docker run -p 7860:7860 leakverity
```

`PORT` is honoured if your host sets it.

## Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

The saved demo audits (`examples/demo_result_*.json`) load instantly, so the demo works even on slow hardware. Re-generate them with `python examples/refresh_demo.py`.
