# RAG Observability Layer

> Production-grade monitoring, tracing, and CI regression-gating for LLM-based RAG pipelines.

[![Tests](https://github.com/Ashok007-cmd/rag-observability-layer/actions/workflows/monitor.yml/badge.svg)](https://github.com/Ashok007-cmd/rag-observability-layer/actions/workflows/monitor.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This package wraps any RAG pipeline with a non-invasive observability layer that provides:

- **Langfuse distributed tracing** — every step (`retrieve`, `rerank`, `generate`) produces a nested span with prompt content, retrieved chunks, token counts, and USD cost
- **OpenTelemetry metrics** — step latency histograms, error counters, query volume, context chunk distribution, and cost histograms exported to Prometheus/Grafana
- **Pluggable extension hooks** — clean lifecycle callbacks (`on_query_start`, `on_step_end`, `on_generation_llm_call`, …) to wire custom logic without modifying core code
- **Telemetry circuit breaker** — automatically bypasses tracing/metrics when the collector is down so pipeline latency is never degraded
- **Real-time guardrails** — keyword-based input/output filtering that raises immediately to block harmful or confidential content
- **CI regression gating** — baseline recording and automated threshold checks (latency, cost, faithfulness) that fail the build on degradation

## Architecture

```
Application Code
       │
       ▼
MonitoredRAGPipeline          ← thin wrapper, zero-copy delegation
  ├── Extensions (hooks)
  │     ├── LangfuseTracingExtension  → Langfuse UI   (traces)
  │     ├── OTelMetricsExtension      → Prometheus     (metrics)
  │     └── GuardrailExtension        → raises ValueError
  │
  └── RAGPipeline (your existing pipeline)
        ├── _retrieve()    ← instrumented in-place
        ├── _apply_reranker()
        └── generator.generate()
```

**Data flow:** each pipeline step fires `on_step_start` / `on_step_end` hooks → the Langfuse extension creates a nested span, the OTel extension records a latency histogram point. Token usage is intercepted transparently from OpenAI / Anthropic clients via a thread-safe `ContextVar` patch.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Or install as an editable package (after `pyproject.toml` is present):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from monitoring import MonitoredRAGPipeline, settings
from monitoring import OTelMetricsExtension, GuardrailExtension

monitored = MonitoredRAGPipeline(
    pipeline=your_rag_pipeline,
    extensions=[
        OTelMetricsExtension(),
        GuardrailExtension(blocked_keywords=["internal_secret"]),
    ],
)

# Standard query — fully traced and metered
answer, citations = monitored.query("How does RAG work?", use_reranker=True)

# Streaming query — measures time-to-first-token
for chunk in monitored.query_stream("Explain embeddings."):
    print(chunk, end="", flush=True)
```

### Custom extension

```python
from monitoring import BaseExtension

class SlackAlerter(BaseExtension):
    def on_query_error(self, exc: Exception) -> None:
        slack_client.post(f"RAG query failed: {exc}")

monitored = MonitoredRAGPipeline(pipeline=..., extensions=[SlackAlerter()])
```

## Configuration

All settings use the `MONITOR_` environment variable prefix. Copy `.env.example` to `.env` and fill in your values:

| Variable | Default | Description |
|---|---|---|
| `MONITOR_ENABLED` | `true` | Master on/off switch |
| `MONITOR_LANGFUSE_SECRET_KEY` | `""` | Langfuse secret key (`sk-lf-…`) |
| `MONITOR_LANGFUSE_PUBLIC_KEY` | `""` | Langfuse public key (`pk-lf-…`) |
| `MONITOR_LANGFUSE_HOST` | `http://localhost:3000` | Langfuse instance URL |
| `MONITOR_LANGFUSE_RELEASE` | `dev` | Release / branch label |
| `MONITOR_OTEL_SERVICE_NAME` | `rag-pipeline` | OTel service name |
| `MONITOR_OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTel collector endpoint |
| `MONITOR_PRICING_FILE` | `pricing.json` | Per-token pricing data file |
| `MONITOR_BASELINE_DIR` | `data/monitoring` | CI baseline storage directory |
| `MONITOR_CIRCUIT_BREAKER_THRESHOLD` | `3` | Failures before circuit opens |
| `MONITOR_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `30.0` | Cooldown before half-open retry |
| `MONITOR_MAX_QUEUE_SIZE` | `10000` | Async telemetry queue depth |

## Local Observability Stack

Spin up Langfuse, OTel Collector, Prometheus, and Grafana with a single command:

```bash
# Copy and fill in secrets first
cp .env.example .env
# Start the stack
docker compose -f infra/docker-compose.yml up -d
# Verify all services are reachable
python scripts/verify_infra.py
```

| Service | URL |
|---|---|
| Langfuse UI | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |
| OTel Collector (HTTP) | http://localhost:4318 |

> **Security:** The compose file requires `POSTGRES_PASSWORD`, `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, and `GF_ADMIN_PASSWORD` to be set as environment variables — it will refuse to start if any are missing.

## Metrics Reference

| Metric | Type | Labels | Description |
|---|---|---|---|
| `rag_retrieve_latency_seconds` | Histogram | — | Vector DB retrieval latency |
| `rag_rerank_latency_seconds` | Histogram | — | Re-ranker latency |
| `rag_generate_latency_seconds` | Histogram | — | LLM generation latency |
| `rag_query_total_latency_seconds` | Histogram | — | End-to-end query latency |
| `rag_{step}_errors_total` | Counter | `error_type` | Errors per step |
| `rag_queries_total` | Counter | — | Total query volume |
| `rag_context_chunks_per_query` | Histogram | — | Retrieved chunk count distribution |
| `rag_cost_per_query_dollars` | Histogram | — | USD cost per query |
| `rag_tokens_{prompt\|completion\|total}` | Histogram | — | Token usage distribution |
| `rag_time_to_first_token_seconds` | Histogram | — | Streaming TTFT |

## CI Regression Gating

The regression gate fails the build if any metric degrades more than 20% from the recorded baseline.

```bash
# 1. Record the current baseline (runs once per branch)
python scripts/record_baseline.py --branch main

# 2. After a change, check for regressions
python scripts/check_regressions.py --branch main
```

**Prompt change detection:** the gate also computes a SHA-256 of the system prompt template. If the hash differs from the baseline, the PR comment is annotated with a warning so the team consciously validates behavior before merging.

The GitHub Actions workflow (`.github/workflows/monitor.yml`) runs this automatically on every push and pull request.

## Adding a New LLM Model / Provider

Model pricing is **data-only** — no code changes required:

```json
// pricing.json
{
  "my-provider": {
    "my-model-v1": { "input": 0.002, "output": 0.006 }
  }
}
```

The `PricingConfig` class picks up file changes automatically via mtime-based caching.

## Running Tests

```bash
pytest tests/ -v --cov=monitoring --cov=scripts --cov-report=term-missing
```

62 tests, ~75% coverage. All tests are self-contained — no external services required.

## Demo

Run an end-to-end demo using mock pipeline and generators (no API keys needed):

```bash
python scripts/demo_observability.py
```

## Project Structure

```
rag-observability-layer/
├── monitoring/               # Core package
│   ├── config.py             # Settings (pydantic-settings) + PricingConfig
│   ├── tracing.py            # Langfuse Tracer wrapper
│   ├── metrics.py            # OTel MetricsCollector
│   ├── prompts.py            # SHA-256 prompt version registry
│   ├── extensions.py         # BaseExtension + CircuitBreaker + async worker
│   └── wrappers.py           # MonitoredRAGPipeline
├── scripts/
│   ├── record_baseline.py    # CI: save current metrics as baseline
│   ├── check_regressions.py  # CI: compare metrics vs baseline, exit 1 on fail
│   ├── verify_infra.py       # Health-check the local observability stack
│   ├── verify_gate.py        # End-to-end integration verification
│   └── demo_observability.py # Runnable demo with mock pipeline
├── tests/                    # pytest unit tests (62 tests)
├── infra/
│   ├── docker-compose.yml    # Full local stack (Langfuse, OTel, Prometheus, Grafana)
│   ├── otel-collector/       # OTel Collector config
│   ├── prometheus/           # Prometheus scrape config
│   └── grafana/              # Dashboard + datasource provisioning
├── data/
│   └── monitoring/           # Runtime baseline storage (gitignored)
├── pricing.json              # Per-token cost data (OpenAI, Anthropic)
├── pyproject.toml            # Package config + pytest/coverage settings
├── requirements.txt          # Runtime + dev dependencies
└── .env.example              # Environment variable template
```

## License

[MIT](LICENSE)
