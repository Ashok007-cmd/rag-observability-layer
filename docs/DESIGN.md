# Monitoring & Observability Layer — Design

## 1. Overview

This package adds a production-grade monitoring and observability layer around any RAG pipeline. It covers three concerns: Langfuse-based tracing of every pipeline step, OpenTelemetry metrics for production observability, and CI regression gating for latency/cost/quality.

**Success criteria:**
- Every RAG pipeline step is traced with prompts, retrieved chunks, and token usage visible in Langfuse
- P50/P95 latency, per-request cost, and error rates are exported via OTel
- CI pipeline fails the build if latency/cost/faithfulness regresses beyond threshold
- Prompt changes trigger conscious CI gates (no silent drift)

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│              monitoring/ package               │
│  ┌────────────────────┐ ┌──────────────────┐  │
│  │  Tracing Layer      │ │  Metrics Layer   │  │
│  │  (Langfuse SDK)     │ │  (OTel + Prom)   │  │
│  └────────┬─────────────┘ └────────┬─────────┘  │
└───────────┼──────────────────────┼────────────┘
            │                      │
┌───────────┼──────────────────────┼────────────┐
│  Your RAG Pipeline                │            │
│  ┌──────┐ ┌──────┐ ┌──────┐      │            │
│  │Load  │ │Chunk │ │Embed │      │            │
│  └──┬───┘ └──┬───┘ └──┬───┘      │            │
│     │        │        │          │            │
│  ┌──▼───┐ ┌──▼───┐ ┌──▼───┐     │            │
│  │Retr. │ │Rerank│ │Gen.  │     │            │
│  └──────┘ └──────┘ └──────┘     │            │
└────────────────────────────────┼────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Infrastructure      │
                    │  ┌────────┐ ┌────────┐  │
                    │  │Grafana │ │Langfuse│  │
                    │  │(OTel)  │ │ (UI)   │  │
                    │  └────────┘ └────────┘  │
                    └─────────────────────────┘
```

**Data flow:**
1. Application code calls `MonitoredRAGPipeline.query()`
2. Each pipeline step emits a Langfuse span (trace) and an OTel metric (latency, tokens, cost)
3. Langfuse stores traces for per-request debugging; Prometheus scrapes OTel metrics
4. Grafana dashboards visualize P50/P95 latency, error rates, cost trends
5. CI runs `check_regressions.py` against stored baselines, fails on degradation

## 3. API Surface

### monitoring.config

```python
settings = MonitoringSettings()    # env-configured settings singleton
PricingConfig(pricing_path)        # loads pricing.json, computes cost per request
PricingConfig.get_cost(provider, model, prompt_tokens, completion_tokens) -> float
```

### monitoring.tracing

```python
Tracer(enabled=True)                              # Langfuse client wrapper
tracer.trace_step(name, input, metadata)           # context manager yielding span
tracer.capture_generation(span, model, provider, prompt, query, response, tokens...)
tracer.get_trace_id() -> str                       # stable trace ID for the request
```

### monitoring.metrics

```python
MetricsCollector(enabled=True, prefix="rag")       # OTel meter wrapper
mc.record_latency(step, seconds)                   # records to histogram
mc.record_error(step, error_type)                  # increments error counter
mc.record_query_count()                            # increments query counter
mc.record_context_count(n)                         # records to context histogram
mc.record_cost(cost)                                # records to cost histogram
mc.record_tokens(prompt, completion, total)         # records to token histograms
```

### monitoring.wrappers

```python
MonitoredRAGPipeline(pipeline, tracer, metrics)    # wraps RAGPipeline
monitored.query(question, top_k, use_hybrid, use_reranker) -> (answer, citations)
monitored.ingest(source) -> int
```

### monitoring.prompts

```python
PromptRegistry(persist_path)                       # version-controlled prompt store
registry.register(name, prompt, metadata) -> hash  # stores version, returns SHA256
registry.current_hash(name) -> str | None           # latest hash for prompt
registry.detect_change(name, new_prompt) -> bool    # True if hash differs
registry.get_versions(name) -> list[PromptVersion]  # full version history
```

## 4. Instrumentation

### Tracing design

Each pipeline step is wrapped in a Langfuse span:

```
query (root span)
├── retrieve  (sub-span: query text, top_k, scores returned, doc metadata)
├── rerank    (sub-span: model, pre/post scores, chunk IDs selected)
└── generate  (sub-span: system prompt, context snippets, response, token count, cost)
```

**Document metadata enhancement:** When logging chunk IDs in `retrieve()`, source titles and filenames are appended to the span metadata. This enables retrieval quality evaluation directly in the Langfuse UI without cross-referencing the vector database.

### Cost tracking

- Pricing data is stored in a decoupled `pricing.json` file (not hardcoded)
- Per-request cost = `(prompt_tokens * input_price) + (completion_tokens * output_price)`
- Costs are logged on Langfuse spans and recorded in OTel histograms

## 5. Production Metrics

### OpenTelemetry metric definitions

| Metric | Type | Tags | Purpose |
|--------|------|------|---------|
| `rag_{step}_latency_seconds` | Histogram | step | Per-step latency for P50/P95 |
| `rag_query_total_latency_seconds` | Histogram | — | End-to-end query latency |
| `rag_{step}_errors_total` | Counter | step, error_type | Error rate per step |
| `rag_queries_total` | Counter | — | Total query volume |
| `rag_context_chunks_per_query` | Histogram | — | Context usage distribution |
| `rag_cost_per_query_dollars` | Histogram | — | Cost distribution |
| `rag_tokens_{type}` | Histogram | type | Prompt/completion/total tokens |

### Decoupled pricing

Model → cost mappings are stored in `pricing.json`, loaded by `PricingConfig`. Adding a new model is a data change, not a code change.

## 6. CI Regression Gating

### Baseline schema

```json
{
  "main": {
    "timestamp": "2026-05-25T12:00:00",
    "prompt_hash": "abc123def456",
    "p50_latency": 0.452,
    "p95_latency": 1.234,
    "avg_cost": 0.0025,
    "avg_faithfulness": 0.85,
    "pass_rate": 0.8,
    "total_queries": 10,
    "errors": 0
  }
}
```

**Branch isolation:** Each branch stores its own baseline under its branch name. PR branches never overwrite main's baseline. New baselines are only written when no baseline exists for the branch.

### CI workflow steps

1. Lint (`ruff`), type-check (`mypy`), and dependency-audit (`pip-audit`) the package
2. Install monitoring package and run the unit test suite
3. Run the demo/evaluation flow with monitoring enabled
4. Record baseline for branch if none exists
5. Check regressions against baseline (fail if any metric degrades >20%)
6. Upload artifacts and post a PR comment

### Prompt change detection

The `PromptRegistry` computes SHA256 hashes of rendered prompt templates. When a hash differs from the stored baseline, the CI gate annotates the PR comment with "prompt template changed — verify behavior".

## 7. Infrastructure

### Docker Compose services

| Service | Image | Host Port | Memory Limit |
|---------|-------|-----------|---------------|
| `postgres` | postgres:16-alpine | *(internal only)* | 1G |
| `langfuse` | langfuse/langfuse:3 | 127.0.0.1:3000 | 512M |
| `otel-collector` | otel/opentelemetry-collector-contrib:0.154.0 | 127.0.0.1:4317/4318/8889 | 256M |
| `prometheus` | prom/prometheus:v2.53.5 | 127.0.0.1:9090 | 512M |
| `grafana` | grafana/grafana:13.1.0 | 127.0.0.1:3001 | 512M |

Images are version-pinned (no `:latest`) for reproducibility, and dev-only ports are bound to `127.0.0.1` rather than `0.0.0.0`. One-command local stack: `scripts/start_monitoring_stack.sh`.

## 8. Configuration

All config via environment variables with `MONITOR_` prefix — see the [README configuration table](../README.md#configuration) for the current, authoritative list.

## 9. Testing Strategy

| Module | Test approach | Key scenarios |
|--------|--------------|---------------|
| `config.py` | Unit tests with real pricing.json | Cost calculation for known/unknown models, key-prefix rejection |
| `tracing.py` | Unit tests with mocked Langfuse | Span creation, generation capture, error propagation |
| `wrappers.py` | Unit tests with mocked pipeline | Query delegation, latency recording, error recording |
| `metrics.py` | Unit tests with mocked OTel | Histogram/counter recording, noop when disabled |
| `prompts.py` | Unit tests with in-memory registry | Registration, change detection, version history |
| `check_regressions.py` | Unit tests with synthetic baselines | Pass/fail on each metric, threshold enforcement |
| Integration | `evaluate.py` with monitoring enabled | End-to-end: trace + metrics + CI output |
