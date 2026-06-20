# Monitoring & Observability Layer — SPEC

## 1. Overview

This project adds a production-grade monitoring and observability layer to the RAG pipeline from Project 1. It covers three phases: Langfuse-based tracing of every pipeline step, OpenTelemetry metrics for production observability, and CI regression gating for latency/cost/quality.

**Success criteria:**
- Every RAG pipeline step is traced with prompts, retrieved chunks, and token usage visible in Langfuse
- P50/P95 latency, per-request cost, and error rates are exported via OTel
- CI pipeline fails the build if latency/cost/faithfulness regresses beyond threshold
- Prompt changes trigger conscious CI gates (no silent drift)

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│              monitoring/ package              │
│  ┌────────────────────┐ ┌──────────────────┐  │
│  │  Tracing Layer     │ │  Metrics Layer   │  │
│  │  (Langfuse SDK)    │ │  (OTel + Prom)   │  │
│  └────────┬───────────┘ └────────┬─────────┘  │
└───────────┼──────────────────────┼────────────┘
            │                      │
┌───────────┼──────────────────────┼────────────┐
│  RAG Pipeline (Project 1)       │            │
│  ┌──────┐ ┌──────┐ ┌──────┐     │            │
│  │Load  │ │Chunk │ │Embed │     │            │
│  └──┬───┘ └──┬───┘ └──┬───┘     │            │
│     │        │        │         │            │
│  ┌──▼───┐ ┌──▼───┐ ┌──▼───┐    │            │
│  │Retr. │ │Rerank│ │Gen.  │    │            │
│  └──────┘ └──────┘ └──────┘    │            │
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
mc.record_cost(cost)                               # records to cost histogram
mc.record_tokens(prompt, completion, total)        # records to token histograms
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
registry.current_hash(name) -> str | None          # latest hash for prompt
registry.detect_change(name, new_prompt) -> bool   # True if hash differs
registry.get_versions(name) -> list[PromptVersion]  # full version history
```

## 4. Phase 1: Instrumentation

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

## 5. Phase 2: Production Metrics

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

## 6. Phase 3: CI Regression Gating

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

1. Spin up Langfuse + PostgreSQL via Docker service containers (memory-limited to 1G)
2. Install monitoring package and RAG pipeline
3. Ingest sample documents
4. Record baseline for branch if none exists
5. Run evaluation with monitoring enabled
6. Check regressions against baseline (fail if any metric degrades >20%)
7. Upload artifacts and post PR comment

### Prompt change detection

The `PromptRegistry` computes SHA256 hashes of rendered prompt templates. When a hash differs from the stored baseline, the CI gate annotates the PR comment with "prompt template changed — verify behavior".

## 7. Infrastructure

### Docker Compose services

| Service | Image | Port | Memory Limit |
|---------|-------|------|-------------|
| `postgres` | postgres:16-alpine | 5432 | 1G |
| `langfuse` | langfuse/langfuse:latest | 3000 | 512M |
| `prometheus` | prom/prometheus:latest | 9090 | — |
| `grafana` | grafana/grafana:latest | 3001 | — |

One-command local stack: `scripts/start_monitoring_stack.sh`

## 8. Configuration

All config via environment variables with `MONITOR_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITOR_ENABLED` | `true` | Master toggle for all monitoring |
| `MONITOR_LANGFUSE_SECRET_KEY` | `""` | Langfuse API secret |
| `MONITOR_LANGFUSE_PUBLIC_KEY` | `""` | Langfuse API public key |
| `MONITOR_LANGFUSE_HOST` | `http://localhost:3000` | Langfuse instance URL |
| `MONITOR_LANGFUSE_RELEASE` | `dev` | Release/branch identifier |
| `MONITOR_OTEL_SERVICE_NAME` | `rag-pipeline` | OTel service name |
| `MONITOR_OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTel collector endpoint |
| `MONITOR_PRICING_FILE` | `pricing.json` | Path to pricing data |
| `MONITOR_BASELINE_DIR` | `data/monitoring` | CI baseline storage |

## 9. Testing Strategy

| Module | Test approach | Key scenarios |
|--------|--------------|---------------|
| `config.py` | Unit tests with real pricing.json | Cost calculation for known/unknown models |
| `tracing.py` | Unit tests with mocked Langfuse | Span creation, generation capture, error propagation |
| `wrappers.py` | Unit tests with mocked pipeline | Query delegation, latency recording, error recording |
| `metrics.py` | Unit tests with mocked OTel | Histogram/counter recording, noop when disabled |
| `prompts.py` | Unit tests with in-memory registry | Registration, change detection, version history |
| `check_regressions.py` | Unit tests with synthetic baselines | Pass/fail on each metric, threshold enforcement |
| Integration | `evaluate.py` with monitoring enabled | End-to-end: trace + metrics + CI output |

## 10. Implementation Order

```
Task 1: Config + pricing + scaffold         (foundation, no deps)
Task 2: Langfuse Tracer                      (depends on Task 1)
Task 3: MonitoredRAGPipeline wrapper         (depends on Task 1, 2)
Task 4: PromptRegistry                       (depends on Task 1)
Task 5: OTel MetricsCollector                (depends on Task 1)
Task 6: Wire metrics into wrapper            (depends on Task 3, 5)
Task 7: Regression scripts                   (depends on Task 1, 4)
Task 8: CI workflow                          (depends on Task 7)
Task 9: Docker Compose + Grafana             (independent)
```
