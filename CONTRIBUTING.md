# Contributing to the RAG Monitoring & Observability Platform

Thank you for your interest in contributing to the project! We aim to make this platform the most robust, real-time, and developer-friendly observability tool for LLM and RAG pipelines.

This document guides you through setting up your development environment, understanding our architecture, and contributing new features or bug fixes.

---

## 1. Local Development Setup

We use **`uv`** as our Python package manager. It is significantly faster than standard `pip`.

### Prerequisites
* Python 3.10 or newer
* Docker & Docker Compose
* `uv` installed (`pip install uv` or standard install)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/rag-observability-layer.git
   cd rag-observability-layer
   ```

2. Create a virtual environment and install requirements:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   uv pip install openai anthropic  # For testing purposes
   ```

3. Spin up the local observability infrastructure (Postgres, Langfuse, Prometheus, Grafana):
   ```bash
   ./scripts/start_monitoring_stack.sh
   ```
   * **Langfuse UI**: http://localhost:3000
   * **Grafana Dashboard**: http://localhost:3001 (Credentials: `admin`/`admin`)
   * **Prometheus UI**: http://localhost:9090

4. Run the test suite to ensure everything is set up correctly:
   ```bash
   pytest
   ```

---

## 2. Pluggable Extension Architecture

To make this platform extensible, we decouple observability collectors (like Langfuse or OpenTelemetry) from the core RAG execution path. We run them as **Extensions** using a middleware pattern.

### Base Extension Hook Registry
All integrations implement `BaseExtension` found in [extensions.py](file:///home/ak/Dev/rag-observability-layer/monitoring/extensions.py). You can override any of these hooks:

```python
class BaseExtension:
    def on_query_start(self, question: str, metadata: dict[str, Any]) -> None:
        """Fires before query execution starts."""
        pass

    def on_query_end(self, answer: str, citations: list[Any], elapsed: float, tokens: dict[str, int], cost: float) -> None:
        """Fires after query execution completes successfully."""
        pass

    def on_query_error(self, exc: Exception) -> None:
        """Fires if the query raises an exception."""
        pass

    def on_step_start(self, step_name: str, input_data: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Fires before sub-step execution (retrieve, rerank, generate, etc.)."""
        pass

    def on_step_end(self, step_name: str, output_data: dict[str, Any], elapsed: float, metadata: dict[str, Any] | None = None) -> None:
        """Fires when sub-step completes successfully."""
        pass

    def on_step_error(self, step_name: str, exc: Exception) -> None:
        """Fires when a sub-step raises an exception."""
        pass

    def on_generation_llm_call(self, model: str, provider: str, prompt: str, query: str, response: str, usage: dict[str, int], cost: float) -> None:
        """Fires when LLM generations conclude."""
        pass
```

### Writing a Custom Extension
Here is how you can write and register a custom extension, such as a slack alerter on errors:

```python
from monitoring.extensions import BaseExtension

class SlackErrorAlertExtension(BaseExtension):
    def on_query_error(self, exc: Exception) -> None:
        self.send_slack_alert(f"RAG Pipeline Error: {exc}")

    def send_slack_alert(self, message: str) -> None:
        # Custom alert logic
        print(f"[SLACK ALERT] {message}")
```

Register your custom extension when instantiating your monitored pipeline wrapper:

```python
from monitoring.wrappers import MonitoredRAGPipeline

monitored = MonitoredRAGPipeline(
    pipeline=base_rag_pipeline,
    extensions=[
        LangfuseTracingExtension(),
        OTelMetricsExtension(),
        SlackErrorAlertExtension()
    ]
)
```

---

## 3. Real-Time Production Execution Design

For production-grade real-time systems, we employ two core architectural behaviors:

### 1. Asynchronous Telemetry Exports
Telemetry collection (e.g. posting to HTTP collectors) runs in a background thread-pool queue. This means tracing or metrics failures never slow down the primary request execution path.

### 2. Failure Isolation (Circuit Breakers)
If downstream observability systems (like Langfuse or OpenTelemetry HTTP collector) experience an outage, our `CircuitBreaker` trips to `OPEN`. Observability is bypassed for a cooldown period (default: 30 seconds), preventing thread pool exhaustions or request delays.

### 3. Streaming Response Hooks
We support tracing and metrics for streaming queries (SSE / generators) via the `query_stream` method. This hooks into the token stream in real-time, capturing:
* **Time-to-First-Token (TTFT)**
* **Token Throughput (Tokens/second)**
* **Incremental trace capturing**

---

## 4. Contributing Checklist
1. **Tests**: Add unit or integration tests for your additions under the `tests/` directory.
2. **Backwards Compatibility**: Ensure existing test cases pass (`pytest`). Do not break public method signatures of `MonitoredRAGPipeline`.
3. **Resilience**: Wrap network operations in try/except blocks or utilize the telemetry executor to run tasks asynchronously.
4. **Documentation**: Update the README or inline docstrings if introducing new metrics or extensions.
