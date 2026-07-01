import pytest
import time
from unittest.mock import MagicMock
from monitoring.extensions import (
    CircuitBreaker,
    CostBudgetExtension,
    TelemetryQueueWorker,
    BaseExtension,
    GuardrailExtension,
    LangfuseTracingExtension,
    OTelMetricsExtension,
)
from monitoring.wrappers import MonitoredRAGPipeline


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
    assert cb.allow_request() is True
    
    # First failure
    cb.record_failure()
    assert cb.state == "CLOSED"
    assert cb.allow_request() is True
    
    # Second failure triggers breaker
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.allow_request() is False
    
    # Wait for cooldown
    time.sleep(0.15)
    assert cb.allow_request() is True
    assert cb.state == "HALF-OPEN"
    
    # Success resets breaker
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.allow_request() is True


def test_telemetry_queue_worker():
    worker = TelemetryQueueWorker(max_workers=1)
    called = []
    
    def task(val):
        called.append(val)
        
    worker.submit(task, 42)
    time.sleep(0.05)  # wait for background thread to run
    assert called == [42]
    worker.shutdown()


def test_custom_extension_execution(mocker):
    pipeline = mocker.MagicMock()
    pipeline.query.return_value = ("answer", [])
    pipeline.config = mocker.MagicMock()
    pipeline.config.top_k_final = 2
    
    mock_ext = mocker.MagicMock(spec=BaseExtension)
    monitored = MonitoredRAGPipeline(pipeline, extensions=[mock_ext])
    
    monitored.query("test query")
    
    mock_ext.on_query_start.assert_called_once()
    mock_ext.on_query_end.assert_called_once()


def test_streaming_query_monitoring(mocker):
    class DummyGenerator:
        def generate_stream(self, q, ctxs, sys_prompt=None):
            yield "streamed"
            yield "answer"

    pipeline = mocker.MagicMock()
    pipeline.config = mocker.MagicMock()
    pipeline.config.top_k_final = 2
    pipeline.generator = DummyGenerator()
    pipeline._retrieve.return_value = [{"id": "c1", "document": "context"}]
    pipeline.citation_formatter.build_citations.return_value = ["cit1"]

    mock_ext = mocker.MagicMock(spec=BaseExtension)
    monitored = MonitoredRAGPipeline(pipeline, extensions=[mock_ext])

    stream = monitored.query_stream("test query")
    chunks = list(stream)

    assert chunks == ["streamed", "answer"]
    assert mock_ext.on_query_start.call_count == 1
    assert mock_ext.on_query_end.call_count == 1
    assert mock_ext.on_step_start.call_count == 1  # generate step
    assert mock_ext.on_step_end.call_count == 1


# --- GuardrailExtension ---

def test_guardrail_blocks_exact_keyword_case_insensitive():
    ext = GuardrailExtension(blocked_keywords=["Internal_Secret"])
    with pytest.raises(ValueError, match="blocked keyword"):
        ext.on_query_start("what is the internal_secret value?", {})


def test_guardrail_does_not_block_substring_match():
    # "secret" alone must not trigger a "internal_secret" block (whole-word match)
    ext = GuardrailExtension(blocked_keywords=["internal_secret"])
    ext.on_query_start("tell me a secret", {})  # should not raise


def test_guardrail_allows_clean_query_and_response():
    ext = GuardrailExtension(blocked_keywords=["internal_secret"])
    ext.on_query_start("how does RAG work?", {})
    ext.on_query_end("a clean answer", [], 0.1, {}, 0.001)  # should not raise


def test_guardrail_blocks_output_containing_keyword():
    ext = GuardrailExtension(blocked_keywords=["confidential"])
    with pytest.raises(ValueError, match="blocked by real-time output policy"):
        ext.on_query_end("this is CONFIDENTIAL data", [], 0.1, {}, 0.001)


def test_guardrail_blocks_over_length_response():
    ext = GuardrailExtension(max_length=10)
    with pytest.raises(ValueError, match="exceeds guardrail"):
        ext.on_query_end("a" * 20, [], 0.1, {}, 0.001)


# --- CostBudgetExtension ---

def test_cost_budget_no_alert_when_under_limits():
    alert = MagicMock()
    ext = CostBudgetExtension(per_query_limit=1.0, daily_limit=10.0, on_budget_exceeded=alert)
    ext.on_query_end("answer", [], 0.1, {}, 0.05)
    alert.assert_not_called()


def test_cost_budget_alerts_on_per_query_limit():
    alert = MagicMock()
    ext = CostBudgetExtension(per_query_limit=0.01, on_budget_exceeded=alert)
    ext.on_query_end("answer", [], 0.1, {}, 0.05)
    alert.assert_called_once_with("per_query", 0.05, 0.01)


def test_cost_budget_alerts_on_daily_limit_after_accumulation():
    alert = MagicMock()
    ext = CostBudgetExtension(daily_limit=0.08, on_budget_exceeded=alert)
    ext.on_query_end("answer", [], 0.1, {}, 0.05)
    alert.assert_not_called()
    ext.on_query_end("answer", [], 0.1, {}, 0.05)
    alert.assert_called_once_with("daily", pytest.approx(0.10), 0.08)


def test_cost_budget_default_alert_logs_warning(caplog):
    ext = CostBudgetExtension(per_query_limit=0.01)
    with caplog.at_level("WARNING"):
        ext.on_query_end("answer", [], 0.1, {}, 0.05)
    assert "Cost budget exceeded" in caplog.text


def test_cost_budget_no_limits_never_alerts():
    alert = MagicMock()
    ext = CostBudgetExtension(on_budget_exceeded=alert)
    ext.on_query_end("answer", [], 0.1, {}, 1000.0)
    alert.assert_not_called()
