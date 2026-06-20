import pytest
import time
from unittest.mock import MagicMock
from monitoring.extensions import CircuitBreaker, TelemetryQueueWorker, BaseExtension, LangfuseTracingExtension, OTelMetricsExtension
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
