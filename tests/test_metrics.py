import pytest
from monitoring.metrics import MetricsCollector


def test_metrics_collector_instantiation():
    mc = MetricsCollector(enabled=True, prefix="rag")
    assert mc.enabled
    assert mc.prefix == "rag"


def test_metrics_collector_none_when_disabled():
    mc = MetricsCollector(enabled=False, prefix="rag")
    assert mc._meter is None


def test_metrics_collector_record_latency(mocker):
    mock_histogram = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._latency_histograms = {"retrieve": mock_histogram}
    mc.record_latency("retrieve", 0.5)
    mock_histogram.record.assert_called_once_with(0.5)


def test_metrics_collector_record_latency_noop_when_disabled():
    mc = MetricsCollector(enabled=False, prefix="rag")
    mc.record_latency("retrieve", 0.5)


def test_metrics_collector_record_error(mocker):
    mock_counter = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._error_counters = {"test_step": mock_counter}
    mc.record_error("test_step", "ValueError")
    mock_counter.add.assert_called_once_with(1, {"error_type": "ValueError"})


def test_metrics_collector_record_error_with_trace_id(mocker):
    mock_counter = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._error_counters = {"test_step": mock_counter}
    mc.record_error("test_step", "ValueError", trace_id="abc-123")
    mock_counter.add.assert_called_once_with(1, {"error_type": "ValueError", "trace_id": "abc-123"})


def test_metrics_collector_record_query_count(mocker):
    mock_counter = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._query_counter = mock_counter
    mc.record_query_count()
    mock_counter.add.assert_called_once_with(1)


def test_metrics_collector_record_context_count(mocker):
    mock_histogram = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._context_histogram = mock_histogram
    mc.record_context_count(5)
    mock_histogram.record.assert_called_once_with(5)


def test_metrics_collector_record_cost(mocker):
    mock_histogram = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._cost_histogram = mock_histogram
    mc.record_cost(0.0025)
    mock_histogram.record.assert_called_once_with(0.0025)
    assert mc.costs == [0.0025]


def test_metrics_collector_record_tokens(mocker):
    mock_prompt = mocker.MagicMock()
    mock_completion = mocker.MagicMock()
    mock_total = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._token_histograms = {"prompt": mock_prompt, "completion": mock_completion, "total": mock_total}
    mc.record_tokens(100, 50, 150)
    mock_prompt.record.assert_called_once_with(100)
    mock_completion.record.assert_called_once_with(50)
    mock_total.record.assert_called_once_with(150)


def test_metrics_collector_export_summary(tmp_path):
    mc = MetricsCollector(enabled=False, prefix="rag")
    mc.latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    mc.costs = [0.001, 0.002]
    mc.queries_count = 5
    mc.errors_count = 1
    out = tmp_path / "summary.json"
    mc.export_summary(out)
    import json
    data = json.loads(out.read_text())
    assert data["total_queries"] == 5
    assert data["errors"] == 1
    assert data["p50_latency"] > 0
    assert data["p95_latency"] > 0
    assert data["avg_cost"] == pytest.approx(0.0015)


def test_metrics_collector_export_summary_empty(tmp_path):
    mc = MetricsCollector(enabled=False, prefix="rag")
    out = tmp_path / "summary.json"
    mc.export_summary(out)
    import json
    data = json.loads(out.read_text())
    assert data["p50_latency"] == 0.0
    assert data["p95_latency"] == 0.0
    assert data["avg_cost"] == 0.0


def test_record_retrieval_quality_hit_and_precision(mocker):
    mock_precision = mocker.MagicMock()
    mock_mrr = mocker.MagicMock()
    mock_hit = mocker.MagicMock()
    mc = MetricsCollector(enabled=True, prefix="rag")
    mc._precision_histogram = mock_precision
    mc._mrr_histogram = mock_mrr
    mc._hit_rate_histogram = mock_hit

    result = mc.record_retrieval_quality(
        retrieved_ids=["a", "b", "c"], relevant_ids={"b", "z"}
    )

    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["reciprocal_rank"] == pytest.approx(1 / 2)
    assert result["hit"] == 1.0
    mock_precision.record.assert_called_once_with(pytest.approx(1 / 3))
    mock_mrr.record.assert_called_once_with(pytest.approx(1 / 2))
    mock_hit.record.assert_called_once_with(1.0)


def test_record_retrieval_quality_no_hit():
    mc = MetricsCollector(enabled=False, prefix="rag")
    result = mc.record_retrieval_quality(retrieved_ids=["a", "b"], relevant_ids={"z"})
    assert result == {"precision_at_k": 0.0, "reciprocal_rank": 0.0, "hit": 0.0}


def test_record_retrieval_quality_respects_k():
    mc = MetricsCollector(enabled=False, prefix="rag")
    # "b" is relevant but outside the top-1 window
    result = mc.record_retrieval_quality(retrieved_ids=["a", "b"], relevant_ids={"b"}, k=1)
    assert result == {"precision_at_k": 0.0, "reciprocal_rank": 0.0, "hit": 0.0}


def test_record_retrieval_quality_empty_retrieved():
    mc = MetricsCollector(enabled=False, prefix="rag")
    result = mc.record_retrieval_quality(retrieved_ids=[], relevant_ids={"a"})
    assert result == {"precision_at_k": 0.0, "reciprocal_rank": 0.0, "hit": 0.0}


def test_export_summary_includes_retrieval_quality(tmp_path):
    mc = MetricsCollector(enabled=False, prefix="rag")
    mc.record_retrieval_quality(retrieved_ids=["a", "b"], relevant_ids={"a"})
    mc.record_retrieval_quality(retrieved_ids=["x", "y"], relevant_ids={"z"})
    out = tmp_path / "summary.json"
    mc.export_summary(out)
    import json
    data = json.loads(out.read_text())
    assert data["avg_precision_at_k"] == pytest.approx(0.25)
    assert data["mrr"] == pytest.approx(0.5)
    assert data["hit_rate"] == pytest.approx(0.5)


def test_export_summary_omits_retrieval_quality_when_unused(tmp_path):
    mc = MetricsCollector(enabled=False, prefix="rag")
    out = tmp_path / "summary.json"
    mc.export_summary(out)
    import json
    data = json.loads(out.read_text())
    assert "avg_precision_at_k" not in data
    assert "mrr" not in data
    assert "hit_rate" not in data
