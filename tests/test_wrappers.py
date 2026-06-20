import pytest
from monitoring.wrappers import MonitoredRAGPipeline
from monitoring.tracing import Tracer


@pytest.fixture
def mock_pipeline(mocker):
    pipeline = mocker.MagicMock()
    pipeline.query.return_value = ("Test answer.", [])
    return pipeline


def test_monitored_pipeline_instantiation(mock_pipeline):
    tracer = Tracer(enabled=False)
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer)
    assert monitored._pipeline is mock_pipeline
    assert monitored._tracer is tracer


def test_monitored_pipeline_query_returns_answer_and_citations(mock_pipeline):
    tracer = Tracer(enabled=False)
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer)
    mock_pipeline.query.return_value = ("Generated answer.", ["cit1", "cit2"])
    answer, citations = monitored.query("What is RAG?", use_hybrid=True, use_reranker=True)
    assert answer == "Generated answer."
    assert len(citations) == 2
    mock_pipeline.query.assert_called_once_with("What is RAG?", use_hybrid=True, use_reranker=True, top_k=None)


def test_monitored_pipeline_query_propagates_exception(mock_pipeline):
    tracer = Tracer(enabled=False)
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer)
    mock_pipeline.query.side_effect = RuntimeError("LLM API error")
    with pytest.raises(RuntimeError, match="LLM API error"):
        monitored.query("What is RAG?")


def test_monitored_pipeline_ingest_returns_count(mock_pipeline):
    tracer = Tracer(enabled=False)
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer)
    mock_pipeline.ingest.return_value = 42
    count = monitored.ingest("/path/to/docs")
    assert count == 42
    mock_pipeline.ingest.assert_called_once_with("/path/to/docs")


# --- Metrics integration tests (Task 6) ---

def test_monitored_pipeline_records_latency_on_query(mock_pipeline, mocker):
    tracer = Tracer(enabled=False)
    metrics = mocker.MagicMock()
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer, metrics=metrics)
    mock_pipeline.query.return_value = ("answer", [])
    monitored.query("What is RAG?")
    metrics.record_latency.assert_called_once()
    metrics.record_query_count.assert_called_once()
    metrics.record_context_count.assert_called_once()


def test_monitored_pipeline_records_error_on_query_failure(mock_pipeline, mocker):
    tracer = Tracer(enabled=False)
    metrics = mocker.MagicMock()
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer, metrics=metrics)
    mock_pipeline.query.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError):
        monitored.query("What is RAG?")
    metrics.record_error.assert_called_once_with("query", "RuntimeError")


def test_monitored_pipeline_metrics_noop_when_not_provided(mock_pipeline):
    tracer = Tracer(enabled=False)
    monitored = MonitoredRAGPipeline(pipeline=mock_pipeline, tracer=tracer)
    mock_pipeline.query.return_value = ("answer", [])
    monitored.query("What is RAG?")  # should not raise


# --- Interception & Dynamic Monkeypatching Integration tests ---

from types import SimpleNamespace
from unittest.mock import MagicMock

def test_intercept_token_usage_openai(mocker):
    from monitoring.wrappers import intercept_token_usage
    import openai

    mock_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40, total_tokens=160),
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    mocker.patch("openai.resources.chat.completions.Completions.create", return_value=mock_response)

    usage = {}
    with intercept_token_usage(usage):
        client = openai.OpenAI(api_key="dummy")
        client.chat.completions.create(model="gpt-4", messages=[])

    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 160


def test_intercept_token_usage_anthropic(mocker):
    from monitoring.wrappers import intercept_token_usage
    import anthropic

    mock_response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=150, output_tokens=50),
        content=[SimpleNamespace(text="answer")]
    )
    mocker.patch("anthropic.resources.messages.Messages.create", return_value=mock_response)

    usage = {}
    with intercept_token_usage(usage):
        client = anthropic.Anthropic(api_key="dummy")
        client.messages.create(model="claude", messages=[], max_tokens=100)

    assert usage["prompt_tokens"] == 150
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 200


def test_monitored_pipeline_instrumentation(mocker):
    from monitoring.wrappers import MonitoredRAGPipeline
    
    class DummyPipeline:
        def __init__(self):
            self.generator = MagicMock()
            self.generator.generate = MagicMock(return_value="answer")
            self.generator.provider = "openai"
            self.generator.model = "gpt-4o-mini"
            self.generator._format_context = MagicMock(return_value="flat contexts")
            
        def _retrieve(self, query, use_hybrid=False, use_reranker=False, k=5):
            return [{"id": "c1", "document": "doc content", "metadata": {"source": "source.txt"}, "score": 0.9}]
            
        def _apply_reranker(self, query, contexts, top_k=5):
            return contexts
            
        def query(self, question, top_k=None, use_hybrid=False, use_reranker=False):
            contexts = self._retrieve(question, use_hybrid, use_reranker, k=top_k or 5)
            if use_reranker:
                contexts = self._apply_reranker(question, contexts, top_k=top_k or 5)
            ans = self.generator.generate(question, contexts)
            return ans, []

    pipeline = DummyPipeline()
    tracer = mocker.MagicMock()
    metrics = mocker.MagicMock()
    
    mocker.patch("monitoring.prompts.PromptRegistry.register")
    
    monitored = MonitoredRAGPipeline(pipeline, tracer=tracer, metrics=metrics)
    monitored.query("test query", use_reranker=True)
    
    # Verify the dynamic instrumentation hooked in and traced correctly
    assert tracer.trace_step.call_count >= 3

