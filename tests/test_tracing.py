import pytest
from monitoring.config import PricingConfig
from monitoring.tracing import Tracer


# --- PricingConfig tests (Task 1) ---

def test_pricing_config_returns_cost_for_openai():
    pricing = PricingConfig(pricing_path="pricing.json")
    cost = pricing.get_cost("openai", "gpt-4o-mini", 500, 200)
    assert cost == pytest.approx(0.000075 + 0.00012, rel=1e-6)


def test_pricing_config_returns_zero_for_unknown_model():
    pricing = PricingConfig(pricing_path="pricing.json")
    cost = pricing.get_cost("openai", "nonexistent-model", 100, 100)
    assert cost == 0.0


def test_pricing_config_returns_zero_for_unknown_provider():
    pricing = PricingConfig(pricing_path="pricing.json")
    cost = pricing.get_cost("unknown", "gpt-4o-mini", 100, 100)
    assert cost == 0.0


# --- Tracer tests (Task 2) ---

def test_tracer_instantiation():
    tracer = Tracer(enabled=False)
    assert not tracer.enabled
    assert tracer._langfuse is None


def test_tracer_trace_step_returns_context_manager():
    tracer = Tracer(enabled=False)
    with tracer.trace_step("test_step", input={"query": "hello"}) as span:
        assert span is not None
        assert span["name"] == "test_step"


def test_tracer_capture_generation_records_tokens(mocker):
    mock_langfuse = mocker.patch("langfuse.Langfuse")
    tracer = Tracer(enabled=True)
    tracer._langfuse = mock_langfuse
    mock_span = mocker.MagicMock()
    mock_generation = mocker.MagicMock()
    mock_span.generation = mocker.MagicMock(return_value=mock_generation)

    tracer.capture_generation(
        mock_span,
        model="gpt-4o-mini",
        provider="openai",
        prompt="system prompt",
        query="user query",
        response="model response",
        prompt_tokens=500,
        completion_tokens=200,
        total_tokens=700,
    )

    mock_span.generation.assert_called_once()
    _, kwargs = mock_span.generation.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["usage"]["prompt"] == 500
    assert kwargs["usage"]["completion"] == 200


def test_tracer_get_trace_id_returns_string_when_disabled():
    tracer = Tracer(enabled=False)
    trace_id = tracer.get_trace_id()
    assert isinstance(trace_id, str)
    assert len(trace_id) > 0


def test_tracer_get_trace_id_stable_within_request():
    tracer = Tracer(enabled=False)
    first = tracer.get_trace_id()
    second = tracer.get_trace_id()
    assert first == second


def test_tracer_trace_step_propagates_exceptions():
    tracer = Tracer(enabled=False)
    with pytest.raises(ValueError, match="step error"):
        with tracer.trace_step("failing_step"):
            raise ValueError("step error")


def test_tracer_trace_id_resets_after_root_span():
    tracer = Tracer(enabled=False)
    with tracer.trace_step("root"):
        first_id = tracer.get_trace_id()
    assert tracer._trace_id == ""
    with tracer.trace_step("root2"):
        second_id = tracer.get_trace_id()
    assert first_id != second_id


def test_pricing_config_anthropic_new_models():
    pricing = PricingConfig(pricing_path="pricing.json")
    cost = pricing.get_cost("anthropic", "claude-sonnet-4-6", 1000, 500)
    assert cost == pytest.approx((1000 * 0.003 / 1000) + (500 * 0.015 / 1000))
    cost_opus = pricing.get_cost("anthropic", "claude-opus-4-8", 1000, 500)
    assert cost_opus == pytest.approx((1000 * 0.015 / 1000) + (500 * 0.075 / 1000))
