#!/usr/bin/env python3
"""Demo script showing the pluggable, real-time, streaming monitoring capabilities."""

import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

# Ensure parent directory is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.extensions import BaseExtension, GuardrailExtension, OTelMetricsExtension
from monitoring.wrappers import MonitoredRAGPipeline


class DummyLLMGenerator:
    """Mock generator simulating LLM responses for OpenAI/Anthropic."""
    def __init__(self):
        self.provider = "openai"
        self.model = "gpt-4o-mini"
        self.temperature = 0.7
        self.max_tokens = 100

    def _format_context(self, contexts):
        return "\n".join([c.get("document", "") for c in contexts])

    def generate(self, query: str, contexts: list[dict], system_prompt: str = None) -> str:
        time.sleep(0.5)  # Simulate network latency
        return f"This is a simulated answer for your query: '{query}'. Based on {len(contexts)} chunks."

    def generate_stream(self, query: str, contexts: list[dict], system_prompt: str = None) -> Generator[str, None, None]:
        time.sleep(0.3)  # Time to first token (TTFT) simulation latency
        words = ["Simulated", "streaming", "response", "word", "by", "word", "to", "demonstrate", "TTFT", "and", "throughput."]
        for w in words:
            time.sleep(0.1)  # Simulate chunk delivery interval
            yield w + " "


class DummyRAGPipeline:
    """Mock RAG pipeline simulating retrieval and indexing."""
    def __init__(self):
        self.config = type("Config", (), {"top_k_final": 2})()
        self.generator = DummyLLMGenerator()
        self.citation_formatter = type("Formatter", (), {
            "build_citations": lambda self_obj, contexts: [f"Citation {i+1}" for i in range(len(contexts))]
        })()

    def _retrieve(self, query: str, use_hybrid: bool = False, use_reranker: bool = False, k: int = 5) -> list[dict]:
        time.sleep(0.1)  # Simulate vector DB query latency
        return [
            {"id": "c1", "document": "Observability helps diagnose distributed systems.", "score": 0.9, "metadata": {"source": "docs.txt"}},
            {"id": "c2", "document": "Real-time products require non-blocking logs.", "score": 0.8, "metadata": {"source": "spec.pdf"}},
        ]

    def _apply_reranker(self, query: str, contexts: list[dict], top_k: int = 5) -> list[dict]:
        time.sleep(0.05)  # Simulate rerank latency
        return contexts

    def query(self, question: str, top_k: int = None, use_hybrid: bool = False, use_reranker: bool = False) -> tuple[str, list]:
        contexts = self._retrieve(question, use_hybrid, use_reranker, k=top_k or 2)
        if use_reranker:
            contexts = self._apply_reranker(question, contexts, top_k=top_k or 2)
        answer = self.generator.generate(question, contexts)
        citations = self.citation_formatter.build_citations(contexts)
        return answer, citations


class ConsoleLoggerExtension(BaseExtension):
    """Custom developer plugin printing real-time event logs to the terminal."""

    def on_query_start(self, question: str, metadata: dict[str, Any]) -> None:
        print(f"\n[ConsoleLog] Query Start: '{question}' | Options: {metadata}")

    def on_query_end(self, answer: str, citations: list[Any], elapsed: float, tokens: dict[str, int], cost: float) -> None:
        print(f"[ConsoleLog] Query Finished in {elapsed:.3f}s | Tokens: {tokens} | Cost: ${cost:.6f}")
        print(f"[ConsoleLog] Answer: '{answer}'\n")

    def on_step_start(self, step_name: str, input_data: dict[str, Any], metadata: dict[str, Any]) -> None:
        print(f"  -> Step '{step_name}' started...")

    def on_step_end(self, step_name: str, output_data: dict[str, Any], elapsed: float, metadata: dict[str, Any] | None = None) -> None:
        print(f"  <- Step '{step_name}' finished in {elapsed:.3f}s")


def main():
    print("=" * 70)
    print("REAL-TIME OBSERVABILITY PLATFORM DEMO")
    print("=" * 70)

    # 1. Instantiate the mock pipeline
    base_pipeline = DummyRAGPipeline()

    # 2. Configure pluggable extensions:
    # We include our ConsoleLoggerExtension, the standard OTel metrics exporter,
    # and a GuardrailExtension blocking restricted keywords.
    custom_extensions = [
        ConsoleLoggerExtension(),
        OTelMetricsExtension(),
        GuardrailExtension(blocked_keywords=["secret_password", "confidential_internal"])
    ]

    monitored_pipeline = MonitoredRAGPipeline(
        pipeline=base_pipeline,
        extensions=custom_extensions
    )

    # --- SCENARIO 1: Standard Synchronous Blocking Query ---
    print("\n--- Running Scenario 1: Standard Blocking Query ---")
    monitored_pipeline.query("How do I monitor my production RAG pipeline?", use_reranker=True)

    # --- SCENARIO 2: Real-time Streaming Query ---
    print("\n--- Running Scenario 2: Real-time Streaming Query ---")
    stream = monitored_pipeline.query_stream("Explain stream logging benefits.", use_hybrid=True)

    print("Streaming tokens: ", end="", flush=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()  # final newline

    # --- SCENARIO 3: Real-time Guardrail Violation (Blocking) ---
    print("\n--- Running Scenario 3: Guardrail Enforcement ---")
    print("Sending query containing restricted phrase: 'secret_password'")
    try:
        monitored_pipeline.query("What is the secret_password value?")
    except ValueError as e:
        print(f"\n[GUARDRAIL TRIGGERED SUCCESSFULLY]: {e}")

    print("\n" + "=" * 70)
    print("Demo completed successfully. All components operational.")
    print("=" * 70)


if __name__ == "__main__":
    main()
