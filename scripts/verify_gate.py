#!/usr/bin/env python3
"""Verification script to test RAG monitoring, baseline recording, and regression checks.

Mocks the OpenAI API, runs the RAG evaluation, generates summaries,
records a baseline, and validates regression detection.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Setup paths
project3_dir = Path(__file__).parent.parent

def get_project1_dir() -> Path:
    env_dir = os.environ.get("PROJECT1_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p
    parent_dir = project3_dir.parent
    for name in ["production-grade-rag", "project-1-Production-Grade-RAG"]:
        p = parent_dir / name
        if p.exists():
            return p
    return parent_dir / "production-grade-rag"

project1_dir = get_project1_dir()

sys.path.insert(0, str(project3_dir))
sys.path.insert(0, str(project1_dir))
sys.path.insert(0, str(project3_dir / "scripts"))
sys.path.insert(0, str(project1_dir / "scripts"))

# Redirect src.monitoring imports to our premium package
import monitoring
sys.modules["src.monitoring"] = sys.modules["monitoring"]
import monitoring.metrics
sys.modules["src.monitoring.metrics"] = sys.modules["monitoring.metrics"]
import monitoring.tracing
sys.modules["src.monitoring.tracing"] = sys.modules["monitoring.tracing"]
import monitoring.wrappers
sys.modules["src.monitoring.wrappers"] = sys.modules["monitoring.wrappers"]
import monitoring.prompts
sys.modules["src.monitoring.prompts"] = sys.modules["monitoring.prompts"]
import monitoring.config
sys.modules["src.monitoring.config"] = sys.modules["monitoring.config"]

# Enable monitoring
os.environ["MONITOR_ENABLED"] = "true"
os.environ["OPENAI_API_KEY"] = "dummy-verification-key"
os.environ["RAG_DATA_DIR"] = str(project1_dir / "data")
os.environ["RAG_CHROMA_PATH"] = str(project1_dir / "data" / "chroma_db")
os.environ["RAG_GOLDEN_DATASET_PATH"] = str(project1_dir / "data" / "golden_dataset" / "dataset.jsonl")
os.environ["RAG_RERANKER_MODEL"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"



logger = logging.getLogger("verify_gate")


class MockCompletion:
    def __init__(self, content: str, prompt_tokens: int = 120, completion_tokens: int = 40):
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content))
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )


def mock_create(*args, **kwargs):
    messages = kwargs.get("messages", [])
    prompt_text = ""
    for msg in messages:
        prompt_text += msg.get("content", "")

    # Check if this is the Faithfulness judge prompt
    if "AI answer faithfulness" in prompt_text or "faithful" in prompt_text:
        # Return judge JSON
        judge_response = json.dumps({
            "faithful": True,
            "faithfulness_score": 0.90,
            "unsupported_claims": [],
            "explanation": "Verified matching claim in context."
        })
        return MockCompletion(judge_response, prompt_tokens=200, completion_tokens=80)
    else:
        # Return standard generator response
        return MockCompletion("This is a verified mocked answer referencing [1].", prompt_tokens=100, completion_tokens=30)


@patch("openai.resources.chat.completions.Completions.create", side_effect=mock_create)
def test_full_pipeline(mock_openai_create):
    print(">>> Phase 1: Cleaning up old results and baselines...")
    eval_summary = Path("eval-summary.json")
    mon_summary = Path("monitoring-summary.json")
    baseline_dir = project3_dir / "data" / "monitoring"
    
    if eval_summary.exists():
        eval_summary.unlink()
    if mon_summary.exists():
        mon_summary.unlink()
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)

    print(">>> Phase 2: Running evaluate.py with monitoring enabled...")
    import evaluate
    
    # Run evaluation
    sys.argv = ["evaluate.py", "--hybrid", "--reranker", "--export-ci-summary"]
    evaluate.main()

    # Trigger atexit handers manually since process is not exiting yet
    import atexit as atexit_mod
    atexit_mod._run_exitfuncs()

    # Assert evaluation and monitoring summaries are generated
    assert eval_summary.exists(), "eval-summary.json was not created!"
    assert mon_summary.exists(), "monitoring-summary.json was not created!"
    
    with open(eval_summary) as f:
        eval_data = json.load(f)
    print(f"Evaluation summary content: {eval_data}")
    assert eval_data["threshold_met"] is True
    assert eval_data["avg_faithfulness"] == 0.90

    with open(mon_summary) as f:
        mon_data = json.load(f)
    print(f"Monitoring summary content: {mon_data}")
    assert mon_data["total_queries"] == 14
    assert mon_data["errors"] == 0
    assert mon_data["p50_latency"] > 0.0

    prompts_path = baseline_dir / "prompts.json"
    assert prompts_path.exists(), "prompts.json was not created!"
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    print("Prompts registry generated successfully.")

    print("\n>>> Phase 3: Recording baseline for branch 'main'...")
    import record_baseline
    sys.argv = [
        "record_baseline.py",
        "--branch", "main",
        "--eval-summary", "eval-summary.json",
        "--monitoring-summary", "monitoring-summary.json"
    ]
    record_baseline.main()

    baselines_path = baseline_dir / "baselines.json"
    assert baselines_path.exists(), "baselines.json was not created!"
    with open(baselines_path) as f:
        baselines_data = json.load(f)
    print(f"Recorded baselines.json: {baselines_data}")
    assert "main" in baselines_data

    print("\n>>> Phase 4: Checking regressions for branch 'main' (expect PASS)...")
    import check_regressions
    # Set check_regressions arguments
    sys.argv = [
        "check_regressions.py",
        "--branch", "main",
        "--eval-summary", "eval-summary.json",
        "--monitoring-summary", "monitoring-summary.json"
    ]
    try:
        check_regressions.main()
        print("Regression check passed successfully!")
    except SystemExit as e:
        assert e.code == 0, f"check_regressions failed with exit code: {e.code}"

    print("\n>>> Phase 5: Simulating latency degradation (expect FAIL)...")
    # Modify monitoring summary to double latency
    mon_data["p50_latency"] = mon_data["p50_latency"] * 2
    mon_data["p95_latency"] = mon_data["p95_latency"] * 2
    with open(mon_summary, "w") as f:
        json.dump(mon_data, f, indent=2)

    sys.argv = [
        "check_regressions.py",
        "--branch", "main",
        "--eval-summary", "eval-summary.json",
        "--monitoring-summary", "monitoring-summary.json"
    ]
    try:
        check_regressions.main()
        raise AssertionError("Regression checker did not fail on latency degradation!")
    except SystemExit as e:
        assert e.code == 1, f"Expected exit code 1 on regression failure, got {e.code}"
        print("Successfully detected latency regression and failed CI build!")

    print("\n>>> Phase 6: Simulating prompt template change warning...")
    # Change current baseline prompt hash in prompts.json to trigger warning
    with open(prompts_path) as f:
        p_data = json.load(f)
    # Append to version list to simulate new hash
    p_data["default_system_prompt"].append({
        "hash": "differenthash123",
        "prompt": "Different prompt",
        "name": "default_system_prompt"
    })
    with open(prompts_path, "w") as f:
        json.dump(p_data, f, indent=2)

    # Restore correct latency to only trigger prompt warning
    with open(baselines_path) as f:
        base_data = json.load(f)
    mon_data["p50_latency"] = base_data["main"]["p50_latency"]
    mon_data["p95_latency"] = base_data["main"]["p95_latency"]
    with open(mon_summary, "w") as f:
        json.dump(mon_data, f, indent=2)

    sys.argv = [
        "check_regressions.py",
        "--branch", "main",
        "--eval-summary", "eval-summary.json",
        "--monitoring-summary", "monitoring-summary.json"
    ]
    try:
        check_regressions.main()
        print("Prompt template change warning verified successfully!")
    except SystemExit as e:
        assert e.code == 0, f"check_regressions failed with code {e.code} on prompt change warning"

    print("\n✅ ALL VERIFICATION PHASES PASSED!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    test_full_pipeline()
