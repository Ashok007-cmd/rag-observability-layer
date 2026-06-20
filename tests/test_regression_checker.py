import json
import pytest
from scripts.record_baseline import record_baseline, MetricsResult, compute_metrics_from_latencies
from scripts.check_regressions import check_regressions, Baseline, RegressionResult, load_baseline


SAMPLE_METRICS = MetricsResult(
    p50_latency=0.452,
    p95_latency=1.234,
    avg_cost=0.0025,
    avg_faithfulness=0.85,
    pass_rate=0.8,
    total_queries=10,
    errors=2,
)


def test_record_baseline_creates_json(tmp_path, mocker):
    mocker.patch("scripts.record_baseline.settings.baseline_dir", str(tmp_path))
    mocker.patch("scripts.record_baseline.settings.langfuse_release", "test-branch")
    result = record_baseline(SAMPLE_METRICS, prompt_hash="abc123")
    baseline_path = tmp_path / "baselines.json"
    assert baseline_path.exists()
    with open(baseline_path) as f:
        data = json.load(f)
    assert "test-branch" in data
    assert data["test-branch"]["prompt_hash"] == "abc123"
    assert data["test-branch"]["p50_latency"] == 0.452


def test_record_baseline_preserves_existing_branch(tmp_path, mocker):
    mocker.patch("scripts.record_baseline.settings.baseline_dir", str(tmp_path))
    mocker.patch("scripts.record_baseline.settings.langfuse_release", "branch-a")
    existing = {"branch-b": {"prompt_hash": "old"}}
    baseline_file = tmp_path / "baselines.json"
    with open(baseline_file, "w") as f:
        json.dump(existing, f)
    record_baseline(SAMPLE_METRICS, prompt_hash="new-hash")
    with open(baseline_file) as f:
        data = json.load(f)
    assert data["branch-b"]["prompt_hash"] == "old"
    assert data["branch-a"]["prompt_hash"] == "new-hash"


def test_check_regressions_all_pass():
    baseline = Baseline(p50_latency=0.5, p95_latency=1.3, avg_cost=0.003,
                        avg_faithfulness=0.8, pass_rate=0.75)
    current = MetricsResult(p50_latency=0.45, p95_latency=1.2, avg_cost=0.0025,
                            avg_faithfulness=0.85, pass_rate=0.8, total_queries=10, errors=1)
    results = check_regressions(baseline, current)
    assert all(r.passed for r in results)


def test_check_regressions_fails_on_latency_regression():
    baseline = Baseline(p50_latency=0.5, p95_latency=1.3, avg_cost=0.003,
                        avg_faithfulness=0.8, pass_rate=0.75)
    current = MetricsResult(p50_latency=0.7, p95_latency=1.2, avg_cost=0.0025,
                            avg_faithfulness=0.85, pass_rate=0.8, total_queries=10, errors=1)
    results = check_regressions(baseline, current)
    p50_result = [r for r in results if r.metric == "p50_latency"][0]
    assert not p50_result.passed


def test_check_regressions_fails_on_cost_regression():
    baseline = Baseline(p50_latency=0.5, p95_latency=1.3, avg_cost=0.003,
                        avg_faithfulness=0.8, pass_rate=0.75)
    current = MetricsResult(p50_latency=0.45, p95_latency=1.2, avg_cost=0.004,
                            avg_faithfulness=0.85, pass_rate=0.8, total_queries=10, errors=1)
    results = check_regressions(baseline, current)
    cost_result = [r for r in results if r.metric == "avg_cost"][0]
    assert not cost_result.passed


def test_check_regressions_fails_on_faithfulness_regression():
    baseline = Baseline(p50_latency=0.5, p95_latency=1.3, avg_cost=0.003,
                        avg_faithfulness=0.9, pass_rate=0.85)
    current = MetricsResult(p50_latency=0.45, p95_latency=1.2, avg_cost=0.0025,
                            avg_faithfulness=0.6, pass_rate=0.8, total_queries=10, errors=1)
    results = check_regressions(baseline, current)
    faith_result = [r for r in results if r.metric == "avg_faithfulness"][0]
    assert not faith_result.passed


# --- CLI execution unit tests ---

import sys

def test_record_baseline_cli(tmp_path, mocker):
    from scripts.record_baseline import main
    
    mocker.patch("scripts.record_baseline.settings.baseline_dir", str(tmp_path))
    
    eval_summary = tmp_path / "eval-summary.json"
    with open(eval_summary, "w") as f:
        json.dump({"avg_faithfulness": 0.8, "pass_rate": 0.9, "threshold_met": True}, f)
        
    mon_summary = tmp_path / "monitoring-summary.json"
    with open(mon_summary, "w") as f:
        json.dump({"p50_latency": 0.1, "p95_latency": 0.2, "avg_cost": 0.001, "total_queries": 10, "errors": 0}, f)
        
    mocker.patch.object(sys, "argv", [
        "record_baseline.py",
        "--branch", "cli-test-branch",
        "--eval-summary", str(eval_summary),
        "--monitoring-summary", str(mon_summary)
    ])
    
    main()
    
    baseline_file = tmp_path / "baselines.json"
    assert baseline_file.exists()
    with open(baseline_file) as f:
        data = json.load(f)
    assert "cli-test-branch" in data
    assert data["cli-test-branch"]["p50_latency"] == 0.1


def test_check_regressions_cli(tmp_path, mocker):
    from scripts.check_regressions import main
    
    mocker.patch("scripts.check_regressions.settings.baseline_dir", str(tmp_path))
    
    baseline_file = tmp_path / "baselines.json"
    with open(baseline_file, "w") as f:
        json.dump({
            "cli-test-branch": {
                "p50_latency": 0.1, "p95_latency": 0.2, "avg_cost": 0.001,
                "avg_faithfulness": 0.8, "pass_rate": 0.9, "total_queries": 10, "errors": 0
            }
        }, f)
        
    eval_summary = tmp_path / "eval-summary.json"
    with open(eval_summary, "w") as f:
        json.dump({"avg_faithfulness": 0.8, "pass_rate": 0.9, "threshold_met": True}, f)
        
    mon_summary = tmp_path / "monitoring-summary.json"
    with open(mon_summary, "w") as f:
        json.dump({"p50_latency": 0.1, "p95_latency": 0.2, "avg_cost": 0.001, "total_queries": 10, "errors": 0}, f)
        
    mocker.patch.object(sys, "argv", [
        "check_regressions.py",
        "--branch", "cli-test-branch",
        "--eval-summary", str(eval_summary),
        "--monitoring-summary", str(mon_summary)
    ])
    
    main()


def test_check_regressions_cli_fails(tmp_path, mocker):
    from scripts.check_regressions import main
    
    mocker.patch("scripts.check_regressions.settings.baseline_dir", str(tmp_path))
    
    baseline_file = tmp_path / "baselines.json"
    with open(baseline_file, "w") as f:
        json.dump({
            "cli-test-branch": {
                "p50_latency": 0.1, "p95_latency": 0.2, "avg_cost": 0.001,
                "avg_faithfulness": 0.8, "pass_rate": 0.9, "total_queries": 10, "errors": 0
            }
        }, f)
        
    eval_summary = tmp_path / "eval-summary.json"
    with open(eval_summary, "w") as f:
        json.dump({"avg_faithfulness": 0.8, "pass_rate": 0.9, "threshold_met": True}, f)
        
    mon_summary = tmp_path / "monitoring-summary.json"
    with open(mon_summary, "w") as f:
        json.dump({"p50_latency": 0.3, "p95_latency": 0.2, "avg_cost": 0.001, "total_queries": 10, "errors": 0}, f)
        
    mocker.patch.object(sys, "argv", [
        "check_regressions.py",
        "--branch", "cli-test-branch",
        "--eval-summary", str(eval_summary),
        "--monitoring-summary", str(mon_summary)
    ])
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


# --- compute_metrics_from_latencies edge cases ---

def test_compute_metrics_empty():
    p50, p95 = compute_metrics_from_latencies([])
    assert p50 == 0.0
    assert p95 == 0.0


def test_compute_metrics_single_element():
    p50, p95 = compute_metrics_from_latencies([0.5])
    assert p50 == 0.5
    assert p95 == 0.5


def test_compute_metrics_two_elements():
    # floor(2 * 0.50) = 1 → upper value; floor(2 * 0.95) = 1 → upper value
    p50, p95 = compute_metrics_from_latencies([0.2, 0.8])
    assert p50 == 0.8
    assert p95 == 0.8


def test_compute_metrics_typical():
    lats = [0.1 * i for i in range(1, 11)]
    p50, p95 = compute_metrics_from_latencies(lats)
    assert p50 <= p95
    assert p50 > 0.0


# --- load_baseline edge cases ---

def test_load_baseline_missing_file(tmp_path, mocker):
    mocker.patch("scripts.check_regressions.settings.baseline_dir", str(tmp_path))
    result = load_baseline("some-branch")
    assert result is None


def test_load_baseline_falls_back_to_main(tmp_path, mocker):
    mocker.patch("scripts.check_regressions.settings.baseline_dir", str(tmp_path))
    baseline_file = tmp_path / "baselines.json"
    with open(baseline_file, "w") as f:
        json.dump({
            "main": {
                "p50_latency": 0.3, "p95_latency": 0.9, "avg_cost": 0.001,
                "avg_faithfulness": 0.8, "pass_rate": 0.9, "prompt_hash": "abc"
            }
        }, f)
    result = load_baseline("feature-branch")
    assert result is not None
    assert result.p50_latency == 0.3


def test_check_regressions_both_zero():
    baseline = Baseline(p50_latency=0.0, p95_latency=0.0, avg_cost=0.0,
                        avg_faithfulness=0.0, pass_rate=0.0)
    current = MetricsResult(p50_latency=0.0, p95_latency=0.0, avg_cost=0.0,
                            avg_faithfulness=0.0, pass_rate=0.0, total_queries=0, errors=0)
    results = check_regressions(baseline, current)
    assert all(r.passed for r in results)
    assert all(r.delta_pct == 0.0 for r in results)

