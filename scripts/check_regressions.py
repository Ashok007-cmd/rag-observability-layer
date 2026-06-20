"""Compare current metrics against baselines and fail on regression."""

from __future__ import annotations

import os
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monitoring.config import settings
from scripts.record_baseline import MetricsResult

logger = logging.getLogger(__name__)

REGRESSION_THRESHOLD = 0.20


@dataclass
class Baseline:
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    avg_cost: float = 0.0
    avg_faithfulness: float = 0.0
    pass_rate: float = 0.0
    prompt_hash: str = ""


@dataclass
class RegressionResult:
    metric: str
    baseline: float
    current: float
    delta_pct: float
    passed: bool


def load_baseline(branch: str) -> Baseline | None:
    base_dir = Path(settings.baseline_dir)
    baseline_path = base_dir / "baselines.json"
    if not baseline_path.exists():
        return None
    with open(baseline_path) as f:
        data: dict[str, Any] = json.load(f)
    entry = data.get(branch)
    if not entry and branch not in ("main", "master"):
        entry = data.get("main") or data.get("master")
    if not entry:
        return None
    return Baseline(
        p50_latency=entry.get("p50_latency", 0.0),
        p95_latency=entry.get("p95_latency", 0.0),
        avg_cost=entry.get("avg_cost", 0.0),
        avg_faithfulness=entry.get("avg_faithfulness", 0.0),
        pass_rate=entry.get("pass_rate", 0.0),
        prompt_hash=entry.get("prompt_hash", ""),
    )


def check_regressions(
    baseline: Baseline,
    current: MetricsResult,
    threshold: float = REGRESSION_THRESHOLD,
) -> list[RegressionResult]:
    checks = [
        ("p50_latency", baseline.p50_latency, current.p50_latency, False),
        ("p95_latency", baseline.p95_latency, current.p95_latency, False),
        ("avg_cost", baseline.avg_cost, current.avg_cost, False),
        ("avg_faithfulness", baseline.avg_faithfulness, current.avg_faithfulness, True),
        ("pass_rate", baseline.pass_rate, current.pass_rate, True),
    ]

    results: list[RegressionResult] = []
    for metric, base_val, curr_val, higher_is_better in checks:
        if base_val == 0.0 and curr_val == 0.0:
            results.append(RegressionResult(metric, base_val, curr_val, 0.0, True))
            continue
        if base_val == 0.0:
            delta = 0.0
            passed = True
        else:
            delta = (curr_val - base_val) / base_val
            if higher_is_better:
                passed = delta >= -threshold
            else:
                passed = delta <= threshold
        results.append(RegressionResult(metric, base_val, curr_val, round(delta, 4), passed))

    return results


def print_report(results: list[RegressionResult]) -> None:
    print("=" * 60)
    print("REGRESSION CHECK REPORT")
    print("=" * 60)
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        arrow = "\u2191" if r.delta_pct > 0 else "\u2193"
        print(
            f"  {status:5s} | {r.metric:20s} | baseline={r.baseline:.4f} | "
            f"current={r.current:.4f} | delta={arrow}{abs(r.delta_pct)*100:.1f}%"
        )
        if not r.passed:
            all_passed = False
    print("=" * 60)
    print(f"VERDICT: {'ALL PASSED' if all_passed else 'REGRESSION DETECTED'}")
    if not all_passed:
        sys.exit(1)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Check regressions against recorded baselines.")
    parser.add_argument("--branch", type=str, help="Branch name to check against.")
    parser.add_argument("--eval-summary", type=str, help="Path to current eval-summary.json")
    parser.add_argument("--monitoring-summary", type=str, help="Path to current monitoring-summary.json")
    args = parser.parse_args()

    branch = args.branch or settings.langfuse_release or "local"

    # Load baseline
    baseline = load_baseline(branch)
    if not baseline:
        logger.warning("No baseline found for branch '%s' or fallback. Skipping regression checks.", branch)
        sys.exit(0)

    # Load current metrics
    def get_project1_dir() -> Path:
        env_dir = os.environ.get("PROJECT1_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p
        parent_dir = Path(__file__).parent.parent.parent
        for name in ["production-grade-rag", "project-1-Production-Grade-RAG"]:
            p = parent_dir / name
            if p.exists():
                return p
        return parent_dir / "production-grade-rag"

    project1_dir = get_project1_dir()
    eval_path = Path(args.eval_summary) if args.eval_summary else (Path("eval-summary.json") if Path("eval-summary.json").exists() else project1_dir / "eval-summary.json")
    mon_path = Path(args.monitoring_summary) if args.monitoring_summary else (Path("monitoring-summary.json") if Path("monitoring-summary.json").exists() else project1_dir / "monitoring-summary.json")


    if not eval_path.exists() or not mon_path.exists():
        logger.error("Current evaluation summary files missing. Cannot check regressions.")
        sys.exit(1)

    with open(eval_path) as f:
        eval_data = json.load(f)
    avg_faithfulness = eval_data.get("avg_faithfulness", 0.0)
    pass_rate = eval_data.get("pass_rate", 0.0)

    with open(mon_path) as f:
        mon_data = json.load(f)
    p50_latency = mon_data.get("p50_latency", 0.0)
    p95_latency = mon_data.get("p95_latency", 0.0)
    avg_cost = mon_data.get("avg_cost", 0.0)
    total_queries = mon_data.get("total_queries", 0)
    errors = mon_data.get("errors", 0)

    current = MetricsResult(
        p50_latency=p50_latency,
        p95_latency=p95_latency,
        avg_cost=avg_cost,
        avg_faithfulness=avg_faithfulness,
        pass_rate=pass_rate,
        total_queries=total_queries,
        errors=errors
    )

    # Check for prompt template changes
    current_prompt_hash = ""
    prompts_registry_path = Path(settings.baseline_dir) / "prompts.json"
    if prompts_registry_path.exists():
        with open(prompts_registry_path) as f:
            prompts_data = json.load(f)
        versions = prompts_data.get("default_system_prompt", [])
        if not versions:
            for k, v in prompts_data.items():
                if v:
                    versions = v
                    break
        if versions:
            current_prompt_hash = versions[-1].get("hash", "")

    if baseline.prompt_hash and current_prompt_hash and baseline.prompt_hash != current_prompt_hash:
        print("\n" + "!" * 80)
        print("WARNING: prompt template changed — verify behavior")
        print(f"  Baseline hash: {baseline.prompt_hash}")
        print(f"  Current hash:  {current_prompt_hash}")
        print("!" * 80 + "\n")

    # Run checks
    results = check_regressions(baseline, current)
    print_report(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
