"""Record current evaluation metrics as CI baselines."""

from __future__ import annotations

import os
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitoring.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    p50_latency: float
    p95_latency: float
    avg_cost: float
    avg_faithfulness: float
    pass_rate: float
    total_queries: int
    errors: int


def compute_metrics_from_latencies(latencies: list[float]) -> tuple[float, float]:
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    if n == 0:
        return 0.0, 0.0
    p50 = sorted_lats[min(int(n * 0.50), n - 1)]
    p95 = sorted_lats[min(int(n * 0.95), n - 1)]
    return p50, p95


def record_baseline(
    metrics: MetricsResult,
    prompt_hash: str = "",
    branch: str | None = None,
) -> str:
    base_dir = Path(settings.baseline_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = base_dir / "baselines.json"

    branch = branch or settings.langfuse_release or "local"

    if baseline_path.exists():
        with open(baseline_path) as f:
            data: dict[str, Any] = json.load(f)
    else:
        data = {}

    # Check if baseline already exists for this branch
    if branch in data:
        logger.info("Baseline already exists for branch '%s'. Skipping write.", branch)
        return str(baseline_path)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_hash": prompt_hash,
        **asdict(metrics),
    }

    data[branch] = entry

    import os
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(baseline_path, flags, 0o600)
    with open(fd, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Baseline recorded for branch '%s' at %s", branch, baseline_path)
    return str(baseline_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Record evaluation metrics as baseline.")
    parser.add_argument("--branch", type=str, help="Branch name to record baseline for.")
    parser.add_argument("--eval-summary", type=str, help="Path to eval-summary.json")
    parser.add_argument("--monitoring-summary", type=str, help="Path to monitoring-summary.json")
    args = parser.parse_args()

    branch = args.branch or settings.langfuse_release or "local"

    # Quick check if baseline already exists for the branch to avoid redundant run
    base_dir = Path(settings.baseline_dir)
    baseline_path = base_dir / "baselines.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
        if branch in data:
            logger.info("Baseline already exists for branch '%s'. Skipping.", branch)
            return

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
    
    # Try looking in current working directory first, then fallback to project1_dir
    eval_path = Path(args.eval_summary) if args.eval_summary else (Path("eval-summary.json") if Path("eval-summary.json").exists() else project1_dir / "eval-summary.json")
    mon_path = Path(args.monitoring_summary) if args.monitoring_summary else (Path("monitoring-summary.json") if Path("monitoring-summary.json").exists() else project1_dir / "monitoring-summary.json")

    # If summaries do not exist, run evaluate.py
    if not eval_path.exists() or not mon_path.exists():
        import subprocess
        logger.info("Evaluation result files missing. Running evaluate.py to generate them...")
        env = os.environ.copy()
        env["MONITOR_ENABLED"] = "true"
        # Ensure our project path is on python path for any extensions/scripts
        env["PYTHONPATH"] = os.path.pathsep.join([
            str(Path(__file__).parent.parent),
            env.get("PYTHONPATH", "")
        ]).strip(os.path.pathsep)
        subprocess.run(
            [sys.executable, "scripts/evaluate.py", "--hybrid", "--reranker", "--export-ci-summary"],
            cwd=str(project1_dir),
            env=env,
            check=True
        )
        # Update path references if they were just created in project1_dir
        if not eval_path.exists():
            eval_path = project1_dir / "eval-summary.json"
        if not mon_path.exists():
            mon_path = project1_dir / "monitoring-summary.json"


    # Load eval metrics
    with open(eval_path) as f:
        eval_data = json.load(f)
    avg_faithfulness = eval_data.get("avg_faithfulness", 0.0)
    pass_rate = eval_data.get("pass_rate", 0.0)

    # Load monitoring metrics
    with open(mon_path) as f:
        mon_data = json.load(f)
    p50_latency = mon_data.get("p50_latency", 0.0)
    p95_latency = mon_data.get("p95_latency", 0.0)
    avg_cost = mon_data.get("avg_cost", 0.0)
    total_queries = mon_data.get("total_queries", 0)
    errors = mon_data.get("errors", 0)

    metrics = MetricsResult(
        p50_latency=p50_latency,
        p95_latency=p95_latency,
        avg_cost=avg_cost,
        avg_faithfulness=avg_faithfulness,
        pass_rate=pass_rate,
        total_queries=total_queries,
        errors=errors
    )

    # Load prompt hash
    prompt_hash = ""
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
            prompt_hash = versions[-1].get("hash", "")

    record_baseline(metrics, prompt_hash=prompt_hash, branch=branch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
