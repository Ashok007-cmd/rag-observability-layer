from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

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
sys.path.insert(0, str(project1_dir))

from monitoring.config import PricingConfig

import pytest

try:
    from src.pipeline import RAGPipeline as _RAGPipeline
except ImportError:
    _RAGPipeline = None  # type: ignore[assignment,misc]


@pytest.fixture
def pricing():
    return PricingConfig(pricing_path="pricing.json")


@pytest.fixture
def sample_contexts():
    return [
        {
            "id": "c1",
            "document": "RAG stands for Retrieval Augmented Generation.",
            "metadata": {"source": "doc1.pdf"},
            "score": 0.95,
        },
        {
            "id": "c2",
            "document": "It combines retrieval and generation.",
            "metadata": {"source": "doc2.md"},
            "score": 0.87,
        },
    ]


@pytest.fixture
def mock_pipeline(mocker):
    pipeline = mocker.Mock(spec=_RAGPipeline)
    pipeline.query.return_value = ("Test answer.", [])
    return pipeline
