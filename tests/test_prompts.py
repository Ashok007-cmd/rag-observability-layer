import hashlib
import pytest
from monitoring.prompts import PromptRegistry

SAMPLE_PROMPT = "You are a helpful assistant. Context: {context}"


def test_prompt_registry_stores_prompt():
    registry = PromptRegistry()
    h = registry.register("default", SAMPLE_PROMPT)
    assert h == hashlib.sha256(SAMPLE_PROMPT.encode()).hexdigest()


def test_prompt_registry_detect_change_returns_false_when_unchanged():
    registry = PromptRegistry()
    h1 = registry.register("default", SAMPLE_PROMPT)
    h2 = registry.register("default", SAMPLE_PROMPT)
    assert h1 == h2


def test_prompt_registry_detect_change_returns_true_when_changed():
    registry = PromptRegistry()
    h1 = registry.register("default", SAMPLE_PROMPT)
    h2 = registry.register("default", SAMPLE_PROMPT + " extra")
    assert h1 != h2


def test_prompt_registry_list_versions():
    registry = PromptRegistry()
    registry.register("default", "v1")
    registry.register("default", "v2")
    versions = registry.get_versions("default")
    assert len(versions) == 2
    assert versions[0].prompt == "v1"
    assert versions[1].prompt == "v2"


def test_prompt_registry_current_hash():
    registry = PromptRegistry()
    h = registry.register("default", "current")
    assert registry.current_hash("default") == h


def test_prompt_registry_current_hash_returns_none_for_unknown():
    registry = PromptRegistry()
    assert registry.current_hash("unknown") is None


def test_prompt_registry_detect_change_external():
    registry = PromptRegistry()
    assert registry.detect_change("default", SAMPLE_PROMPT) is True
    registry.register("default", SAMPLE_PROMPT)
    assert registry.detect_change("default", SAMPLE_PROMPT) is False
    assert registry.detect_change("default", SAMPLE_PROMPT + " changed") is True


def test_prompt_registry_persists_and_reloads(tmp_path):
    path = tmp_path / "prompts.json"
    reg1 = PromptRegistry(persist_path=path)
    reg1.register("default", SAMPLE_PROMPT)

    reg2 = PromptRegistry(persist_path=path)
    assert reg2.current_hash("default") == reg1.current_hash("default")
    assert len(reg2.get_versions("default")) == 1


def test_prompt_registry_load_handles_malformed_json(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("{this is not valid json}")
    reg = PromptRegistry(persist_path=path)
    assert reg.current_hash("default") is None


def test_prompt_registry_load_handles_missing_fields(tmp_path):
    path = tmp_path / "prompts.json"
    import json
    path.write_text(json.dumps({"my_prompt": [{"hash": "abc"}]}))
    reg = PromptRegistry(persist_path=path)
    assert reg.current_hash("my_prompt") is None
