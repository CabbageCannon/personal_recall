"""Regression tests for LLM endpoint configuration.

Covers a real bug found while building the recall CLI: `RetrievalConfig.__init__` calls
``llm_config.set_api_key(force_reset=True)``, and that reset used to discard an explicitly
configured ``env_variable_name`` — replacing it with ``OPENAI_API_KEY`` and wiping the key.
Configurations that point an OPENAI-supplier endpoint at another provider (this project's
DeepSeek setup) therefore only worked by accident of ordering: the runner built the LLM
endpoint *before* constructing the RetrievalConfig.
"""

from __future__ import annotations

import pytest

from quivr_core.rag.entities.config import (
    DefaultModelSuppliers,
    LLMEndpointConfig,
    RetrievalConfig,
)


@pytest.fixture()
def deepseek_env(monkeypatch: pytest.MonkeyPatch):
    """A provider env var that is NOT the supplier's default name."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-value")
    yield


def _config() -> LLMEndpointConfig:
    return LLMEndpointConfig(
        supplier=DefaultModelSuppliers.OPENAI,
        model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
        env_variable_name="DEEPSEEK_API_KEY",
        max_context_tokens=20000,
        max_output_tokens=8192,
        temperature=0.0,
    )


def test_custom_env_variable_name_resolves_the_key(deepseek_env) -> None:
    config = _config()

    assert config.env_variable_name == "DEEPSEEK_API_KEY"
    assert config.llm_api_key == "sk-test-value"


def test_retrieval_config_reset_must_not_discard_the_configured_env_var(deepseek_env) -> None:
    config = _config()

    RetrievalConfig(llm_config=config, k=10)

    assert config.env_variable_name == "DEEPSEEK_API_KEY", "reset must not clobber a custom name"
    assert config.llm_api_key == "sk-test-value", "the key must survive RetrievalConfig construction"


def test_default_env_variable_name_is_still_derived_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-value")

    config = LLMEndpointConfig(supplier=DefaultModelSuppliers.OPENAI, model="gpt-4o")

    assert config.env_variable_name == "OPENAI_API_KEY"
    assert config.llm_api_key == "sk-openai-value"


def test_missing_key_is_tolerated_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = _config()

    assert config.llm_api_key is None, "construction must not raise; the endpoint reports it later"
