import json
from uuid import uuid4

from quivr_core.brain.serialization import (
    BrainSerialized,
    EmbedderConfig,
    FAISSConfig,
    TransparentStorageConfig,
)
from quivr_core.rag.entities.config import LLMEndpointConfig


TEST_ENV_NAME = "TEST_LLM_API_KEY"
RUNTIME_SECRET = "SUPER_SECRET_RUNTIME_KEY"
ENV_SECRET = "REHYDRATED_SECRET_KEY"


def test_llm_api_key_is_excluded_from_direct_serialization(monkeypatch):
    """Runtime API key must never be serialized by LLMEndpointConfig."""

    monkeypatch.delenv(TEST_ENV_NAME, raising=False)

    config = LLMEndpointConfig(
        model="gpt-4o",
        env_variable_name=TEST_ENV_NAME,
        llm_api_key=RUNTIME_SECRET,
    )

    # The secret must still be available at runtime.
    assert config.llm_api_key == RUNTIME_SECRET

    dumped = config.model_dump()
    dumped_json = config.model_dump_json()

    # Secret field/value must not enter serialized data.
    assert "llm_api_key" not in dumped
    assert "llm_api_key" not in dumped_json
    assert RUNTIME_SECRET not in dumped_json

    # Non-secret configuration must still be persisted.
    assert dumped["model"] == "gpt-4o"
    assert dumped["env_variable_name"] == TEST_ENV_NAME


def test_llm_api_key_is_excluded_from_nested_brain_serialization(monkeypatch):
    """BrainSerialized must not leak the nested LLM API key."""

    monkeypatch.delenv(TEST_ENV_NAME, raising=False)

    llm_config = LLMEndpointConfig(
        model="gpt-4o",
        env_variable_name=TEST_ENV_NAME,
        llm_api_key=RUNTIME_SECRET,
    )

    brain = BrainSerialized(
        id=uuid4(),
        name="security-test-brain",
        chat_history=[],
        vectordb_config=FAISSConfig(
            vectordb_folder_path="fake/faiss/path",
        ),
        storage_config=TransparentStorageConfig(
            files={},
        ),
        llm_config=llm_config,
        embedding_config=EmbedderConfig(
            config={},
        ),
    )

    serialized = brain.model_dump_json()
    payload = json.loads(serialized)

    # The secret must never appear anywhere in config.json-style output.
    assert RUNTIME_SECRET not in serialized
    assert "llm_api_key" not in payload["llm_config"]

    # Required non-secret configuration must survive nesting.
    assert payload["llm_config"]["model"] == "gpt-4o"
    assert payload["llm_config"]["env_variable_name"] == TEST_ENV_NAME


def test_llm_api_key_is_rehydrated_from_environment_after_round_trip(monkeypatch):
    """
    Persisted config contains only the environment-variable name.
    Loading it should recover the actual secret from the environment.
    """

    monkeypatch.setenv(TEST_ENV_NAME, ENV_SECRET)

    original = LLMEndpointConfig(
        model="gpt-4o",
        env_variable_name=TEST_ENV_NAME,
        llm_api_key=RUNTIME_SECRET,
    )

    serialized = original.model_dump_json()

    # Runtime secret must not be persisted.
    assert RUNTIME_SECRET not in serialized
    assert ENV_SECRET not in serialized
    assert "llm_api_key" not in serialized

    restored = LLMEndpointConfig.model_validate_json(serialized)

    # The persisted env-variable name tells set_api_key() where to reload it.
    assert restored.env_variable_name == TEST_ENV_NAME
    assert restored.llm_api_key == ENV_SECRET

    # Other LLM configuration must survive the round trip.
    assert restored.model == original.model
    assert restored.max_context_tokens == original.max_context_tokens
    assert restored.max_output_tokens == original.max_output_tokens
    assert restored.temperature == original.temperature
    assert restored.streaming == original.streaming


def test_nested_brain_round_trip_rehydrates_llm_api_key(monkeypatch):
    """The same behavior must hold through BrainSerialized."""

    monkeypatch.setenv(TEST_ENV_NAME, ENV_SECRET)

    brain = BrainSerialized(
        id=uuid4(),
        name="security-round-trip-brain",
        chat_history=[],
        vectordb_config=FAISSConfig(
            vectordb_folder_path="fake/faiss/path",
        ),
        storage_config=TransparentStorageConfig(
            files={},
        ),
        llm_config=LLMEndpointConfig(
            model="gpt-4o",
            env_variable_name=TEST_ENV_NAME,
            llm_api_key=RUNTIME_SECRET,
        ),
        embedding_config=EmbedderConfig(
            config={},
        ),
    )

    serialized = brain.model_dump_json()

    # Neither the original runtime key nor the environment's secret value
    # should be present in persisted JSON.
    assert RUNTIME_SECRET not in serialized
    assert ENV_SECRET not in serialized
    assert "llm_api_key" not in serialized

    restored = BrainSerialized.model_validate_json(serialized)

    assert restored.llm_config.env_variable_name == TEST_ENV_NAME
    assert restored.llm_config.llm_api_key == ENV_SECRET
    assert restored.llm_config.model == "gpt-4o"