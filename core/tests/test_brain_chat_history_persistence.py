from uuid import uuid4

import langchain_openai
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

import quivr_core.brain.brain as brain_module
from quivr_core.brain import Brain
from quivr_core.rag.entities.config import LLMEndpointConfig
from quivr_core.storage.local_storage import TransparentStorage


class FakeOpenAIEmbeddings(Embeddings):
    """Offline embeddings used only for FAISS save/load in this test."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), 1.0, 0.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]

    def dict(self, *args, **kwargs):
        # Brain.save() serializes the embedder configuration.
        return {}


class FakeLLM:
    def __init__(self):
        self.config = LLMEndpointConfig(model="gpt-4o")

    def get_config(self):
        return self.config


@pytest.mark.asyncio
async def test_brain_save_load_preserves_chat_history(monkeypatch, tmp_path):
    """
    Saving and loading a Brain must rehydrate the existing ChatHistory
    instead of creating a new set of ChatMessage objects.
    """

    fake_llm = FakeLLM()
    embedder = FakeOpenAIEmbeddings()

    # Brain.save() checks isinstance(embedder, OpenAIEmbeddings).
    monkeypatch.setattr(
        brain_module,
        "OpenAIEmbeddings",
        FakeOpenAIEmbeddings,
    )

    # Brain.load() imports OpenAIEmbeddings locally from langchain_openai.
    monkeypatch.setattr(
        langchain_openai,
        "OpenAIEmbeddings",
        FakeOpenAIEmbeddings,
    )

    # Prevent Brain.load() from constructing a real external LLM.
    monkeypatch.setattr(
        brain_module.LLMEndpoint,
        "from_config",
        staticmethod(lambda config: fake_llm),
    )

    vector_db = FAISS.from_texts(
        ["local test document"],
        embedder,
    )

    brain_id = uuid4()

    brain = Brain(
        id=brain_id,
        name="chat-history-persistence-test",
        llm=fake_llm,
        embedder=embedder,
        storage=TransparentStorage(),
        vector_db=vector_db,
    )

    original_chat_id = brain.chat_history.id

    brain.chat_history.append(
        HumanMessage(content="hello"),
        metadata={"source": "test"},
    )
    brain.chat_history.append(
        AIMessage(content="hi"),
    )
    brain.chat_history.append(
        HumanMessage(content="What is Quivr?"),
    )
    brain.chat_history.append(
        AIMessage(content="A RAG framework"),
        metadata={"source": "assistant-test"},
    )

    original_messages = brain.chat_history.get_chat_history()

    assert len(original_messages) == 4

    brain_path = await brain.save(tmp_path)

    loaded_brain = Brain.load(brain_path)

    restored_messages = loaded_brain.chat_history.get_chat_history()

    # ChatHistory identity itself must be restored.
    assert loaded_brain.chat_history.id == original_chat_id
    assert loaded_brain.chat_id == original_chat_id

    # Brain's internal chat references must remain consistent.
    assert loaded_brain.default_chat is loaded_brain.chat_history
    assert loaded_brain._chats[original_chat_id] is loaded_brain.default_chat

    # Brain association must also survive the round trip.
    assert loaded_brain.id == brain_id
    assert loaded_brain.chat_history.brain_id == brain_id

    # The complete message history must survive.
    assert len(restored_messages) == len(original_messages)

    for original, restored in zip(
        original_messages,
        restored_messages,
        strict=True,
    ):
        # Message ordering and Human/AI type.
        assert type(restored.msg) is type(original.msg)

        # User-visible content.
        assert restored.msg.content == original.msg.content

        # Persisted ChatMessage identity/state.
        assert restored.message_id == original.message_id
        assert restored.message_time == original.message_time
        assert restored.metadata == original.metadata
        assert restored.chat_id == original.chat_id
        assert restored.brain_id == original.brain_id