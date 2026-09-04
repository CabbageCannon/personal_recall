import os
import tempfile
from uuid import uuid4
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

from quivr_core import Brain
from quivr_core.llm import LLMEndpoint
from quivr_core.rag.entities.config import (
    DefaultModelSuppliers,
    LLMEndpointConfig,
)
from quivr_core.files.file import FileExtension
from quivr_core.processor.registry import register_processor
from quivr_core.processor.implementations.simple_txt_processor import (
    SimpleTxtProcessor,
)

dotenv.load_dotenv()


if __name__ == "__main__":
    llm_config = LLMEndpointConfig(
        supplier=DefaultModelSuppliers.OPENAI,
        model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
        env_variable_name="DEEPSEEK_API_KEY",
        max_context_tokens=20000,
        max_output_tokens=4096,
    )

    llm = LLMEndpoint.from_config(llm_config)

    embedder = HuggingFaceEmbeddings(
        model_name=r"D:\AIModels\bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        encoding="utf-8",
        delete=False,
    )

    try:
        temp_file.write(
            "DevPilot 是一个面向研发团队的企业知识库与 Skill Agent。"
        )
        temp_file.close()

        register_processor(
            FileExtension.txt,
            SimpleTxtProcessor,
            override=True,
        )

        brain = Brain.from_files(
            name="test_brain",
            file_paths=[temp_file.name],
            llm=llm,
            embedder=embedder,
        )

        answer = answer = brain.ask(
            run_id=uuid4(),
            question="DevPilot 是什么？请根据知识库内容回答。",
        )

        print("answer:", answer)

    finally:
        os.unlink(temp_file.name)