from pathlib import Path
from uuid import uuid4

import pytest

from quivr_core.brain.serialization import LocalStorageConfig
from quivr_core.files.file import QuivrFileSerialized, load_qfile
from quivr_core.storage.local_storage import LocalStorage


def test_local_storage_load_restores_hash_index(tmp_path: Path):
    """Loading LocalStorage must rebuild hashes from persisted files."""

    brain_id = uuid4()
    file_a_id = uuid4()
    file_b_id = uuid4()

    file_a = QuivrFileSerialized(
        id=file_a_id,
        brain_id=brain_id,
        path=tmp_path / "stored-a.txt",
        original_filename="a.txt",
        file_size=10,
        file_extension=".txt",
        file_sha1="sha1-a",
        additional_metadata={},
    )

    file_b = QuivrFileSerialized(
        id=file_b_id,
        brain_id=brain_id,
        path=tmp_path / "stored-b.txt",
        original_filename="b.txt",
        file_size=20,
        file_extension=".txt",
        file_sha1="sha1-b",
        additional_metadata={},
    )

    config = LocalStorageConfig(
        storage_path=tmp_path / "storage",
        files={
            file_a_id: file_a,
            file_b_id: file_b,
        },
    )

    loaded = LocalStorage.load(config)

    assert len(loaded.files) == 2

    assert loaded.hashes == {
        file.file_sha1
        for file in loaded.files
    }

    assert loaded.hashes == {
        "sha1-a",
        "sha1-b",
    }


@pytest.mark.asyncio
async def test_duplicate_detection_still_works_after_local_storage_load(
    tmp_path: Path,
):
    """
    A storage restored from persisted config must still reject a file
    whose content hash was already present before persistence.
    """

    brain_id = uuid4()

    source_a = tmp_path / "file_a.txt"
    source_b = tmp_path / "file_b.txt"

    source_a.write_text("same content", encoding="utf-8")
    source_b.write_text("same content", encoding="utf-8")

    file_a = await load_qfile(brain_id, source_a)
    file_b = await load_qfile(brain_id, source_b)

    # Different source files with identical content must have the same SHA1.
    assert file_a.id != file_b.id
    assert file_a.path != file_b.path
    assert file_a.file_sha1 == file_b.file_sha1

    storage_path = tmp_path / "storage"

    # upload_file() writes beneath:
    # storage_path / brain_id / <file-id>.txt
    (storage_path / str(brain_id)).mkdir(parents=True)

    storage = LocalStorage(dir_path=storage_path)

    await storage.upload_file(
        file_a,
        exists_ok=False,
    )

    assert storage.hashes == {file_a.file_sha1}

    # Simulate the persisted LocalStorageConfig used during save/load.
    config = LocalStorageConfig(
        storage_path=storage_path,
        files={
            file.id: file.serialize()
            for file in storage.files
        },
    )

    loaded = LocalStorage.load(config)

    assert loaded.hashes == {
        file.file_sha1
        for file in loaded.files
    }

    # This is the important behavioral regression check:
    # before the fix, load() left hashes empty and this duplicate slipped through.
    with pytest.raises(FileExistsError):
        await loaded.upload_file(
            file_b,
            exists_ok=False,
        )