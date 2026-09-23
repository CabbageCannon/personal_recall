"""Makes ``examples/personal_recall`` importable for the test suite.

pytest inserts the directory containing the root ``conftest.py`` into
``sys.path``, which is what lets ``tests/`` import ``memory`` and ``eval_utils``
without installing the example as a package.
"""

from __future__ import annotations


def pytest_configure(config) -> None:  # noqa: ANN001 - the pytest hook signature
    """Register the marker the PostgreSQL tests carry.

    Registered here rather than in a ``pytest.ini`` so that adding the store's tests changes nothing
    about how the existing suite is collected: ``pytest tests`` still collects the same files as
    before, and the provenance of the list is unchanged.

    ``pytest -m postgres`` runs only the tests that need a live server; ``pytest -m "not postgres"``
    runs everything else on a machine with none. The projection tests in
    ``test_memory_projection.py`` are deliberately **not** marked: the properties they check must
    hold everywhere, not only where a container happens to be running.
    """
    config.addinivalue_line(
        "markers",
        "postgres: requires a live PostgreSQL server "
        "(set PERSONAL_RECALL_DATABASE_URL; see .env.example)",
    )
