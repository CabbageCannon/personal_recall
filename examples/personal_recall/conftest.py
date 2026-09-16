"""Makes ``examples/personal_recall`` importable for the test suite.

pytest inserts the directory containing the root ``conftest.py`` into
``sys.path``, which is what lets ``tests/`` import ``memory`` and ``eval_utils``
without installing the example as a package.
"""
