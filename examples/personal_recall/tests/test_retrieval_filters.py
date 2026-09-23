"""``RetrievalFilter`` and the SQL predicate it becomes — with no database.

The clause is generated here and executed by `store.py`, which is why these tests can check the
*shape* of every filter — including the ones that must select nothing and the ones that must never
merge two people — on a machine with no PostgreSQL. What they cannot check is that PostgreSQL
returns the right rows; `test_pgvector_retrieval.py` does that against a live server.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory_store.filters import RetrievalFilter, combine, from_mapping  # noqa: E402


def test_an_empty_filter_constrains_nothing() -> None:
    active = RetrievalFilter()
    assert active.is_empty
    assert active.clause() == ("", [])


def test_a_filter_is_empty_only_when_every_field_is() -> None:
    assert not RetrievalFilter(person_ids=("p",)).is_empty
    assert not RetrievalFilter(conversation_ids=("c",)).is_empty
    assert not RetrievalFilter(start_time=datetime(2024, 1, 1)).is_empty
    assert not RetrievalFilter(end_time=datetime(2024, 1, 1)).is_empty


def test_a_conversation_filter_is_one_predicate_with_a_bound_parameter() -> None:
    sql, params = RetrievalFilter(conversation_ids=("11111111@chatroom",)).clause()
    assert sql == "c.conversation_id = ANY(%s)"
    assert params == [["11111111@chatroom"]], "the id is a parameter, never interpolated"


def test_a_person_filter_is_a_semi_join_against_the_evidence_not_against_membership() -> None:
    """"What did this person say" must not return a chunk they merely sat in."""
    sql, params = RetrievalFilter(person_ids=("p1",)).clause()
    assert "chunk_events" in sql and "memory_events" in sql
    assert "speaker_person_id = ANY(%s)" in sql
    assert params == [["p1"]]


def test_two_people_are_a_union_not_a_choice() -> None:
    """Two people share a display name; filtering must not silently pick one of them."""
    sql, params = RetrievalFilter(person_ids=("p1", "p2")).clause()
    assert params == [["p1", "p2"]]
    assert sql.count("ANY(%s)") == 1, "one predicate, one list — not two conjuncts"


def test_a_time_window_is_an_overlap_in_both_directions() -> None:
    """A session that starts before the window and ends inside it contains part of the period."""
    start, end = datetime(2024, 3, 1), datetime(2024, 3, 31)
    sql, params = RetrievalFilter(start_time=start, end_time=end).clause()
    assert "c.end_time >= %s" in sql, "a session running into the window is kept"
    assert "c.start_time <= %s" in sql, "a session beginning before it and continuing past is kept"
    assert params == [start, end]


def test_combined_fields_are_conjoined() -> None:
    active = RetrievalFilter(
        person_ids=("p1",), conversation_ids=("c1",), start_time=datetime(2024, 1, 1)
    )
    sql, params = active.clause()
    # `AND` also appears inside the EXISTS subquery, so presence is checked per predicate rather
    # than by counting a token that means two different things at two different nesting levels.
    assert sql.startswith("c.conversation_id = ANY(%s) AND EXISTS (")
    assert sql.endswith("c.end_time >= %s")
    assert len(params) == 3


def test_the_alias_is_configurable_so_a_join_can_use_it() -> None:
    sql, _ = RetrievalFilter(conversation_ids=("c1",)).clause(alias="chunks")
    assert sql.startswith("chunks.conversation_id")


def test_duplicate_ids_collapse_so_two_spellings_are_one_filter() -> None:
    """The BM25 index is cached by filter; a repeated id must not rebuild a twelve-second index."""
    assert RetrievalFilter(person_ids=("p1", "p1")).person_ids == ("p1",)
    assert RetrievalFilter(person_ids=["p1", "p1"]).key() == RetrievalFilter(person_ids=("p1",)).key()


def test_the_key_is_hashable_and_order_sensitive() -> None:
    assert hash(RetrievalFilter(person_ids=("p1",)).key())
    assert RetrievalFilter(conversation_ids=("a",)).key() != RetrievalFilter(
        conversation_ids=("b",)
    ).key()


def test_describe_names_the_shape_and_never_an_identity() -> None:
    described = RetrievalFilter(
        person_ids=("wxid_real_person",), conversation_ids=("12345@chatroom",)
    ).describe()
    assert "wxid" not in described
    assert "@chatroom" not in described
    assert "1 person(s)" in described and "1 conversation(s)" in described


# ---------------------------------------------------------------------------------------------
# combine / from_mapping
# ---------------------------------------------------------------------------------------------


def test_combining_takes_the_union_of_ids_and_the_intersection_of_times() -> None:
    first = RetrievalFilter(person_ids=("p1",), start_time=datetime(2024, 1, 1))
    second = RetrievalFilter(conversation_ids=("c1",), end_time=datetime(2024, 12, 31))
    merged = combine(first, second)
    assert merged.person_ids == ("p1",)
    assert merged.conversation_ids == ("c1",)
    assert merged.start_time == datetime(2024, 1, 1)
    assert merged.end_time == datetime(2024, 12, 31)


def test_combining_nothing_is_an_empty_filter() -> None:
    assert combine().is_empty


def test_a_loose_mapping_is_read_leniently() -> None:
    """A planner's output is a model's. An unreadable filter must widen the search, not fail."""
    assert from_mapping(None).is_empty
    assert from_mapping("nonsense").is_empty
    assert from_mapping({"person_ids": "p1"}).person_ids == ("p1",)
    assert from_mapping({"people": ["p1", "p2"]}).person_ids == ("p1", "p2")
    assert from_mapping({"conversations": "c1"}).conversation_ids == ("c1",)


def test_an_iso_time_in_a_mapping_becomes_a_naive_local_time() -> None:
    active = from_mapping({"start_time": "2024-03-01T00:00:00"})
    assert active.start_time == datetime(2024, 3, 1)
    assert active.start_time.tzinfo is None, "the stored times carry no zone"


def test_an_offset_time_is_converted_rather_than_compared_across_zones() -> None:
    """Comparing an aware value to a naive column is an error in PostgreSQL, not a near miss."""
    active = from_mapping({"start_time": "2024-03-01T00:00:00+00:00"})
    assert active.start_time is not None
    assert active.start_time.tzinfo is None


def test_an_unparsable_time_is_dropped_rather_than_guessed() -> None:
    assert from_mapping({"start_time": "whenever"}).start_time is None


def test_empty_strings_do_not_become_id_filters() -> None:
    assert from_mapping({"person_ids": ["", None]}).is_empty


@pytest.mark.parametrize("field", ["person_ids", "conversation_ids", "start_time", "end_time"])
def test_every_field_is_optional(field: str) -> None:
    active = from_mapping({field: None})
    assert active.is_empty
