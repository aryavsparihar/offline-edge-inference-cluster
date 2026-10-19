"""Every CRDT merge is commutative, associative and idempotent (a join-semilattice)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.strategies import ALL

TYPES = sorted(ALL)


def _st(name: str) -> st.SearchStrategy:
    return ALL[name]


@pytest.mark.parametrize("name", TYPES)
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_merge_commutative(name: str, data: st.DataObject) -> None:
    a, b = data.draw(_st(name)), data.draw(_st(name))
    assert a.merge(b) == b.merge(a)


@pytest.mark.parametrize("name", TYPES)
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_merge_associative(name: str, data: st.DataObject) -> None:
    a, b, c = data.draw(_st(name)), data.draw(_st(name)), data.draw(_st(name))
    assert a.merge(b).merge(c) == a.merge(b.merge(c))


@pytest.mark.parametrize("name", TYPES)
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_merge_idempotent(name: str, data: st.DataObject) -> None:
    a = data.draw(_st(name))
    assert a.merge(a) == a


@pytest.mark.parametrize("name", TYPES)
@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_merge_is_upper_bound(name: str, data: st.DataObject) -> None:
    """Merging never loses information: (a ⊔ b) ⊔ a == a ⊔ b."""
    a, b = data.draw(_st(name)), data.draw(_st(name))
    m = a.merge(b)
    assert m.merge(a) == m
    assert m.merge(b) == m


@pytest.mark.parametrize("name", TYPES)
@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_serialization_roundtrip(name: str, data: st.DataObject) -> None:
    a = data.draw(_st(name))
    assert type(a).from_dict(a.to_dict()) == a


@pytest.mark.parametrize("name", TYPES)
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_merge_order_independent(name: str, data: st.DataObject) -> None:
    """Folding any permutation of the same states yields the same result."""
    states = data.draw(st.lists(_st(name), min_size=1, max_size=5))
    perm = data.draw(st.permutations(states))
    fold = states[0]
    for s in states[1:]:
        fold = fold.merge(s)
    fold2 = perm[0]
    for s in perm[1:]:
        fold2 = fold2.merge(s)
    assert fold == fold2
