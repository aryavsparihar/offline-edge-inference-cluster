"""Hypothesis strategies producing arbitrary (reachable and unreachable) CRDT states."""

from __future__ import annotations

from hypothesis import strategies as st

from edgecluster.sync.crdt import GCounter, LWWMap, LWWRegister, ORSet, PNCounter
from edgecluster.sync.crdt.lwwmap import TOMBSTONE, entry_value
from edgecluster.sync.hlc import Timestamp

node_ids = st.sampled_from(["a", "b", "c", "d"])
json_scalars = st.one_of(st.none(), st.booleans(), st.integers(-1000, 1000), st.text(max_size=8))
json_values = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=3), st.dictionaries(st.text(max_size=4), children, max_size=3)
    ),
    max_leaves=6,
)
timestamps = st.builds(Timestamp, st.integers(0, 50), st.integers(0, 5), node_ids)

gcounters = st.dictionaries(node_ids, st.integers(0, 10_000)).map(GCounter)
pncounters = st.builds(PNCounter, gcounters, gcounters)
registers = st.builds(LWWRegister, json_values, timestamps)
tags = st.tuples(node_ids, st.integers(1, 6))
orsets = st.builds(
    ORSet,
    st.dictionaries(st.sampled_from(["x", "y", "z", "w"]), st.frozensets(tags, max_size=4), max_size=4),
    st.frozensets(tags, max_size=6),
)
map_payloads = st.one_of(json_values.map(entry_value), st.just(dict(TOMBSTONE)))
lwwmaps = st.dictionaries(
    st.sampled_from(["k1", "k2", "k3", "k4"]),
    st.builds(LWWRegister, map_payloads, timestamps),
    max_size=4,
).map(LWWMap)

ALL = {
    "GCounter": gcounters,
    "PNCounter": pncounters,
    "LWWRegister": registers,
    "ORSet": orsets,
    "LWWMap": lwwmaps,
}
