"""SQLite replicated store."""

from __future__ import annotations

from pathlib import Path

from edgecluster.sync.store import CounterSlot, Entry, ReplicatedStore


def test_put_get_delete_and_count(tmp_path: Path) -> None:
    s = ReplicatedStore(tmp_path / "a.db", "a")
    s.put("cases", "c1", {"status": "open"})
    s.put("cases", "c2", {"status": "open"})
    assert s.get("cases", "c1") == {"status": "open"}
    s.delete("cases", "c1")
    assert s.get("cases", "c1") is None
    assert s.get_entry("cases", "c1").deleted
    assert s.count("cases") == 1
    assert s.count("cases", include_deleted=True) == 2
    assert [k for k, _ in s.list("cases")] == ["c2"]


def test_unicode_roundtrip(tmp_path: Path) -> None:
    s = ReplicatedStore(tmp_path / "a.db", "a")
    s.put("queries", "q", {"text": "راشن کارڈ گم ہو گیا ہے", "hi": "पेंशन"})
    assert s.get("queries", "q")["text"] == "راشن کارڈ گم ہو گیا ہے"


def test_apply_is_idempotent_and_lww(tmp_path: Path) -> None:
    a = ReplicatedStore(tmp_path / "a.db", "a", physical_ms=lambda: 1000)
    b = ReplicatedStore(tmp_path / "b.db", "b", physical_ms=lambda: 2000)
    ea = a.put("cases", "c", {"v": "from-a"})
    eb = b.put("cases", "c", {"v": "from-b"})  # later physical time
    assert a.apply([eb]) == 1
    assert a.apply([eb]) == 0  # idempotent
    assert b.apply([ea]) == 0  # older write loses
    assert a.get("cases", "c") == b.get("cases", "c") == {"v": "from-b"}
    assert a.digest() != b.digest() or a.snapshot() == b.snapshot()
    assert a.snapshot() == b.snapshot()


def test_changes_since_paginates_in_seq_order(tmp_path: Path) -> None:
    s = ReplicatedStore(tmp_path / "a.db", "a")
    for i in range(7):
        s.put("q", f"k{i}", i)
    s.increment("hits", 2)
    seen, since = [], 0
    while True:
        batch = s.changes_since(since, limit=3)
        seen += [e.key for e in batch.entries] + [c.name for c in batch.counters]
        since = batch.upto
        if not batch.has_more:
            break
    assert seen == [f"k{i}" for i in range(7)] + ["hits"]
    assert s.changes_since(since).entries == []


def test_updated_row_moves_to_new_seq(tmp_path: Path) -> None:
    s = ReplicatedStore(tmp_path / "a.db", "a")
    s.put("q", "k", 1)
    cursor = s.changes_since(0).upto
    s.put("q", "k", 2)
    batch = s.changes_since(cursor)
    assert [(e.key, e.value) for e in batch.entries] == [("k", 2)]


def test_persistence_and_clock_monotonic_across_restart(tmp_path: Path) -> None:
    s = ReplicatedStore(tmp_path / "a.db", "a", physical_ms=lambda: 50_000)
    e1 = s.put("q", "k", 1)
    s.increment("n", 3)
    store_id = s.store_id
    s.close()
    s2 = ReplicatedStore(tmp_path / "a.db", "a", physical_ms=lambda: 10_000)  # clock went backwards
    assert s2.store_id == store_id
    assert s2.get("q", "k") == 1
    assert s2.counter("n").value == 3
    e2 = s2.put("q", "k", 2)
    assert e2.ts > e1.ts
    assert s2.get("q", "k") == 2


def test_counter_merge(tmp_path: Path) -> None:
    a = ReplicatedStore(tmp_path / "a.db", "a")
    b = ReplicatedStore(tmp_path / "b.db", "b")
    a.increment("queries:pension", 3)
    b.increment("queries:pension", 2)
    b.increment("queries:pension", -1)
    a.apply([], b.changes_since(0).counters)
    b.apply([], a.changes_since(0).counters)
    assert a.counter("queries:pension").value == b.counter("queries:pension").value == 4
    # stale slot never lowers a counter
    assert a.apply([], [CounterSlot("queries:pension", "a", 1, 0)]) == 0


def test_wire_roundtrip(tmp_path: Path) -> None:
    s = ReplicatedStore(tmp_path / "a.db", "a")
    e = s.put("cases", "x", {"a": [1, 2]})
    assert Entry.from_wire(e.to_wire()) == e
