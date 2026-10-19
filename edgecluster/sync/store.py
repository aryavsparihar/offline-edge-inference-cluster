"""SQLite-backed replicated record store.

The store holds two kinds of replicated state:

* **records**: ``(collection, key) -> value`` entries, each an
  :class:`~edgecluster.sync.crdt.LWWRegister` (one :class:`LWWMap` per
  collection). Used for welfare query logs and case records.
* **counters**: named :class:`~edgecluster.sync.crdt.PNCounter` values,
  stored one row per contributing node (e.g. queries per category).

Every row that changes, through a local write or a merge, gets the next
value of a node-local sequence number ``seq`` and remembers ``src``, the node
it received the current version from. A peer that pulls is never sent rows
whose current version it supplied itself, so changes don't echo back. Peers remember the highest
``seq`` they have pulled, so anti-entropy only ships the delta. ``store_id``
is a random identifier minted when the database is created. If a node's disk
is wiped its seq space restarts, and the new ``store_id`` tells peers to
discard their cursors.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edgecluster.sync.crdt import GCounter, LWWRegister, PNCounter, canonical_json
from edgecluster.sync.crdt.lwwmap import TOMBSTONE, entry_value
from edgecluster.sync.hlc import HybridLogicalClock, Timestamp, wall_clock_ms

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    collection TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT,
    deleted    INTEGER NOT NULL,
    ts         TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    src        TEXT NOT NULL,
    PRIMARY KEY (collection, key)
);
CREATE INDEX IF NOT EXISTS records_seq ON records(seq);
CREATE TABLE IF NOT EXISTS counters (
    name    TEXT NOT NULL,
    node_id TEXT NOT NULL,
    p       INTEGER NOT NULL,
    n       INTEGER NOT NULL,
    seq     INTEGER NOT NULL,
    src     TEXT NOT NULL,
    PRIMARY KEY (name, node_id)
);
CREATE INDEX IF NOT EXISTS counters_seq ON counters(seq);
CREATE TABLE IF NOT EXISTS cursors (
    peer_id   TEXT NOT NULL,
    direction TEXT NOT NULL,
    store_id  TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    PRIMARY KEY (peer_id, direction)
);
"""


@dataclass(frozen=True)
class Entry:
    """One replicated record version."""

    collection: str
    key: str
    value: Any
    deleted: bool
    ts: Timestamp

    def register(self) -> LWWRegister:
        """The LWW register state this entry represents."""
        payload = dict(TOMBSTONE) if self.deleted else entry_value(self.value)
        return LWWRegister(payload, self.ts)

    @classmethod
    def from_register(cls, collection: str, key: str, reg: LWWRegister) -> Entry:
        """Build an entry from a register produced by a merge."""
        deleted = bool(reg.value["deleted"])
        return cls(collection, key, None if deleted else reg.value["value"], deleted, reg.timestamp)

    def to_wire(self) -> dict[str, Any]:
        """JSON-friendly representation for the sync protocol."""
        return {
            "c": self.collection,
            "k": self.key,
            "v": self.value,
            "d": self.deleted,
            "ts": self.ts.to_list(),
        }

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> Entry:
        """Inverse of :meth:`to_wire`."""
        return cls(
            str(raw["c"]), str(raw["k"]), raw.get("v"), bool(raw["d"]), Timestamp.from_list(raw["ts"])
        )


@dataclass(frozen=True)
class CounterSlot:
    """One node's contribution to a named PN-counter."""

    name: str
    node_id: str
    p: int
    n: int

    def counter(self) -> PNCounter:
        """This slot as a single-slot PN-counter."""
        return PNCounter(GCounter({self.node_id: self.p}), GCounter({self.node_id: self.n}))

    def to_wire(self) -> dict[str, Any]:
        """JSON-friendly representation for the sync protocol."""
        return {"name": self.name, "node": self.node_id, "p": self.p, "n": self.n}

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> CounterSlot:
        """Inverse of :meth:`to_wire`."""
        return cls(str(raw["name"]), str(raw["node"]), int(raw["p"]), int(raw["n"]))


@dataclass(frozen=True)
class ChangeBatch:
    """A contiguous slice ``(since, upto]`` of a store's change log."""

    store_id: str
    entries: list[Entry]
    counters: list[CounterSlot]
    upto: int
    has_more: bool


class ReplicatedStore:
    """Persistent CRDT store for one node.

    Args:
        path: SQLite database file (``":memory:"`` for ephemeral stores).
        node_id: This node's id, also used for HLC timestamps.
        physical_ms: Physical clock source (injectable for simulation).
        max_drift_ms: Forwarded to :class:`HybridLogicalClock`.
    """

    def __init__(
        self,
        path: str | Path,
        node_id: str,
        physical_ms: Callable[[], int] = wall_clock_ms,
        max_drift_ms: int | None = None,
    ) -> None:
        self.path = str(path)
        self.node_id = node_id
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        with self._tx():
            self.store_id = self._meta("store_id") or self._set_meta("store_id", uuid.uuid4().hex)
            self._seq = int(self._meta("seq") or 0)
            last = self._meta("hlc")
        self.clock = HybridLogicalClock(
            node_id,
            physical_ms=physical_ms,
            max_drift_ms=max_drift_ms,
            last=Timestamp.decode(last) if last else None,
        )

    # ------------------------------------------------------------------ internals
    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> str:
        self._db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        return value

    def _next_seq(self) -> int:
        self._seq += 1
        self._set_meta("seq", str(self._seq))
        return self._seq

    def _persist_clock(self) -> None:
        self._set_meta("hlc", self.clock.last.encode())

    def _load_entry(self, collection: str, key: str) -> Entry | None:
        row = self._db.execute(
            "SELECT value, deleted, ts FROM records WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0]) if row[0] is not None else None
        return Entry(collection, key, value, bool(row[1]), Timestamp.decode(row[2]))

    def _write_entry(self, entry: Entry, src: str) -> None:
        self._db.execute(
            "INSERT INTO records(collection, key, value, deleted, ts, seq, src) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(collection, key) DO UPDATE SET value = excluded.value, "
            "deleted = excluded.deleted, ts = excluded.ts, seq = excluded.seq, src = excluded.src",
            (
                entry.collection,
                entry.key,
                None if entry.deleted else canonical_json(entry.value),
                int(entry.deleted),
                entry.ts.encode(),
                self._next_seq(),
                src,
            ),
        )

    def _merge_entry(self, incoming: Entry, src: str) -> bool:
        local = self._load_entry(incoming.collection, incoming.key)
        if local is None:
            self._write_entry(incoming, src)
            return True
        merged = local.register().merge(incoming.register())
        if merged == local.register():
            return False
        self._write_entry(Entry.from_register(incoming.collection, incoming.key, merged), src)
        return True

    def _merge_counter(self, slot: CounterSlot, src: str) -> bool:
        row = self._db.execute(
            "SELECT p, n FROM counters WHERE name = ? AND node_id = ?", (slot.name, slot.node_id)
        ).fetchone()
        local = CounterSlot(slot.name, slot.node_id, row[0], row[1]) if row else None
        if local is None:
            merged = slot.counter()
        else:
            merged = local.counter().merge(slot.counter())
            if merged == local.counter():
                return False
        self._db.execute(
            "INSERT INTO counters(name, node_id, p, n, seq, src) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(name, node_id) DO UPDATE SET p = excluded.p, n = excluded.n, "
            "seq = excluded.seq, src = excluded.src",
            (
                slot.name,
                slot.node_id,
                merged.p.counts.get(slot.node_id, 0),
                merged.n.counts.get(slot.node_id, 0),
                self._next_seq(),
                src,
            ),
        )
        return True

    # ------------------------------------------------------------------ local API
    def put(self, collection: str, key: str, value: Any) -> Entry:
        """Write a record locally (always succeeds, even when offline)."""
        with self._tx():
            entry = Entry(collection, key, value, False, self.clock.now())
            self._merge_entry(entry, self.node_id)
            self._persist_clock()
        return entry

    def delete(self, collection: str, key: str) -> Entry:
        """Tombstone a record locally."""
        with self._tx():
            entry = Entry(collection, key, None, True, self.clock.now())
            self._merge_entry(entry, self.node_id)
            self._persist_clock()
        return entry

    def get(self, collection: str, key: str) -> Any:
        """Current value of a record, or ``None`` if missing/deleted."""
        with self._lock:
            entry = self._load_entry(collection, key)
        return None if entry is None or entry.deleted else entry.value

    def get_entry(self, collection: str, key: str) -> Entry | None:
        """Raw entry including tombstones and timestamp."""
        with self._lock:
            return self._load_entry(collection, key)

    def list(self, collection: str, limit: int = 100, offset: int = 0) -> list[tuple[str, Any]]:
        """Live records of a collection ordered by key."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value FROM records WHERE collection = ? AND deleted = 0 "
                "ORDER BY key LIMIT ? OFFSET ?",
                (collection, limit, offset),
            ).fetchall()
        return [(k, json.loads(v)) for k, v in rows]

    def count(self, collection: str | None = None, include_deleted: bool = False) -> int:
        """Number of records, optionally restricted to one collection."""
        sql, args = "SELECT COUNT(*) FROM records WHERE 1=1", []
        if collection is not None:
            sql += " AND collection = ?"
            args.append(collection)
        if not include_deleted:
            sql += " AND deleted = 0"
        with self._lock:
            return int(self._db.execute(sql, args).fetchone()[0])

    def increment(self, name: str, amount: int = 1) -> int:
        """Add ``amount`` (may be negative) to this node's slot of counter ``name``."""
        with self._tx():
            row = self._db.execute(
                "SELECT p, n FROM counters WHERE name = ? AND node_id = ?", (name, self.node_id)
            ).fetchone()
            p, n = row if row else (0, 0)
            if amount >= 0:
                p += amount
            else:
                n -= amount
            self._merge_counter(CounterSlot(name, self.node_id, p, n), self.node_id)
        return self.counter(name).value

    def counter(self, name: str) -> PNCounter:
        """Full PN-counter state for ``name``."""
        with self._lock:
            rows = self._db.execute(
                "SELECT node_id, p, n FROM counters WHERE name = ?", (name,)
            ).fetchall()
        return PNCounter(GCounter({r[0]: r[1] for r in rows}), GCounter({r[0]: r[2] for r in rows}))

    def counter_names(self) -> list[str]:
        """Names of all counters known to this replica."""
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT name FROM counters ORDER BY name").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ replication API
    @property
    def max_seq(self) -> int:
        """Highest sequence number assigned so far."""
        with self._lock:
            return self._seq

    def changes_since(self, since: int, limit: int = 500, exclude_src: str | None = None) -> ChangeBatch:
        """Return the next ``limit`` changes with ``seq > since``, in seq order.

        Rows whose current version came from ``exclude_src`` are skipped (that
        node already has them) but still advance ``upto``.
        """
        with self._lock:
            erows = self._db.execute(
                "SELECT seq, collection, key, value, deleted, ts, src FROM records "
                "WHERE seq > ? ORDER BY seq LIMIT ?",
                (since, limit),
            ).fetchall()
            crows = self._db.execute(
                "SELECT seq, name, node_id, p, n, src FROM counters WHERE seq > ? ORDER BY seq LIMIT ?",
                (since, limit),
            ).fetchall()
            merged = sorted([("e", r) for r in erows] + [("c", r) for r in crows], key=lambda x: x[1][0])
            merged = merged[:limit]
            entries: list[Entry] = []
            counters: list[CounterSlot] = []
            for kind, r in merged:
                if exclude_src is not None and r[-1] == exclude_src:
                    continue
                if kind == "e":
                    value = json.loads(r[3]) if r[3] is not None else None
                    entries.append(Entry(r[1], r[2], value, bool(r[4]), Timestamp.decode(r[5])))
                else:
                    counters.append(CounterSlot(r[1], r[2], r[3], r[4]))
            # Rows only ever move to higher seqs, so if nothing is newer than
            # ``since`` the requester is fully caught up to our current seq.
            upto = merged[-1][1][0] if merged else max(since, self._seq)
            return ChangeBatch(self.store_id, entries, counters, upto, upto < self._seq)

    def apply(
        self, entries: Iterable[Entry], counters: Iterable[CounterSlot] = (), source: str = "?"
    ) -> int:
        """Merge state received from node ``source``. Returns the number of rows changed."""
        changed = 0
        with self._tx():
            for e in entries:
                self.clock.update(e.ts)
                changed += self._merge_entry(e, source)
            for c in counters:
                changed += self._merge_counter(c, source)
            self._persist_clock()
        return changed

    def get_cursor(self, peer_id: str, direction: str, store_id: str | None) -> int:
        """Replication cursor for ``peer_id``; 0 if unknown or the peer's store changed."""
        with self._lock:
            row = self._db.execute(
                "SELECT store_id, seq FROM cursors WHERE peer_id = ? AND direction = ?",
                (peer_id, direction),
            ).fetchone()
        if row is None or (store_id is not None and row[0] != store_id):
            return 0
        return int(row[1])

    def known_store_id(self, peer_id: str, direction: str) -> str | None:
        """Last ``store_id`` seen for ``peer_id`` in ``direction``."""
        with self._lock:
            row = self._db.execute(
                "SELECT store_id FROM cursors WHERE peer_id = ? AND direction = ?",
                (peer_id, direction),
            ).fetchone()
        return row[0] if row else None

    def set_cursor(self, peer_id: str, direction: str, store_id: str, seq: int) -> None:
        """Persist a replication cursor."""
        with self._tx():
            self._db.execute(
                "INSERT INTO cursors(peer_id, direction, store_id, seq) VALUES(?,?,?,?) "
                "ON CONFLICT(peer_id, direction) DO UPDATE SET store_id = excluded.store_id, "
                "seq = excluded.seq",
                (peer_id, direction, store_id, seq),
            )

    # ------------------------------------------------------------------ inspection
    def snapshot(self) -> dict[tuple[str, str], tuple[str, bool, str | None]]:
        """All record versions as ``(collection, key) -> (ts, deleted, value_json)``."""
        with self._lock:
            rows = self._db.execute("SELECT collection, key, ts, deleted, value FROM records").fetchall()
        return {(r[0], r[1]): (r[2], bool(r[3]), r[4]) for r in rows}

    def counter_snapshot(self) -> dict[tuple[str, str], tuple[int, int]]:
        """All counter slots as ``(name, node_id) -> (p, n)``."""
        with self._lock:
            rows = self._db.execute("SELECT name, node_id, p, n FROM counters").fetchall()
        return {(r[0], r[1]): (r[2], r[3]) for r in rows}

    def digest(self) -> str:
        """SHA-256 over the full replicated state; equal digests mean equal state."""
        h = hashlib.sha256()
        for (c, k), (ts, deleted, value) in sorted(self.snapshot().items()):
            h.update(canonical_json([c, k, ts, deleted, value]).encode())
        for (name, node), (p, n) in sorted(self.counter_snapshot().items()):
            h.update(canonical_json(["#", name, node, p, n]).encode())
        return h.hexdigest()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()
