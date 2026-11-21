"""SQLite persistence for offline tuning tickets, merges and issuances.

The tuning store lives in the *same database file* as the master
:class:`~wrsource.source.SourceService`, but in disjoint tables (the
``tune_*`` prefix).  Master products (``versions`` / ``version_rows`` /
``rows``) are read-only from here, so an offline preview can never mutate a
published master version and the old product stays queryable on its own.
Offline body edits are appended to the shared content-addressed ``bodies``
table, which cannot disturb an existing snapshot that references a digest.

Identity model
--------------

* ``ticket_id``   = content hash of one branch ticket (diffs only).
* ``merge_id``    = content hash of the pair + merged rows + conflicts.
* ``rehearsal_id``= content hash of (merge, seed, quotas, promises).
* ``issuance_no`` = caller-supplied human ticket number.  One number can ever
  bind to one ``(merge content, rehearsal)`` pair; retries are idempotent while
  the content matches and rejected otherwise, even after the master baseline
  has advanced.

A merged source version is allocated only when an issuance is *approved*, and
the sample rows flip to the published reference in the same transaction, so
the merged source version and its paired sample become visible together.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import tuning
from .sampling import lottery_a
from .source import SourceError
from .tuning import (MergeResult, Promise, Rehearsal, Ticket,
                     canonical_json, content_hash, effective_view,
                     merge_tickets, parse_promise, parse_ticket,
                     run_rehearsal, validate_against_baseline)


class TuneStoreError(SourceError):
    pass


class IssuanceError(TuneStoreError):
    pass


# ---------------------------------------------------------------------------
# Read-only view of a merged source, shaped like the subset of SourceService
# consumed by wrsource.dataset (so the unchanged exact producer runs on it).
# ---------------------------------------------------------------------------

class MergedView:
    """Dataset-producer adapter over one merged (possibly preview-only) row set."""

    def __init__(self, store: "TuneStore", merge_id: str):
        meta = store.get_merge(merge_id)
        if meta["status"] != "ok":
            raise TuneStoreError(
                f"merge {merge_id[:12]} is conflicted; cannot open a view")
        self.store = store
        self.merge_id = merge_id
        self._rows = store.effective_rows(merge_id)
        self._by_id = {r["row_id"]: r for r in self._rows}

    def get_version(self, version: Optional[int] = None) -> dict:
        return {"version_id": f"merged-preview:{self.merge_id}",
                "row_count": sum(1 for r in self._rows if r["active"]),
                "note": "offline merged preview"}

    def group_counts(self, version: Optional[int] = None) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self._rows:
            if r["active"]:
                out[r["group"]] = out.get(r["group"], 0) + 1
        return out

    def version_groups(self, version: Optional[int] = None) -> List[str]:
        return sorted(self.group_counts())

    def all_rows_meta(self, version: Optional[int] = None) -> List[dict]:
        return [{
            "row_id": r["row_id"], "group": r["group"], "weight": r["weight"],
            "body_sha256": r["body_sha256"], "rev": r["rev"],
            "shard": "offline-merged",
        } for r in self._rows if r["active"]]

    def get_rows(self, version: Optional[int],
                 row_ids: Sequence[str]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for rid in dict.fromkeys(row_ids):
            r = self._by_id.get(rid)
            if r is None or not r["active"]:
                continue
            body = self.store.read_body(r["body_sha256"])
            out[rid] = {
                "row_id": rid, "group": r["group"], "weight": r["weight"],
                "rev": r["rev"], "shard": "offline-merged",
                "body_sha256": r["body_sha256"], "body": body,
            }
        return out


class TuneStore:
    def __init__(self, path: str):
        self.path = path
        self._initialised = False

    # -- plumbing --------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def init(self) -> None:
        if not os.path.exists(self.path):
            raise TuneStoreError(
                f"{self.path} does not exist; initialise a SourceService first")
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS tune_tickets(
                    ticket_id TEXT PRIMARY KEY,
                    branch_id TEXT NOT NULL,
                    base_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tune_merges(
                    merge_id TEXT PRIMARY KEY,
                    base_version INTEGER NOT NULL,
                    left_ticket_id TEXT NOT NULL,
                    right_ticket_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('ok','conflicted')),
                    content_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tune_merge_rows(
                    merge_id TEXT NOT NULL REFERENCES tune_merges(merge_id),
                    row_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    weight INTEGER NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    rev INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    changed INTEGER NOT NULL,
                    sources_json TEXT NOT NULL,
                    PRIMARY KEY(merge_id, row_id)
                );
                CREATE TABLE IF NOT EXISTS tune_rehearsals(
                    rehearsal_id TEXT PRIMARY KEY,
                    merge_id TEXT NOT NULL REFERENCES tune_merges(merge_id),
                    base_version INTEGER NOT NULL,
                    merge_content_sha256 TEXT NOT NULL,
                    seed_hex TEXT NOT NULL,
                    quotas_json TEXT NOT NULL,
                    promises_json TEXT NOT NULL,
                    promise_ok INTEGER NOT NULL,
                    violations_json TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    selected_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tune_issuances(
                    issuance_no TEXT PRIMARY KEY,
                    merge_id TEXT NOT NULL,
                    rehearsal_id TEXT NOT NULL,
                    merge_content_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('pending','approved','rejected')),
                    merged_version TEXT,
                    seed_hex TEXT NOT NULL,
                    sample_records_sha256 TEXT,
                    note TEXT,
                    created_at REAL NOT NULL,
                    decided_at REAL
                );
                CREATE TABLE IF NOT EXISTS tune_merged_versions(
                    merged_version INTEGER PRIMARY KEY AUTOINCREMENT,
                    merge_id TEXT NOT NULL,
                    base_version INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    issuance_no TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tune_samples(
                    issuance_no TEXT NOT NULL
                        REFERENCES tune_issuances(issuance_no),
                    source_ref_kind TEXT NOT NULL,
                    source_ref_version TEXT,
                    group_name TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    row_id TEXT NOT NULL,
                    weight INTEGER NOT NULL,
                    body_sha256 TEXT NOT NULL,
                    hash_hex TEXT NOT NULL,
                    a TEXT NOT NULL,
                    log_screen REAL NOT NULL,
                    body TEXT NOT NULL,
                    PRIMARY KEY(issuance_no, group_name, rank)
                );
                COMMIT;
                """
            )
            self._initialised = True
        finally:
            conn.close()

    def _ensure(self):
        if not self._initialised:
            self.init()

    # -- master baseline reads (never written from here) ----------------
    def baseline_snapshot(self, base_version: int) -> Dict[str, dict]:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM versions WHERE version_id=?",
                (base_version,)).fetchone()
            if row is None:
                raise TuneStoreError(f"unknown baseline version: {base_version}")
            rows = conn.execute(
                "SELECT row_id, group_name, weight, body_sha256, rev "
                "FROM version_rows WHERE version_id=? ORDER BY row_id",
                (base_version,)).fetchall()
            return {r["row_id"]: {
                "row_id": r["row_id"], "group": r["group_name"],
                "weight": r["weight"], "body_sha256": r["body_sha256"],
                "rev": r["rev"], "active": True,
            } for r in rows}
        finally:
            conn.close()

    def master_latest_version(self) -> int:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MAX(version_id) AS v FROM versions").fetchone()
            return int(row["v"])
        finally:
            conn.close()

    def read_body(self, body_sha256: str) -> str:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT data FROM bodies WHERE sha256=?",
                (body_sha256,)).fetchone()
            if row is None:
                raise TuneStoreError(f"body missing from library: {body_sha256}")
            return bytes(row["data"]).decode("utf-8")
        finally:
            conn.close()

    # -- tickets ---------------------------------------------------------
    def register_ticket(self, payload: dict) -> Ticket:
        self._ensure()
        ticket = parse_ticket(payload)
        baseline = self.baseline_snapshot(ticket.base_version)
        validate_against_baseline(ticket, baseline)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Offline body edits join the shared append-only body library.
                for rec in ticket.records:
                    for chg in rec.changes:
                        if chg.field == tuning.BODY:
                            conn.execute(
                                "INSERT INTO bodies(sha256, data) VALUES (?,?) "
                                "ON CONFLICT(sha256) DO NOTHING",
                                (chg.body_sha256,
                                 chg.value.encode("utf-8")))
                conn.execute(
                    "INSERT INTO tune_tickets(ticket_id, branch_id, "
                    "base_version, payload_json, created_at) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(ticket_id) DO NOTHING",
                    (ticket.ticket_id, ticket.branch_id,
                     ticket.base_version,
                     canonical_json(ticket.payload), time.time()))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return ticket

    def _load_ticket(self, conn, ticket_id: str) -> Ticket:
        row = conn.execute(
            "SELECT payload_json FROM tune_tickets WHERE ticket_id=?",
            (ticket_id,)).fetchone()
        if row is None:
            raise TuneStoreError(f"unknown ticket: {ticket_id}")
        return parse_ticket(json.loads(row["payload_json"]))

    def list_tickets(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT ticket_id, branch_id, base_version, created_at "
                "FROM tune_tickets ORDER BY created_at")]
        finally:
            conn.close()

    # -- merges ----------------------------------------------------------
    def merge_tickets(self, left_ticket_id: str,
                      right_ticket_id: str) -> MergeResult:
        self._ensure()
        # Resolve and compute outside the write transaction: the merge itself
        # is pure, and a previously stored pair is returned unchanged.
        conn = self._connect()
        try:
            left = self._load_ticket(conn, left_ticket_id)
            right = self._load_ticket(conn, right_ticket_id)
            baseline = self.baseline_snapshot(left.base_version)
            result = merge_tickets(left, right, baseline)
            existing = conn.execute(
                "SELECT merge_id FROM tune_merges WHERE merge_id=?",
                (result.merge_id,)).fetchone()
        finally:
            conn.close()
        if existing is not None:
            return result

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Re-check under the write lock; identical content wins.
                double = conn.execute(
                    "SELECT 1 FROM tune_merges WHERE merge_id=?",
                    (result.merge_id,)).fetchone()
                if double is None:
                    conn.execute(
                        "INSERT INTO tune_merges(merge_id, base_version, "
                        "left_ticket_id, right_ticket_id, status, "
                        "content_sha256, result_json, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (result.merge_id, result.base_version,
                         result.left_ticket_id, result.right_ticket_id,
                         result.status, result.content_sha256,
                         canonical_json(result.as_dict()), time.time()))
                    # Persist the *full* effective view, not only the touched
                    # rows: a published merged version must enumerate every
                    # active row, and inactive rows keep the diff reviewable.
                    # A conflicted merge has no effective view; its row states
                    # (and the full conflict list with vouchers) still persist.
                    if result.ok:
                        full_view = effective_view(
                            result, list(baseline.values()))
                    else:
                        full_view = [{
                            "row_id": rid,
                            "group": mr.group, "weight": mr.weight,
                            "body_sha256": mr.body_sha256,
                            "rev": mr.base_rev, "active": mr.active,
                        } for rid, mr in sorted(result.rows.items())]
                    for r in full_view:
                        mr = result.rows.get(r["row_id"])
                        sources = mr.sources if mr is not None else {}
                        conn.execute(
                            "INSERT INTO tune_merge_rows(merge_id, row_id, "
                            "group_name, weight, body_sha256, rev, active, "
                            "changed, sources_json) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (result.merge_id, r["row_id"], r["group"],
                             r["weight"], r["body_sha256"], r["rev"],
                             1 if r["active"] else 0,
                             1 if (mr is not None and mr.changed) else 0,
                             canonical_json(sources)))
                conn.execute("COMMIT")
                return result
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def list_merges(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT merge_id, base_version, left_ticket_id, "
                "right_ticket_id, status, content_sha256, created_at "
                "FROM tune_merges ORDER BY created_at")]
        finally:
            conn.close()

    def get_merge(self, merge_id: str) -> dict:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT result_json FROM tune_merges WHERE merge_id=?",
                (merge_id,)).fetchone()
            if row is None:
                raise TuneStoreError(f"unknown merge: {merge_id}")
            return json.loads(row["result_json"])
        finally:
            conn.close()

    def merge_conflicts(self, merge_id: str) -> List[dict]:
        return self.get_merge(merge_id)["conflicts"]

    def merge_diff(self, merge_id: str) -> dict:
        return tuning.merge_diff(self._hydrate_merge(merge_id))

    def _hydrate_merge(self, merge_id: str) -> MergeResult:
        data = self.get_merge(merge_id)
        # Re-derive via the pure function so callers get typed objects.
        conn = self._connect()
        try:
            left = self._load_ticket(conn, data["left_ticket_id"])
            right = self._load_ticket(conn, data["right_ticket_id"])
        finally:
            conn.close()
        baseline = self.baseline_snapshot(data["base_version"])
        result = merge_tickets(left, right, baseline)
        if result.merge_id != merge_id:
            raise TuneStoreError("stored merge content does not recompute")
        return result

    def effective_rows(self, merge_id: str) -> List[dict]:
        """Full merged row set (inactive rows included for the diff)."""
        merge = self._hydrate_merge(merge_id)
        baseline_rows = list(self.baseline_snapshot(merge.base_version)
                             .values())
        return effective_view(merge, baseline_rows)

    def open_view(self, merge_id: str) -> MergedView:
        return MergedView(self, merge_id)

    def baseline_status(self, merge_id: str) -> dict:
        """Whether the master baseline has advanced past the merge base."""
        merge = self.get_merge(merge_id)
        latest = self.master_latest_version()
        base = int(merge["base_version"])
        return {"merge_id": merge_id, "base_version": base,
                "master_latest_version": latest,
                "baseline_current": base == latest,
                "advanced_by": max(0, latest - base)}

    # -- rehearsals ------------------------------------------------------
    def rehearse(self, merge_id: str, seed: bytes,
                 quotas: Mapping[str, int],
                 promises: Sequence[Promise]) -> Rehearsal:
        self._ensure()
        if len(seed) != 32:
            raise ValueError("seed must be exactly 32 bytes")
        merge = self._hydrate_merge(merge_id)
        baseline_rows = list(self.baseline_snapshot(merge.base_version)
                             .values())
        rehearsal = run_rehearsal(merge, baseline_rows, seed, quotas,
                                  list(promises))
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO tune_rehearsals(rehearsal_id, merge_id, "
                    "base_version, merge_content_sha256, seed_hex, "
                    "quotas_json, promises_json, promise_ok, "
                    "violations_json, manifest_json, selected_json, "
                    "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(rehearsal_id) DO NOTHING",
                    (rehearsal.rehearsal_id, merge_id,
                     rehearsal.base_version,
                     rehearsal.merge_content_sha256, rehearsal.seed_hex,
                     canonical_json(rehearsal.quotas),
                     canonical_json([{"group": p.group, "keep": p.keep,
                                      "forbid": p.forbid}
                                     for p in rehearsal.promises]),
                     1 if rehearsal.promise_ok else 0,
                     canonical_json(rehearsal.violations),
                     canonical_json(rehearsal.manifest_groups),
                     canonical_json(rehearsal.selected), time.time()))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return rehearsal

    def list_rehearsals(self, merge_id: Optional[str] = None) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            if merge_id is None:
                rows = conn.execute(
                    "SELECT rehearsal_id, merge_id, base_version, "
                    "merge_content_sha256, seed_hex, promise_ok, created_at "
                    "FROM tune_rehearsals ORDER BY created_at").fetchall()
            else:
                rows = conn.execute(
                    "SELECT rehearsal_id, merge_id, base_version, "
                    "merge_content_sha256, seed_hex, promise_ok, created_at "
                    "FROM tune_rehearsals WHERE merge_id=? ORDER BY created_at",
                    (merge_id,)).fetchall()
            out = [dict(r) for r in rows]
            for r in out:
                r["promise_ok"] = bool(r["promise_ok"])
            return out
        finally:
            conn.close()

    def get_rehearsal(self, rehearsal_id: str) -> dict:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM tune_rehearsals WHERE rehearsal_id=?",
                (rehearsal_id,)).fetchone()
            if row is None:
                raise TuneStoreError(f"unknown rehearsal: {rehearsal_id}")
            d = dict(row)
            for key in ("quotas_json", "promises_json", "violations_json",
                        "manifest_json", "selected_json"):
                d[key.removesuffix("_json")] = json.loads(d.pop(key))
            d["promise_ok"] = bool(d["promise_ok"])
            return d
        finally:
            conn.close()

    # -- issuance: request, crash-safe pending state, paired decision ----
    def request_issuance(self, issuance_no: str, rehearsal_id: str,
                         note: Optional[str] = None) -> dict:
        """Register an issuance request and stage its paired sample.

        Idempotent on ``issuance_no`` while the requested content is identical;
        a different merge/rehearsal under the same number is rejected even if
        the master baseline has advanced.
        """
        self._ensure()
        if not isinstance(issuance_no, str) or not issuance_no:
            raise IssuanceError("issuance_no must be a non-empty string")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                reh = self._get_rehearsal_row(conn, rehearsal_id)
                merge_id = reh["merge_id"]
                merge_sha = reh["merge_content_sha256"]
                existing = conn.execute(
                    "SELECT * FROM tune_issuances WHERE issuance_no=?",
                    (issuance_no,)).fetchone()
                if existing is not None:
                    if (existing["merge_content_sha256"] != merge_sha
                            or existing["rehearsal_id"] != rehearsal_id):
                        raise IssuanceError(
                            f"issuance_no {issuance_no!r} already binds a "
                            "different merge/rehearsal; one number can never "
                            "carry two merge results")
                    conn.execute("ROLLBACK")
                    return self.get_issuance(issuance_no)
                if not reh["promise_ok"]:
                    raise IssuanceError(
                        f"rehearsal {rehearsal_id[:12]} violates its promise; "
                        "only a satisfiable rehearsal can be requested")
                records = self._sample_records(conn, merge_id, reh)
                records_sha = content_hash([
                    {k: v for k, v in r.items()
                     if k not in ("source_ref_kind", "source_ref_version")}
                    for r in records])
                conn.execute(
                    "INSERT INTO tune_issuances(issuance_no, merge_id, "
                    "rehearsal_id, merge_content_sha256, state, "
                    "merged_version, seed_hex, sample_records_sha256, note, "
                    "created_at, decided_at) VALUES (?,?,?,?, 'pending', "
                    "NULL, ?, ?, ?, ?, NULL)",
                    (issuance_no, merge_id, rehearsal_id, merge_sha,
                     reh["seed_hex"], records_sha, note, time.time()))
                self._insert_sample_rows(conn, issuance_no, records,
                                         "preview", None)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return self.get_issuance(issuance_no)

    def decide_issuance(self, issuance_no: str, approve: bool,
                        note: Optional[str] = None) -> dict:
        """Approve (publish pair atomically) or reject (keep preview only)."""
        self._ensure()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                iss = conn.execute(
                    "SELECT * FROM tune_issuances WHERE issuance_no=?",
                    (issuance_no,)).fetchone()
                if iss is None:
                    raise IssuanceError(
                        f"unknown issuance_no: {issuance_no!r}")
                if iss["state"] != "pending":
                    raise IssuanceError(
                        f"issuance {issuance_no!r} already {iss['state']}; "
                        "decisions are one-shot")
                if approve:
                    count = conn.execute(
                        "SELECT COUNT(*) AS n FROM tune_merge_rows "
                        "WHERE merge_id=? AND active=1",
                        (iss["merge_id"],)).fetchone()["n"]
                    cur = conn.execute(
                        "INSERT INTO tune_merged_versions(merge_id, "
                        "base_version, content_sha256, issuance_no, row_count, "
                        "created_at) VALUES (?,?,?,?,?,?)",
                        (iss["merge_id"],
                         self._merge_base_version(conn, iss["merge_id"]),
                         iss["merge_content_sha256"], issuance_no, count,
                         time.time()))
                    # The ``merged-`` prefix keeps this namespace disjoint
                    # from integer master version ids.
                    merged_ref = f"merged-{int(cur.lastrowid)}"
                    conn.execute(
                        "UPDATE tune_samples SET source_ref_kind='merged', "
                        "source_ref_version=? WHERE issuance_no=?",
                        (merged_ref, issuance_no))
                    conn.execute(
                        "UPDATE tune_issuances SET state='approved', "
                        "merged_version=?, decided_at=?, "
                        "note=COALESCE(?, note) WHERE issuance_no=?",
                        (merged_ref, time.time(), note, issuance_no))
                else:
                    conn.execute(
                        "UPDATE tune_issuances SET state='rejected', "
                        "decided_at=?, note=COALESCE(?, note) "
                        "WHERE issuance_no=?",
                        (time.time(), note, issuance_no))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return self.get_issuance(issuance_no)

    @staticmethod
    def _merge_base_version(conn, merge_id: str) -> int:
        return int(conn.execute(
            "SELECT base_version FROM tune_merges WHERE merge_id=?",
            (merge_id,)).fetchone()["base_version"])

    @staticmethod
    def _get_rehearsal_row(conn, rehearsal_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM tune_rehearsals WHERE rehearsal_id=?",
            (rehearsal_id,)).fetchone()
        if row is None:
            raise TuneStoreError(f"unknown rehearsal: {rehearsal_id}")
        return row

    def _sample_records(self, conn, merge_id: str,
                        reh: sqlite3.Row) -> List[dict]:
        """Materialise the exact selected rows with real bodies."""
        manifest = json.loads(reh["manifest_json"])
        seed = bytes.fromhex(reh["seed_hex"])
        rows = {r["row_id"]: r for r in
                conn.execute(
                    "SELECT row_id, group_name, weight, body_sha256 FROM "
                    "tune_merge_rows WHERE merge_id=? AND active=1",
                    (merge_id,)).fetchall()}
        bodies: Dict[str, str] = {}
        records: List[dict] = []
        for group in sorted(manifest):
            for sel in manifest[group]["selected"]:
                rid = sel["row_id"]
                mr = rows.get(rid)
                if mr is None or mr["group_name"] != group:
                    raise TuneStoreError(
                        f"rehearsal row {rid!r} missing from merged view")
                digest = mr["body_sha256"]
                if digest not in bodies:
                    bodies[digest] = self.read_body(digest)
                a = lottery_a(seed, rid)
                records.append({
                    "group": group, "rank": sel["rank"], "row_id": rid,
                    "weight": sel["weight"], "hash_hex": sel["hash_hex"],
                    "a": sel["a"], "log_screen": sel["log_screen"],
                    "body_sha256": digest,
                    "body_bytes": len(bodies[digest].encode("utf-8")),
                    "seed_hex": reh["seed_hex"], "body": bodies[digest],
                    "_a_check": str(a),
                })
        for r in records:
            if r.pop("_a_check") != r["a"]:
                raise TuneStoreError(f"lottery key mismatch for {r['row_id']!r}")
        return records

    @staticmethod
    def _insert_sample_rows(conn, issuance_no: str, records: List[dict],
                            kind: str, ref_version: Optional[str]) -> None:
        for r in records:
            conn.execute(
                "INSERT INTO tune_samples(issuance_no, source_ref_kind, "
                "source_ref_version, group_name, rank, row_id, weight, "
                "body_sha256, hash_hex, a, log_screen, body) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (issuance_no, kind, ref_version, r["group"], r["rank"],
                 r["row_id"], r["weight"], r["body_sha256"], r["hash_hex"],
                 r["a"], r["log_screen"], r["body"]))

    # -- queries: old product vs offline previews stay independent -------
    def list_issuances(self, state: Optional[str] = None) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            sql = ("SELECT issuance_no, merge_id, rehearsal_id, "
                   "merge_content_sha256, state, merged_version, seed_hex, "
                   "sample_records_sha256, note, created_at, decided_at "
                   "FROM tune_issuances")
            params: List[Any] = []
            if state is not None:
                sql += " WHERE state=?"
                params.append(state)
            sql += " ORDER BY created_at"
            return [dict(r) for r in conn.execute(sql, params)]
        finally:
            conn.close()

    def get_issuance(self, issuance_no: str) -> dict:
        self._ensure()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM tune_issuances WHERE issuance_no=?",
                (issuance_no,)).fetchone()
            if row is None:
                raise IssuanceError(f"unknown issuance_no: {issuance_no!r}")
            return dict(row)
        finally:
            conn.close()

    def get_sample_records(self, issuance_no: str) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_ref_kind, source_ref_version, group_name, "
                "rank, row_id, weight, body_sha256, hash_hex, a, log_screen, "
                "body FROM tune_samples WHERE issuance_no=? "
                "ORDER BY group_name, rank", (issuance_no,)).fetchall()
            if not rows:
                iss = self.get_issuance(issuance_no)
                if iss is None:
                    raise IssuanceError(
                        f"unknown issuance_no: {issuance_no!r}")
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_merged_versions(self) -> List[dict]:
        self._ensure()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT merged_version, merge_id, base_version, "
                "content_sha256, issuance_no, row_count, created_at "
                "FROM tune_merged_versions ORDER BY merged_version").fetchall()
            out = [dict(r) for r in rows]
            for r in out:
                r["merged_ref"] = f"merged-{r['merged_version']}"
            return out
        finally:
            conn.close()

    def get_merged_version(self, merged_ref) -> dict:
        self._ensure()
        if isinstance(merged_ref, str):
            if not merged_ref.startswith("merged-"):
                raise TuneStoreError(
                    f"merged version ref must look like merged-N, got "
                    f"{merged_ref!r}")
            seq = int(merged_ref.removeprefix("merged-"))
        else:
            seq = int(merged_ref)
        conn = self._connect()
        try:
            head = conn.execute(
                "SELECT * FROM tune_merged_versions WHERE merged_version=?",
                (seq,)).fetchone()
            if head is None:
                raise TuneStoreError(
                    f"unknown merged version: merged-{seq}")
            rows = conn.execute(
                "SELECT row_id, group_name AS \"group\", weight, "
                "body_sha256, rev, active FROM tune_merge_rows "
                "WHERE merge_id=? ORDER BY row_id",
                (head["merge_id"],)).fetchall()
            out = dict(head)
            out["merged_ref"] = f"merged-{seq}"
            out["rows"] = [dict(r, active=bool(r["active"])) for r in rows]
            return out
        finally:
            conn.close()


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_quotas(items: Sequence[str], quota_file: Optional[str]
                  ) -> Dict[str, int]:
    quotas: Dict[str, int] = {}
    if quota_file:
        quotas.update({k: int(v) for k, v in
                       _load_json(quota_file).items()})
    for item in items or []:
        group, _, value = item.partition("=")
        if not group or not value:
            raise SystemExit(f"bad --quota {item!r}; expected GROUP=N")
        quotas[group] = int(value)
    return quotas


def _parse_seed(text: str) -> bytes:
    seed = bytes.fromhex(text)
    if len(seed) != 32:
        raise SystemExit("seed must be 32 bytes (64 hex characters)")
    return seed


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m wrsource.tunestore",
        description="Offline tuning tickets, two-branch merges and "
                    "promise-bound sample issuance.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ticket", help="register one branch ticket")
    p.add_argument("--db", required=True)
    p.add_argument("--ticket-file", required=True)

    p = sub.add_parser("tickets", help="list registered tickets")
    p.add_argument("--db", required=True)

    p = sub.add_parser("merge", help="merge two tickets")
    p.add_argument("--db", required=True)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)

    p = sub.add_parser("merges", help="list merges")
    p.add_argument("--db", required=True)

    p = sub.add_parser("merge-info", help="full merge result incl. vouchers")
    p.add_argument("--db", required=True)
    p.add_argument("--merge", required=True)

    p = sub.add_parser("diff", help="merged-vs-baseline structural diff")
    p.add_argument("--db", required=True)
    p.add_argument("--merge", required=True)

    p = sub.add_parser("baseline-status",
                       help="whether master advanced past the merge base")
    p.add_argument("--db", required=True)
    p.add_argument("--merge", required=True)

    p = sub.add_parser("rehearse", help="exact lottery + promise check")
    p.add_argument("--db", required=True)
    p.add_argument("--merge", required=True)
    p.add_argument("--seed-hex", required=True)
    p.add_argument("--quota", action="append")
    p.add_argument("--quota-file")
    p.add_argument("--promise-file",
                   help="JSON list of {group, keep, forbid}")

    p = sub.add_parser("rehearsals", help="list rehearsals")
    p.add_argument("--db", required=True)
    p.add_argument("--merge")

    p = sub.add_parser("rehearsal", help="one stored rehearsal")
    p.add_argument("--db", required=True)
    p.add_argument("--rehearsal", required=True)

    p = sub.add_parser("request", help="stage an issuance (pending)")
    p.add_argument("--db", required=True)
    p.add_argument("--no", required=True)
    p.add_argument("--rehearsal", required=True)
    p.add_argument("--note")

    p = sub.add_parser("decide", help="approve or reject a pending issuance")
    p.add_argument("--db", required=True)
    p.add_argument("--no", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    p.add_argument("--note")

    p = sub.add_parser("issuances", help="list issuance records")
    p.add_argument("--db", required=True)
    p.add_argument("--state", choices=("pending", "approved", "rejected"))

    p = sub.add_parser("issuance", help="one issuance record")
    p.add_argument("--db", required=True)
    p.add_argument("--no", required=True)

    p = sub.add_parser("sample", help="paired sample rows of one issuance")
    p.add_argument("--db", required=True)
    p.add_argument("--no", required=True)

    p = sub.add_parser("merged-versions", help="published merged versions")
    p.add_argument("--db", required=True)

    p = sub.add_parser("merged-version", help="one published merged version")
    p.add_argument("--db", required=True)
    p.add_argument("--merged-version", required=True,
                   help="merged-N reference from merged-versions")

    args = parser.parse_args(argv)
    store = TuneStore(args.db)
    store.init()

    if args.cmd == "ticket":
        ticket = store.register_ticket(_load_json(args.ticket_file))
        _print({"ticket_id": ticket.ticket_id,
                "branch_id": ticket.branch_id,
                "base_version": ticket.base_version,
                "records": len(ticket.records)})
    elif args.cmd == "tickets":
        _print(store.list_tickets())
    elif args.cmd == "merge":
        result = store.merge_tickets(args.left, args.right)
        _print({"merge_id": result.merge_id, "status": result.status,
                "conflicts": [c.as_dict() for c in result.conflicts]})
    elif args.cmd == "merges":
        _print(store.list_merges())
    elif args.cmd == "merge-info":
        _print(store.get_merge(args.merge))
    elif args.cmd == "diff":
        _print(store.merge_diff(args.merge))
    elif args.cmd == "baseline-status":
        _print(store.baseline_status(args.merge))
    elif args.cmd == "rehearse":
        promises = [parse_promise(p) for p in
                    (_load_json(args.promise_file)
                     if args.promise_file else [])]
        reh = store.rehearse(args.merge, _parse_seed(args.seed_hex),
                             _parse_quotas(args.quota, args.quota_file),
                             promises)
        _print({"rehearsal_id": reh.rehearsal_id,
                "merge_id": reh.merge_id,
                "merge_content_sha256": reh.merge_content_sha256,
                "selected": reh.selected,
                "promise_ok": reh.promise_ok,
                "violations": reh.violations})
    elif args.cmd == "rehearsals":
        _print(store.list_rehearsals(args.merge))
    elif args.cmd == "rehearsal":
        _print(store.get_rehearsal(args.rehearsal))
    elif args.cmd == "request":
        _print(store.request_issuance(args.no, args.rehearsal, args.note))
    elif args.cmd == "decide":
        _print(store.decide_issuance(args.no, args.approve, args.note))
    elif args.cmd == "issuances":
        _print(store.list_issuances(args.state))
    elif args.cmd == "issuance":
        _print(store.get_issuance(args.no))
    elif args.cmd == "sample":
        _print(store.get_sample_records(args.no))
    elif args.cmd == "merged-versions":
        _print(store.list_merged_versions())
    elif args.cmd == "merged-version":
        _print(store.get_merged_version(args.merged_version))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
