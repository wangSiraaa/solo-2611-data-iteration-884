"""Offline two-branch tuning-ticket merge, preview rehearsal and promises.

This module is pure domain logic: no SQLite, no filesystem.  The store layer
(:mod:`wrsource.tunestore`) supplies baseline snapshots and persists results.

Vocabulary
----------

* A **ticket** is what one editing point brings back: it carries *diffs only*,
  never a whole database.  Every ticket is bound to one shared baseline
  source version and each row record names the original row and the original
  row revision it was edited from.
* Two tickets (left/right) are merged independently of the master source.
  Each branch may touch a field at most once per row.
* A **merge** is content-addressed.  The same two tickets always produce the
  same ``merge_id``, so a merge that passed can be trial-computed against many
  promises and every rehearsal can name the exact merge version it reused.
* A **rehearsal** runs the unchanged exact lottery (see :mod:`wrsource.sampling`)
  on the *full* active groups of the merged view.  A promise only *checks* the
  deterministic result; it can never substitute a row.

Merge rules (fixed by the task)
-------------------------------

Per row, per field:

* disjoint fields combine;
* two parallel weight adds add their deltas;
* two identical absolute weight sets combine;
* a set meeting an add is a **conflict even when both branches alone reach the
  same number**;
* a delete combines with an untouched row, two parallel deletes combine, and a
  delete meeting a real edit on the other side is a listed conflict;
* the merged weight must stay inside the old ``1..100`` range.

Rows are compared only when their ``row_id`` matches.  Every conflict is
collected and carries both sides' vouchers (the original op, ticket id and
baseline reference).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .dataset import build_group_result
from .sampling import make_candidate
from .source import MAX_WEIGHT, MIN_WEIGHT

_JSON_SEP = (",", ":")

WEIGHT = "weight"
GROUP = "group"
BODY = "body"
ACTIVE = "active"

FIELD_KINDS = {WEIGHT, GROUP, BODY, ACTIVE}


class TuningError(Exception):
    pass


class TicketError(TuningError):
    pass


class PromiseError(TuningError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=_JSON_SEP,
                      ensure_ascii=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tickets (diffs carried by one editing point)
# ---------------------------------------------------------------------------

@dataclass
class Change:
    """One field-level op inside a row record."""

    field: str
    op: str
    value: Any = None
    body_sha256: Optional[str] = None

    def as_dict(self) -> dict:
        out = {"field": self.field, "op": self.op}
        if self.field == WEIGHT and self.op == "add":
            out["delta"] = self.value
        elif self.field == BODY:
            # Tickets carry the diff payload itself; the digest is derivable.
            out["body"] = self.value
            out["body_sha256"] = self.body_sha256
        else:
            out["value"] = self.value
        return out


@dataclass
class RowRecord:
    """One branch's offline edit of one row, with its baseline reference."""

    row_id: str
    base_rev: int
    base: Dict[str, Any]          # group, weight, body_sha256, active
    changes: List[Change]

    def change_map(self) -> Dict[str, Change]:
        return {c.field: c for c in self.changes}


@dataclass
class Ticket:
    ticket_id: str
    branch_id: str
    base_version: int
    records: List[RowRecord]
    payload: dict


def _parse_change(raw: dict) -> Change:
    if not isinstance(raw, dict):
        raise TicketError("each change must be an object")
    fld = raw.get("field")
    if fld not in FIELD_KINDS:
        raise TicketError(f"unknown change field: {fld!r}")
    if fld == WEIGHT:
        op = raw.get("op")
        if op == "set":
            n = raw.get("value")
            if isinstance(n, bool) or not isinstance(n, int):
                raise TicketError("weight set value must be an integer")
            if not MIN_WEIGHT <= n <= MAX_WEIGHT:
                raise TicketError(
                    f"weight set value must be in {MIN_WEIGHT}..{MAX_WEIGHT}")
            return Change(WEIGHT, "set", n)
        if op == "add":
            d = raw.get("delta")
            if isinstance(d, bool) or not isinstance(d, int) or d == 0:
                raise TicketError("weight add delta must be a nonzero integer")
            return Change(WEIGHT, "add", d)
        raise TicketError(f"weight op must be 'set' or 'add', got {op!r}")
    if fld == GROUP:
        if raw.get("op") != "set":
            raise TicketError("group change must be op=set")
        g = raw.get("value")
        if not isinstance(g, str) or not g:
            raise TicketError("group value must be a non-empty string")
        return Change(GROUP, "set", g)
    if fld == BODY:
        if raw.get("op") != "set":
            raise TicketError("body change must be op=set")
        body = raw.get("body")
        if not isinstance(body, str):
            raise TicketError("body must be a UTF-8 string")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return Change(BODY, "set", body, digest)
    # active
    if raw.get("op") != "retract":
        raise TicketError("active change must be op=retract")
    return Change(ACTIVE, "retract")


def parse_ticket(payload: dict) -> Ticket:
    """Validate shape and normalise one branch ticket (diffs only)."""
    if not isinstance(payload, dict):
        raise TicketError("ticket must be an object")
    branch_id = payload.get("branch_id")
    if not isinstance(branch_id, str) or not branch_id:
        raise TicketError("branch_id must be a non-empty string")
    base_version = payload.get("base_version")
    if isinstance(base_version, bool) or not isinstance(base_version, int) \
            or base_version < 0:
        raise TicketError("base_version must be a non-negative integer")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise TicketError("ticket must carry at least one row record")
    records: List[RowRecord] = []
    seen_rows = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise TicketError("each record must be an object")
        rid = raw.get("row_id")
        if not isinstance(rid, str) or not rid:
            raise TicketError("record row_id must be a non-empty string")
        if rid in seen_rows:
            raise TicketError(f"duplicate record for {rid!r} in one ticket")
        seen_rows.add(rid)
        base_rev = raw.get("base_rev")
        if isinstance(base_rev, bool) or not isinstance(base_rev, int) \
                or base_rev <= 0:
            raise TicketError(f"base_rev for {rid!r} must be a positive int")
        base = raw.get("base")
        if not isinstance(base, dict):
            raise TicketError(f"base snapshot missing for {rid!r}")
        for key in ("group", "weight", "body_sha256", "active"):
            if key not in base:
                raise TicketError(f"base.{key} missing for {rid!r}")
        if not isinstance(base["group"], str) or not base["group"]:
            raise TicketError(f"base.group invalid for {rid!r}")
        if isinstance(base["weight"], bool) or not isinstance(
                base["weight"], int) or not MIN_WEIGHT <= base["weight"] \
                <= MAX_WEIGHT:
            raise TicketError(f"base.weight invalid for {rid!r}")
        if not isinstance(base["body_sha256"], str) or \
                len(base["body_sha256"]) != 64:
            raise TicketError(f"base.body_sha256 invalid for {rid!r}")
        if not isinstance(base["active"], bool):
            raise TicketError(f"base.active invalid for {rid!r}")
        raw_changes = raw.get("changes")
        if not isinstance(raw_changes, list) or not raw_changes:
            raise TicketError(f"record for {rid!r} carries no changes")
        changes = [_parse_change(c) for c in raw_changes]
        fields = [c.field for c in changes]
        if len(set(fields)) != len(fields):
            raise TicketError(
                f"branch {branch_id!r} touches a field more than once on "
                f"{rid!r}: {fields}")
        if ACTIVE in fields and len(changes) > 1:
            raise TicketError(
                f"retract cannot be combined with other edits for {rid!r}")
        records.append(RowRecord(rid, base_rev, dict(base), changes))
    identity = {
        "branch_id": branch_id,
        "base_version": base_version,
        "records": [
            {
                "row_id": r.row_id,
                "base_rev": r.base_rev,
                "base": r.base,
                "changes": [c.as_dict() for c in r.changes],
            } for r in records
        ],
    }
    return Ticket(content_hash(identity), branch_id, base_version, records,
                  identity)


def validate_against_baseline(ticket: Ticket,
                              baseline: Mapping[str, dict]) -> None:
    """Verify every record against the frozen baseline snapshot."""
    for rec in ticket.records:
        snap = baseline.get(rec.row_id)
        if snap is None:
            raise TicketError(
                f"ticket {ticket.ticket_id[:12]} row {rec.row_id!r} is not in "
                f"baseline version {ticket.base_version}")
        if snap["rev"] != rec.base_rev:
            raise TicketError(
                f"stale revision for {rec.row_id!r}: ticket edited rev "
                f"{rec.base_rev}, baseline now has rev {snap['rev']}")
        for key in ("group", "weight", "body_sha256", "active"):
            if snap[key] != rec.base[key]:
                raise TicketError(
                    f"baseline mismatch for {rec.row_id!r}.{key}: ticket says "
                    f"{rec.base[key]!r}, snapshot says {snap[key]!r}")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass
class MergeConflict:
    row_id: str
    field: str
    reason: str
    left: Optional[dict]
    right: Optional[dict]

    def as_dict(self) -> dict:
        return {"row_id": self.row_id, "field": self.field,
                "reason": self.reason, "left": self.left,
                "right": self.right}


@dataclass
class MergedRow:
    row_id: str
    base_rev: int
    active: bool
    group: str
    weight: int
    body_sha256: str
    changed: bool
    sources: Dict[str, List[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"row_id": self.row_id, "base_rev": self.base_rev,
                "active": self.active, "group": self.group,
                "weight": self.weight, "body_sha256": self.body_sha256,
                "changed": self.changed, "sources": self.sources}


@dataclass
class MergeResult:
    merge_id: str
    base_version: int
    left_ticket_id: str
    right_ticket_id: str
    status: str                       # "ok" | "conflicted"
    conflicts: List[MergeConflict]
    rows: Dict[str, MergedRow]
    content_sha256: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict:
        return {
            "merge_id": self.merge_id,
            "base_version": self.base_version,
            "left_ticket_id": self.left_ticket_id,
            "right_ticket_id": self.right_ticket_id,
            "status": self.status,
            "conflicts": [c.as_dict() for c in self.conflicts],
            "rows": {rid: row.as_dict() for rid, row in
                     sorted(self.rows.items())},
            "content_sha256": self.content_sha256,
        }


def _voucher(ticket: Optional[Ticket], rec: Optional[RowRecord],
             chg: Optional[Change]) -> Optional[dict]:
    if ticket is None or rec is None:
        return None
    out = {
        "ticket_id": ticket.ticket_id,
        "branch_id": ticket.branch_id,
        "row_id": rec.row_id,
        "base_rev": rec.base_rev,
        "base": rec.base,
    }
    if chg is not None:
        out["change"] = chg.as_dict()
    if len(rec.changes) > 1:
        out["all_changes"] = [c.as_dict() for c in rec.changes]
    return out


def _side(ticket: Ticket, row_id: str) -> Tuple[Optional[RowRecord],
                                                Optional[Change],
                                                Dict[str, Change]]:
    rec = next((r for r in ticket.records if r.row_id == row_id), None)
    if rec is None:
        return None, None, {}
    cmap = rec.change_map()
    return rec, None, cmap


def merge_tickets(left: Ticket, right: Ticket,
                  baseline: Mapping[str, dict]) -> MergeResult:
    """Merge two diff-only tickets over a shared baseline snapshot.

    Pure and total: malformed tickets raise :class:`TicketError`; semantic
    incompatibilities are returned as conflicts, never raised, so the full
    conflict list (with both vouchers) is always available.
    """
    if left.base_version != right.base_version:
        raise TicketError(
            f"baseline versions differ: {left.base_version} vs "
            f"{right.base_version}")
    # Normalise left/right by ticket identity so merge(a, b) and merge(b, a)
    # hash identically; vouchers still name each branch explicitly.
    if left.ticket_id > right.ticket_id:
        left, right = right, left
    validate_against_baseline(left, baseline)
    validate_against_baseline(right, baseline)
    base_version = left.base_version

    row_ids = sorted({r.row_id for r in left.records}
                     | {r.row_id for r in right.records})
    conflicts: List[MergeConflict] = []
    rows: Dict[str, MergedRow] = {}

    for rid in row_ids:
        lrec = next((r for r in left.records if r.row_id == rid), None)
        rrec = next((r for r in right.records if r.row_id == rid), None)
        lmap = lrec.change_map() if lrec else {}
        rmap = rrec.change_map() if rrec else {}
        snap = baseline[rid]
        sources: Dict[str, List[str]] = {}
        row_conflicts: List[MergeConflict] = []

        # -- deletion ----------------------------------------------------
        l_del = ACTIVE in lmap
        r_del = ACTIVE in rmap
        deleted = False
        if l_del or r_del:
            # A parallel retract is not "another change"; only real edits on
            # the non-deleting side conflict with the deletion.
            other_map = rmap if l_del else lmap
            other_changes = {f: c for f, c in other_map.items()
                             if f != ACTIVE}
            if other_changes:
                row_conflicts.append(MergeConflict(
                    rid, ACTIVE, "retract_vs_change",
                    _voucher(left, lrec, lmap.get(ACTIVE)
                             if l_del else next(iter(lmap.values()))),
                    _voucher(right, rrec, rmap.get(ACTIVE)
                             if r_del else next(iter(rmap.values())))))
            else:
                deleted = True
                sources[ACTIVE] = sorted(
                    {b for b, d in ((left.branch_id, l_del),
                                    (right.branch_id, r_del)) if d})

        # -- weight ------------------------------------------------------
        lw, rw = lmap.get(WEIGHT), rmap.get(WEIGHT)
        weight = snap["weight"]
        if lw is not None or rw is not None:
            if lw is None:
                weight, src = _apply_one(snap["weight"], rw), [right.branch_id]
            elif rw is None:
                weight, src = _apply_one(snap["weight"], lw), [left.branch_id]
            elif lw.op == "set" and rw.op == "set":
                if lw.value == rw.value:
                    weight, src = lw.value, sorted(
                        {left.branch_id, right.branch_id})
                else:
                    row_conflicts.append(_weight_conflict(
                        rid, snap, left, lrec, lw, right, rrec, rw,
                        "distinct_sets"))
                    src = []
            elif lw.op == "add" and rw.op == "add":
                weight = snap["weight"] + lw.value + rw.value
                src = sorted({left.branch_id, right.branch_id})
            else:
                # set meets add in either order: conflict even if the two
                # branches alone would reach the same number.
                row_conflicts.append(_weight_conflict(
                    rid, snap, left, lrec, lw, right, rrec, rw,
                    "set_vs_add"))
                src = []
            if not row_conflicts or not any(
                    c.field == WEIGHT for c in row_conflicts):
                if not MIN_WEIGHT <= weight <= MAX_WEIGHT:
                    row_conflicts.append(MergeConflict(
                        rid, WEIGHT, "weight_out_of_range",
                        _voucher(left, lrec, lw),
                        _voucher(right, rrec, rw)))
                else:
                    sources[WEIGHT] = src

        # -- group -------------------------------------------------------
        lg, rg = lmap.get(GROUP), rmap.get(GROUP)
        group = snap["group"]
        if lg is not None or rg is not None:
            chosen, src = _merge_scalar(
                rid, GROUP, snap["group"],
                (lg.value if lg else None, left.branch_id, lrec, lg),
                (rg.value if rg else None, right.branch_id, rrec, rg),
                row_conflicts)
            group = chosen
            if src:
                sources[GROUP] = src

        # -- body --------------------------------------------------------
        lb, rb = lmap.get(BODY), rmap.get(BODY)
        body_sha = snap["body_sha256"]
        if lb is not None or rb is not None:
            chosen, src = _merge_body(
                rid, snap, left, lrec, lb, right, rrec, rb, row_conflicts)
            body_sha = chosen
            if src:
                sources[BODY] = src

        if row_conflicts:
            conflicts.extend(row_conflicts)
            # Keep a best-effort row state for diffing, but mark the merge
            # conflicted; the effective view is never published.
        changed = (deleted != (not snap["active"])
                   or (not deleted and (
                       weight != snap["weight"] or group != snap["group"]
                       or body_sha != snap["body_sha256"])))
        rows[rid] = MergedRow(
            rid, snap["rev"], not deleted, group, weight, body_sha, changed,
            sources)

    status = "conflicted" if conflicts else "ok"
    identity = {
        "base_version": base_version,
        "left_ticket_id": left.ticket_id,
        "right_ticket_id": right.ticket_id,
        "rows": [r.as_dict() for _, r in sorted(rows.items())],
        "conflicts": [c.as_dict() for c in conflicts],
    }
    digest = content_hash(identity)
    return MergeResult(digest, base_version, left.ticket_id,
                       right.ticket_id, status, conflicts, rows, digest)


def _apply_one(base_weight: int, chg: Change) -> int:
    if chg.op == "set":
        return chg.value
    return base_weight + chg.value


def _weight_conflict(rid, snap, left, lrec, lw, right, rrec, rw,
                     reason) -> MergeConflict:
    left_end = _apply_one(snap["weight"], lw) if lw else None
    right_end = _apply_one(snap["weight"], rw) if rw else None
    c = MergeConflict(rid, WEIGHT, reason,
                      _voucher(left, lrec, lw),
                      _voucher(right, rrec, rw))
    # Record the per-branch end states so an equal-number set/add conflict is
    # visibly a rule conflict rather than a numeric one.
    c.left["branch_end_weight"] = left_end
    c.right["branch_end_weight"] = right_end
    c.left["baseline_weight"] = snap["weight"]
    c.right["baseline_weight"] = snap["weight"]
    return c


def _merge_scalar(rid, field, base_value, left_side, right_side,
                  conflicts) -> Tuple[Any, List[str]]:
    lval, lbranch, lrec, lchg = left_side
    rval, rbranch, rrec, rchg = right_side
    if lval is None:
        return rval, [rbranch]
    if rval is None:
        return lval, [lbranch]
    if lval == rval:
        return lval, sorted({lbranch, rbranch})
    conflicts.append(MergeConflict(
        rid, field, f"distinct_{field}_values",
        _voucher_ticket(left_side), _voucher_ticket(right_side)))
    return base_value, []


def _voucher_ticket(side) -> Optional[dict]:
    val, branch, rec, chg = side
    if rec is None:
        return None
    return {"branch_id": branch, "row_id": rec.row_id,
            "base_rev": rec.base_rev, "base": rec.base,
            "change": chg.as_dict() if chg else None}


def _merge_body(rid, snap, left, lrec, lb, right, rrec, rb,
                conflicts) -> Tuple[str, List[str]]:
    if lb is None:
        return rb.body_sha256, [right.branch_id]
    if rb is None:
        return lb.body_sha256, [left.branch_id]
    if lb.body_sha256 == rb.body_sha256:
        return lb.body_sha256, sorted({left.branch_id, right.branch_id})
    conflicts.append(MergeConflict(
        rid, BODY, "distinct_body_values",
        _voucher(left, lrec, lb), _voucher(right, rrec, rb)))
    return snap["body_sha256"], []


# ---------------------------------------------------------------------------
# Effective merged view (full rows, baseline + merged edits)
# ---------------------------------------------------------------------------

def effective_view(merge: MergeResult,
                   baseline_rows: Sequence[dict]) -> List[dict]:
    """Apply a successful merge to the full baseline snapshot.

    Returns every row of the merged source: untouched baseline rows plus the
    merged rows, each with ``row_id, group, weight, body_sha256, rev,
    active``.  Deleted rows are returned with ``active=False`` so the diff is
    fully reviewable.
    """
    if not merge.ok:
        raise TuningError(
            f"merge {merge.merge_id[:12]} is conflicted; no effective view")
    out: List[dict] = []
    for snap in baseline_rows:
        rid = snap["row_id"]
        mr = merge.rows.get(rid)
        if mr is None:
            out.append({"row_id": rid, "group": snap["group"],
                        "weight": snap["weight"],
                        "body_sha256": snap["body_sha256"],
                        "rev": snap["rev"], "active": snap["active"]})
        else:
            out.append({"row_id": rid, "group": mr.group,
                        "weight": mr.weight, "body_sha256": mr.body_sha256,
                        "rev": mr.base_rev, "active": mr.active})
    return out


def merge_diff(merge: MergeResult) -> dict:
    """Reviewable structural diff of the merge against its baseline."""
    changed, deleted = [], []
    for rid, mr in sorted(merge.rows.items()):
        if not mr.active:
            deleted.append(rid)
        elif mr.changed:
            changed.append(mr.as_dict())
    return {"merge_id": merge.merge_id,
            "base_version": merge.base_version,
            "changed": changed, "deleted": deleted,
            "conflicts": [c.as_dict() for c in merge.conflicts]}


# ---------------------------------------------------------------------------
# Promises and rehearsals
# ---------------------------------------------------------------------------

@dataclass
class Promise:
    group: str
    keep: List[str]
    forbid: List[str]


def parse_promise(raw: dict) -> Promise:
    if not isinstance(raw, dict):
        raise PromiseError("promise must be an object")
    group = raw.get("group")
    if not isinstance(group, str) or not group:
        raise PromiseError("promise.group must be a non-empty string")
    keep = list(raw.get("keep", []))
    forbid = list(raw.get("forbid", []))
    for name, ids in (("keep", keep), ("forbid", forbid)):
        if not all(isinstance(x, str) and x for x in ids):
            raise PromiseError(f"promise.{name} entries must be strings")
        if len(set(ids)) != len(ids):
            raise PromiseError(f"duplicate row in promise.{name}")
    overlap = set(keep) & set(forbid)
    if overlap:
        raise PromiseError(f"row in both keep and forbid: {sorted(overlap)}")
    return Promise(group, keep, forbid)


@dataclass
class Rehearsal:
    rehearsal_id: str
    merge_id: str
    base_version: int
    merge_content_sha256: str
    seed_hex: str
    quotas: Dict[str, int]
    promises: List[Promise]
    manifest_groups: Dict[str, dict]
    selected: Dict[str, List[str]]
    promise_ok: bool
    violations: List[dict]

    def as_dict(self) -> dict:
        return {
            "rehearsal_id": self.rehearsal_id,
            "merge_id": self.merge_id,
            "base_version": self.base_version,
            "merge_content_sha256": self.merge_content_sha256,
            "seed_hex": self.seed_hex,
            "quotas": self.quotas,
            "promises": [{"group": p.group, "keep": p.keep,
                          "forbid": p.forbid} for p in self.promises],
            "selected": self.selected,
            "promise_ok": self.promise_ok,
            "violations": self.violations,
            "groups": self.manifest_groups,
        }


def run_rehearsal(merge: MergeResult, baseline_rows: Sequence[dict],
                  seed: bytes, quotas: Mapping[str, int],
                  promises: Sequence[Promise]) -> Rehearsal:
    """Exact lottery on the full merged active groups, then promise checks.

    The lottery is run first, over complete groups, exactly as in the master
    producer.  Promises are only verified against the deterministic result;
    nothing in a promise can add, remove or substitute a selected row.
    """
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    view = effective_view(merge, baseline_rows)
    active = [r for r in view if r["active"]]
    by_group: Dict[str, List[dict]] = {}
    for r in active:
        by_group.setdefault(r["group"], []).append(r)

    group_names = sorted(set(by_group) | set(quotas)
                         | {p.group for p in promises})
    manifest_groups: Dict[str, dict] = {}
    selected: Dict[str, List[str]] = {}
    for g in group_names:
        pool = [make_candidate(r["row_id"], r["group"], r["weight"],
                               r["body_sha256"], seed)
                for r in by_group.get(g, [])]
        quota = int(quotas.get(g, 0))
        manifest_groups[g] = build_group_result(
            g, quota, len(by_group.get(g, [])), pool)
        selected[g] = [c["row_id"] for c in
                       manifest_groups[g]["selected"]]

    violations: List[dict] = []
    for p in promises:
        if p.group not in by_group:
            violations.append({"group": p.group,
                               "kind": "unknown_group"})
            continue
        chosen = set(selected.get(p.group, []))
        present = {r["row_id"] for r in by_group[p.group]}
        for rid in p.keep:
            if rid not in present:
                violations.append({"group": p.group, "kind": "keep_missing",
                                   "row_id": rid})
            elif rid not in chosen:
                violations.append({"group": p.group,
                                   "kind": "keep_not_selected",
                                   "row_id": rid})
        for rid in p.forbid:
            if rid in chosen:
                violations.append({"group": p.group, "kind": "forbid_present",
                                   "row_id": rid})

    identity = {
        "merge_id": merge.merge_id,
        "seed_hex": seed.hex(),
        "quotas": {g: int(q) for g, q in sorted(quotas.items())},
        "promises": [{"group": p.group, "keep": p.keep,
                      "forbid": p.forbid} for p in promises],
    }
    return Rehearsal(
        content_hash(identity), merge.merge_id, merge.base_version,
        merge.content_sha256, seed.hex(),
        {g: int(q) for g, q in sorted(quotas.items())},
        list(promises), manifest_groups, selected, not violations, violations)
