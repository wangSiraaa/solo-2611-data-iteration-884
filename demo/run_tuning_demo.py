#!/usr/bin/env python3
"""End-to-end demo: offline two-branch tuning merge + promise-bound release.

Run from the repository root::

    python3 demo/run_tuning_demo.py

It starts from a *real* old library built through the normal SourceService,
then walks through the exact scenario from the task:

1. two editing points check out the same published source version and bring
   back diff-only tickets (never a whole database);
2. double increment on A (merge weight 3) -> zero-seed lottery picks A ->
   the "keep A" promise is satisfiable and the pair is issued, with a
   simulated crash between staging and the final decision and an idempotent
   retry;
3. "set 2" vs "+1" -> conflict even though both branches alone reach 2;
4. the same double-increment merge with a "keep B" promise -> the merge is
   fine but the paired release is refused;
5. the master baseline advances; the same issuance number can never be
   rebound to a different merge, while old master products, offline previews
   and published merged versions remain independently queryable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wrsource.source import SourceService
from wrsource.tuning import parse_promise
from wrsource.tunestore import IssuanceError, TuneStore, TuneStoreError

ZERO_SEED = bytes(32)


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def show(title, obj):
    print(f"-- {title}")
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "_work_tuning"))
    args = ap.parse_args()

    if os.path.isdir(args.work):
        shutil.rmtree(args.work)
    os.makedirs(args.work)
    db = os.path.join(args.work, "source.sqlite")

    # ------------------------------------------------------------------
    banner("1. Build the real old library through the master service")
    svc = SourceService(db)
    svc.init()
    svc.apply_batch([
        {"op": "add", "row_id": "A", "group": "gAB", "weight": 1,
         "body": "alpha", "shard": "s0"},
        {"op": "add", "row_id": "B", "group": "gAB", "weight": 1,
         "body": "bravo", "shard": "s0"},
        {"op": "add", "row_id": "C", "group": "gAB", "weight": 2,
         "body": "charlie", "shard": "s1"},
        {"op": "add", "row_id": "D", "group": "gOther", "weight": 1,
         "body": "delta", "shard": "s1"},
    ], note="master baseline v1")
    v1 = svc.latest_version()
    snap = {r["row_id"]: r for r in svc.all_rows_meta(v1)}
    store = TuneStore(db)
    store.init()
    print(f"master version {v1}: rows",
          {rid: (r["group"], r["weight"]) for rid, r in sorted(snap.items())})

    def ticket(branch, edits, base_version=v1):
        records = []
        for rid, changes in edits:
            row = snap[rid]
            records.append({
                "row_id": rid, "base_rev": row["rev"],
                "base": {"group": row["group"], "weight": row["weight"],
                         "body_sha256": row["body_sha256"], "active": True},
                "changes": changes,
            })
        return store.register_ticket(
            {"branch_id": branch, "base_version": base_version,
             "records": records})

    def wadd(d):
        return {"field": "weight", "op": "add", "delta": d}

    # ------------------------------------------------------------------
    banner("2. Both editing points add +1 to A (diffs only) and merge")
    left = ticket("editing-point-L", [("A", [wadd(1)])])
    right = ticket("editing-point-R", [("A", [wadd(1)])])
    merge = store.merge_tickets(left.ticket_id, right.ticket_id)
    show("merge result (A should land at weight 3)", {
        "merge_id": merge.merge_id,
        "status": merge.status,
        "A": merge.rows["A"].as_dict(),
    })
    show("reviewable merge diff vs baseline", store.merge_diff(merge.merge_id))
    print("baseline status:", store.baseline_status(merge.merge_id))

    # ------------------------------------------------------------------
    banner("3. Rehearse the exact lottery over the full active group")
    promise_keep_a = [parse_promise({"group": "gAB", "keep": ["A"]})]
    reh = store.rehearse(merge.merge_id, ZERO_SEED, {"gAB": 1, "gOther": 1},
                         promise_keep_a)
    show("zero-seed draw on the merged source (baseline drew B; now A)", {
        "rehearsal_id": reh.rehearsal_id,
        "reused_merge_id": reh.merge_id,
        "reused_merge_sha256": reh.merge_content_sha256,
        "selected": reh.selected,
        "promise_ok": reh.promise_ok,
        "violations": reh.violations,
    })

    # ------------------------------------------------------------------
    banner("4. Stage issuance, simulate a crash, then retry idempotently")
    pending = store.request_issuance("ISS-2026-001", reh.rehearsal_id,
                                     note="promise: keep A")
    show("after request (sample is preview-only, no merged version yet)",
         pending)
    print("merged versions while pending:", store.list_merged_versions())
    print("preview sample rows:")
    for r in store.get_sample_records("ISS-2026-001"):
        print("  ", r["group_name"], r["rank"], r["row_id"], "w=", r["weight"],
              "ref=", r["source_ref_kind"], r["source_ref_version"])

    # Simulated process restart: the request log survived; re-requesting the
    # same (issuance_no, rehearsal) must be idempotent rather than duplicate.
    retry = store.request_issuance("ISS-2026-001", reh.rehearsal_id)
    print("retry returns the same pending record:",
          retry["state"] == "pending",
          retry["sample_records_sha256"] == pending["sample_records_sha256"])

    approved = store.decide_issuance("ISS-2026-001", True,
                                     note="release approved after retry")
    show("approval publishes merged source + sample as one pair", {
        "state": approved["state"],
        "merged_version": approved["merged_version"],
    })
    print("published sample rows now reference the merged version:")
    for r in store.get_sample_records("ISS-2026-001"):
        print("  ", r["group_name"], r["rank"], r["row_id"], "w=", r["weight"],
              "ref=", r["source_ref_kind"], r["source_ref_version"])
    show("published merged version (full active set, A at weight 3)",
         store.get_merged_version(approved["merged_version"]))

    # ------------------------------------------------------------------
    banner("5. Rule conflict: 'set 2' vs '+1' even though both end at 2")
    set_left = ticket("editing-point-L2",
                      [("A", [{"field": "weight", "op": "set",
                               "value": 2}])])
    conflict_merge = store.merge_tickets(set_left.ticket_id, right.ticket_id)
    show("conflict with both-side vouchers", {
        "status": conflict_merge.status,
        "conflicts": conflict_merge.as_dict()["conflicts"],
    })
    try:
        store.open_view(conflict_merge.merge_id)
    except TuneStoreError as exc:
        print("cannot open an effective view of a conflicted merge:", exc)

    # ------------------------------------------------------------------
    banner("6. Same valid merge, promise switched to 'keep B' -> refused")
    reh_keep_b = store.rehearse(
        merge.merge_id, ZERO_SEED, {"gAB": 1, "gOther": 1},
        [parse_promise({"group": "gAB", "keep": ["B"]})])
    show("the lottery is unchanged; only the promise now fails", {
        "selected": reh_keep_b.selected,
        "promise_ok": reh_keep_b.promise_ok,
        "violations": reh_keep_b.violations,
        "reused_merge_sha256": reh_keep_b.merge_content_sha256,
    })
    try:
        store.request_issuance("ISS-2026-002", reh_keep_b.rehearsal_id)
    except IssuanceError as exc:
        print("issuance refused:", exc)
    # The rejected attempt leaves only a reviewable rehearsal, no issuance.
    print("issuances so far:", [i["issuance_no"] for i in
                                store.list_issuances()])
    # Approving then rejecting is one-shot: show reject path on a fresh number.
    reh_forbid = store.rehearse(
        merge.merge_id, ZERO_SEED, {"gAB": 1},
        [parse_promise({"group": "gAB", "forbid": ["C"]})])
    store.request_issuance("ISS-2026-003", reh_forbid.rehearsal_id)
    rejected = store.decide_issuance("ISS-2026-003", False,
                                     note="held back manually")
    print("explicit reject keeps the preview, publishes nothing:",
          rejected["state"], "merged_version=", rejected["merged_version"])

    # ------------------------------------------------------------------
    banner("7. Master baseline advances; one issuance number = one merge")
    svc.apply_batch([{"op": "set_weight", "row_id": "B", "weight": 9}],
                    note="master v2 lands while offline work continues")
    v2 = svc.latest_version()
    show("baseline status after master advances",
         store.baseline_status(merge.merge_id))
    # A fresh offline merge pinned to the new baseline.
    snap2 = {r["row_id"]: r for r in svc.all_rows_meta(v2)}
    new_left_payload = {
        "branch_id": "editing-point-L3", "base_version": v2,
        "records": [{
            "row_id": "C", "base_rev": snap2["C"]["rev"],
            "base": {"group": snap2["C"]["group"],
                     "weight": snap2["C"]["weight"],
                     "body_sha256": snap2["C"]["body_sha256"],
                     "active": True},
            "changes": [{"field": "group", "op": "set", "value": "gOther"}],
        }]}
    new_left = store.register_ticket(new_left_payload)
    # The second editing point carries a real diff on another row.
    new_right_payload = {
        "branch_id": "editing-point-R3", "base_version": v2,
        "records": [{
            "row_id": "D", "base_rev": snap2["D"]["rev"],
            "base": {"group": snap2["D"]["group"],
                     "weight": snap2["D"]["weight"],
                     "body_sha256": snap2["D"]["body_sha256"],
                     "active": True},
            "changes": [{"field": "weight", "op": "add", "delta": 1}],
        }]}
    new_right = store.register_ticket(new_right_payload)
    merge2 = store.merge_tickets(new_left.ticket_id, new_right.ticket_id)
    reh2 = store.rehearse(merge2.merge_id, ZERO_SEED,
                          {"gAB": 1, "gOther": 2}, [])
    try:
        store.request_issuance("ISS-2026-001", reh2.rehearsal_id,
                               note="attempt to reuse the old number")
    except IssuanceError as exc:
        print("rebinding ISS-2026-001 to a different merge is blocked:", exc)

    # ------------------------------------------------------------------
    banner("8. Independent queries: old product, preview, published pair")
    print("master versions (untouched integer namespace):",
          [v["version_id"] for v in svc.list_versions()])
    print("master v1 still serves A at weight",
          svc.get_row(v1, "A")["weight"], "and body",
          svc.get_row(v1, "A")["body"])
    print("offline preview of the first merge still shows A at weight",
          next(r for r in store.effective_rows(merge.merge_id)
               if r["row_id"] == "A")["weight"])
    print("published merged versions:",
          [(v["merged_ref"], v["base_version"], v["issuance_no"])
           for v in store.list_merged_versions()])
    print("all issuance records:")
    for i in store.list_issuances():
        print("  ", i["issuance_no"], i["state"],
              "merge=", i["merge_id"][:10], "merged_version=",
              i["merged_version"])
    print("all rehearsals kept for review:",
          [(r["rehearsal_id"][:10], r["promise_ok"])
           for r in store.list_rehearsals()])

    banner("Replay with the CLI")
    print(f"""
python3 -m wrsource.tunestore merges        --db {db}
python3 -m wrsource.tunestore merge-info    --db {db} --merge {merge.merge_id}
python3 -m wrsource.tunestore diff          --db {db} --merge {merge.merge_id}
python3 -m wrsource.tunestore baseline-status --db {db} --merge {merge.merge_id}
python3 -m wrsource.tunestore rehearsals    --db {db}
python3 -m wrsource.tunestore issuances     --db {db}
python3 -m wrsource.tunestore sample        --db {db} --no ISS-2026-001
python3 -m wrsource.tunestore merged-versions --db {db}
""")
    print("work directory:", args.work)


if __name__ == "__main__":
    main()
