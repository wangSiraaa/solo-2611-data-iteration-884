import hashlib
import json
import os
import shutil
import tempfile
import unittest

from wrsource.dataset import build_dataset
from wrsource.sampling import make_candidate, select_top
from wrsource.source import SourceService
from wrsource.tuning import TicketError, parse_promise
from wrsource.tunestore import (IssuanceError, MergedView, TuneStore,
                                TuneStoreError)

ZERO_SEED = bytes(32)


def make_row(rid, group, weight, body, shard="s0"):
    return {"op": "add", "row_id": rid, "group": group, "weight": weight,
            "body": body, "shard": shard}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "src.sqlite")
        self.svc = SourceService(self.db)
        self.svc.init()
        self.store = TuneStore(self.db)
        self.store.init()
        # Real old product: A, B in gAB weight 1; C in gOther.
        self.svc.apply_batch([
            make_row("A", "gAB", 1, "body-A"),
            make_row("B", "gAB", 1, "body-B"),
            make_row("C", "gOther", 1, "body-C"),
        ], note="seed master v1")
        self.v1 = self.svc.latest_version()
        self.snap = {r["row_id"]: r for r in self.svc.all_rows_meta(self.v1)}

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def ticket_payload(self, branch, edits, base_version=None):
        records = []
        for rid, changes in edits:
            row = self.snap[rid]
            records.append({
                "row_id": rid,
                "base_rev": row["rev"],
                "base": {"group": row["group"], "weight": row["weight"],
                         "body_sha256": row["body_sha256"], "active": True},
                "changes": changes,
            })
        return {"branch_id": branch,
                "base_version": base_version or self.v1,
                "records": records}

    def register_pair(self, left_edits, right_edits):
        left = self.store.register_ticket(
            self.ticket_payload("L", left_edits))
        right = self.store.register_ticket(
            self.ticket_payload("R", right_edits))
        return self.store.merge_tickets(left.ticket_id, right.ticket_id)


class TicketAndMergePersistenceTest(StoreFixture):
    def test_body_edits_join_shared_library_without_touching_master(self):
        merge = self.register_pair(
            [("A", [{"field": "body", "op": "set", "body": "offline A"}])],
            [("B", [{"field": "weight", "op": "add", "delta": 1}])])
        # Master v1 body is unchanged and still readable.
        self.assertEqual(self.svc.get_row(self.v1, "A")["body"], "body-A")
        # The merged view exposes the offline body.
        view = self.store.open_view(merge.merge_id)
        self.assertEqual(view.get_rows(None, ["A"])["A"]["body"], "offline A")

    def test_merge_idempotent_same_merge_id(self):
        t1 = self.store.register_ticket(
            self.ticket_payload("L", [("A", [{"field": "weight",
                                              "op": "add", "delta": 1}])]))
        t2 = self.store.register_ticket(
            self.ticket_payload("R", [("A", [{"field": "weight",
                                              "op": "add", "delta": 1}])]))
        m1 = self.store.merge_tickets(t1.ticket_id, t2.ticket_id)
        m2 = self.store.merge_tickets(t2.ticket_id, t1.ticket_id)
        self.assertEqual(m1.merge_id, m2.merge_id)
        self.assertEqual(len(self.store.list_merges()), 1)

    def test_conflicted_merge_persists_full_conflict_list(self):
        merge = self.register_pair(
            [("A", [{"field": "weight", "op": "set", "value": 2}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])
        self.assertFalse(merge.ok)
        stored = self.store.get_merge(merge.merge_id)
        self.assertEqual(stored["status"], "conflicted")
        self.assertEqual(stored["conflicts"][0]["reason"], "set_vs_add")
        with self.assertRaises(TuneStoreError):
            self.store.open_view(merge.merge_id)

    def test_ticket_with_diffs_only_is_rejected_on_stale_baseline(self):
        # Advance the master, then hand in a ticket still pinned to v1 rev 1.
        self.svc.apply_batch(
            [{"op": "set_weight", "row_id": "A", "weight": 50}],
            note="master advances")
        v2 = self.svc.latest_version()
        snap2 = {r["row_id"]: r for r in self.svc.all_rows_meta(v2)}
        payload = self.ticket_payload(
            "L", [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            base_version=v2)
        # The record claims the old revision while the v2 snapshot has rev 2.
        payload["records"][0]["base_rev"] = 1
        payload["records"][0]["base"]["weight"] = 1
        with self.assertRaises(TicketError):
            self.store.register_ticket(payload)


class RehearsalAndIssuanceTest(StoreFixture):
    def _double_increment_merge(self):
        return self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])

    def test_rehearsal_runs_full_group_lottery_and_promise_check(self):
        merge = self._double_increment_merge()
        reh = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        self.assertTrue(reh.promise_ok)
        self.assertEqual(reh.selected["gAB"], ["A"])
        # Stored and reloadable, naming the exact merge content it reused.
        loaded = self.store.get_rehearsal(reh.rehearsal_id)
        self.assertEqual(loaded["merge_content_sha256"],
                         merge.content_sha256)
        self.assertEqual(loaded["selected"], reh.selected)

    def test_same_merge_many_promises_each_reuse_that_merge(self):
        merge = self._double_increment_merge()
        ok = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        bad = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["B"]})])
        self.assertTrue(ok.promise_ok)
        self.assertFalse(bad.promise_ok)
        for listed in self.store.list_rehearsals(merge.merge_id):
            self.assertEqual(listed["merge_id"], merge.merge_id)

    def test_unsatisfiable_promise_cannot_be_requested(self):
        merge = self._double_increment_merge()
        bad = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["B"]})])
        with self.assertRaises(IssuanceError):
            self.store.request_issuance("REL-1", bad.rehearsal_id)

    def test_request_stages_pending_then_approve_publishes_pair(self):
        merge = self._double_increment_merge()
        reh = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        pending = self.store.request_issuance("REL-1", reh.rehearsal_id,
                                              note="first cut")
        self.assertEqual(pending["state"], "pending")
        self.assertIsNone(pending["merged_version"])
        # While pending the sample exists but only as a preview reference.
        preview = self.store.get_sample_records("REL-1")
        self.assertEqual({r["row_id"] for r in preview}, {"A"})
        self.assertTrue(all(r["source_ref_kind"] == "preview" for r in preview))
        self.assertEqual(self.store.list_merged_versions(), [])

        approved = self.store.decide_issuance("REL-1", True)
        self.assertEqual(approved["state"], "approved")
        self.assertIsNotNone(approved["merged_version"])
        published = self.store.get_sample_records("REL-1")
        self.assertTrue(all(r["source_ref_kind"] == "merged"
                            for r in published))
        versions = self.store.list_merged_versions()
        self.assertEqual(len(versions), 1)
        # Source version and sample reference the same merged version number.
        self.assertEqual(published[0]["source_ref_version"],
                         versions[0]["merged_ref"])
        detail = self.store.get_merged_version(
            approved["merged_version"])
        self.assertEqual({r["row_id"] for r in detail["rows"]},
                         {"A", "B", "C"})
        self.assertTrue(all(r["active"] for r in detail["rows"]))
        merged_a = next(r for r in detail["rows"] if r["row_id"] == "A")
        self.assertEqual(merged_a["weight"], 3)

    def test_reject_keeps_reviewable_preview_but_no_published_pair(self):
        merge = self._double_increment_merge()
        reh = self.store.rehearse(merge.merge_id, ZERO_SEED, {"gAB": 1}, [])
        self.store.request_issuance("REL-2", reh.rehearsal_id)
        rejected = self.store.decide_issuance("REL-2", False, note="held")
        self.assertEqual(rejected["state"], "rejected")
        self.assertIsNone(rejected["merged_version"])
        # Preview and diff remain queryable; no merged version is allocated.
        self.assertEqual(len(self.store.get_sample_records("REL-2")), 1)
        self.assertEqual(self.store.list_merged_versions(), [])
        with self.assertRaises(IssuanceError):
            self.store.decide_issuance("REL-2", True)

    def test_retry_same_content_is_idempotent(self):
        merge = self._double_increment_merge()
        reh = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        first = self.store.request_issuance("REL-9", reh.rehearsal_id)
        again = self.store.request_issuance("REL-9", reh.rehearsal_id)
        self.assertEqual(first["sample_records_sha256"],
                         again["sample_records_sha256"])
        self.assertEqual(len(self.store.list_issuances()), 1)
        # A crash between request and decision is recoverable: pending stays.
        approved = self.store.decide_issuance("REL-9", True)
        self.assertEqual(approved["state"], "approved")
        # After approval another decide is impossible.
        with self.assertRaises(IssuanceError):
            self.store.decide_issuance("REL-9", True)

    def test_same_issuance_number_rejects_a_different_merge(self):
        merge = self._double_increment_merge()
        reh_a = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        self.store.request_issuance("REL-7", reh_a.rehearsal_id)
        # A different promise on the same merge already yields a new
        # rehearsal identity; the number must not be reused.
        reh_b = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 2}, [])
        with self.assertRaises(IssuanceError):
            self.store.request_issuance("REL-7", reh_b.rehearsal_id)

    def test_number_rebind_blocked_after_master_baseline_advances(self):
        merge = self._double_increment_merge()
        reh = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        self.store.request_issuance("REL-3", reh.rehearsal_id)
        status = self.store.baseline_status(merge.merge_id)
        self.assertTrue(status["baseline_current"])
        # Master advances; the preview merge stays pinned to its base.
        self.svc.apply_batch(
            [{"op": "set_weight", "row_id": "B", "weight": 9}],
            note="master v2")
        status2 = self.store.baseline_status(merge.merge_id)
        self.assertFalse(status2["baseline_current"])
        self.assertEqual(status2["advanced_by"], 1)
        # Even now the same number cannot take a different merge result.
        merge2 = self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 2}])],
            [("B", [{"field": "group", "op": "set", "value": "gOther"}])])
        reh2 = self.store.rehearse(merge2.merge_id, ZERO_SEED,
                                   {"gAB": 1, "gOther": 1}, [])
        with self.assertRaises(IssuanceError):
            self.store.request_issuance("REL-3", reh2.rehearsal_id)
        # The original pending issuance can still be decided on its own merge.
        approved = self.store.decide_issuance("REL-3", True)
        self.assertEqual(approved["merge_id"], merge.merge_id)

    def test_old_master_and_offline_preview_are_independently_queryable(self):
        merge = self._double_increment_merge()
        reh = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        self.store.request_issuance("REL-4", reh.rehearsal_id)
        # Master snapshot still has A at weight 1.
        self.assertEqual(self.svc.get_row(self.v1, "A")["weight"], 1)
        # Offline effective view independently shows weight 3.
        view_rows = {r["row_id"]: r for r in
                     self.store.effective_rows(merge.merge_id)}
        self.assertEqual(view_rows["A"]["weight"], 3)
        # The two surfaces have disjoint version namespaces.
        self.assertEqual(self.svc.latest_version(), self.v1)
        self.store.decide_issuance("REL-4", True)
        merged_refs = [v["merged_ref"] for v in
                       self.store.list_merged_versions()]
        self.assertTrue(all(r.startswith("merged-") for r in merged_refs))
        self.assertNotIn(self.v1, merged_refs)


class MergedViewProducerTest(StoreFixture):
    def test_dataset_producer_runs_unchanged_on_merged_view(self):
        merge = self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])
        view = self.store.open_view(merge.merge_id)
        out_dir = os.path.join(self.dir, "ds_offline")
        manifest, records = build_dataset(
            view, None, ZERO_SEED, {"gAB": 1, "gOther": 1}, out_dir)
        self.assertEqual([r["row_id"] for r in records if r["group"] == "gAB"],
                         ["A"])
        # Real bodies come from the merged view.
        a_rec = next(r for r in records if r["row_id"] == "A")
        self.assertEqual(a_rec["body"], "body-A")
        self.assertEqual(a_rec["weight"], 3)
        # Re-read from disk: manifest references the merge, not a master id.
        with open(os.path.join(out_dir, "manifest.json"),
                  encoding="utf-8") as fh:
            on_disk = json.load(fh)
        self.assertTrue(
            str(on_disk["source_version"]).startswith("merged-preview:"))

    def test_preview_matches_exact_reference_priority(self):
        merge = self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])
        view = self.store.open_view(merge.merge_id)
        meta = [r for r in view.all_rows_meta() if r["group"] == "gAB"]
        cands = [make_candidate(r["row_id"], r["group"], r["weight"],
                                r["body_sha256"], ZERO_SEED) for r in meta]
        self.assertEqual(select_top(cands, 1)[0].row_id, "A")

    def test_promise_never_substitutes_a_row(self):
        # A keep-promise naming the row that *would* lose cannot promote it;
        # the full-group deterministic draw is the only thing that selects.
        merge = self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])
        reh = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1, "gOther": 1},
            [parse_promise({"group": "gAB", "keep": ["A", "B"]})])
        # Quota 1 means exactly one row; B cannot be substituted in for A.
        self.assertEqual(reh.selected["gAB"], ["A"])
        self.assertFalse(reh.promise_ok)
        kinds = {v["kind"] for v in reh.violations}
        self.assertIn("keep_not_selected", kinds)

    def test_over_quota_group_draws_full_pool_with_cut_certificate(self):
        merge = self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])
        reh = self.store.rehearse(merge.merge_id, ZERO_SEED,
                                  {"gAB": 1, "gOther": 1}, [])
        entry = reh.manifest_groups["gAB"]
        self.assertEqual(entry["eligible"], 2)
        self.assertEqual(len(entry["selected"]), 1)
        self.assertIsNotNone(entry["cut"])
        self.assertEqual(entry["cut"]["cut"]["row_id"], "B")
        # gOther is untouched by the merge but still drawn in full.
        self.assertEqual(reh.selected["gOther"], ["C"])

    def test_rehearsals_and_issuance_explain_reused_merge_version(self):
        merge = self.register_pair(
            [("A", [{"field": "weight", "op": "add", "delta": 1}])],
            [("A", [{"field": "weight", "op": "add", "delta": 1}])])
        first = self.store.rehearse(
            merge.merge_id, ZERO_SEED, {"gAB": 1},
            [parse_promise({"group": "gAB", "keep": ["A"]})])
        second = self.store.rehearse(
            merge.merge_id, b"\x11" * 32, {"gAB": 1}, [])
        for reh in (first, second):
            self.assertEqual(reh.merge_id, merge.merge_id)
            self.assertEqual(reh.merge_content_sha256,
                             merge.content_sha256)
        self.assertNotEqual(first.rehearsal_id, second.rehearsal_id)


if __name__ == "__main__":
    unittest.main()
