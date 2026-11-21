import copy
import hashlib
import unittest

from wrsource.tuning import (MergeConflict, PromiseError, TicketError,
                             TuningError, content_hash, effective_view,
                             merge_diff, merge_tickets, parse_promise,
                             parse_ticket, run_rehearsal, validate_against_baseline)

ZERO_SEED = bytes(32)
H_A = "a" * 64
H_B = "b" * 64
H_C = "c" * 64


def base(rid, group="gAB", weight=1, digest=None, rev=1, active=True):
    return {"row_id": rid, "group": group, "weight": weight,
            "body_sha256": digest or hashlib.sha256(rid.encode()).hexdigest(),
            "rev": rev, "active": active}


BASELINE = {r["row_id"]: r for r in (
    base("A", digest=H_A), base("B", digest=H_B), base("C", digest=H_C))}


def ticket(branch, records, base_version=1):
    payload = {"branch_id": branch, "base_version": base_version,
               "records": records}
    return parse_ticket(payload)


def rec(rid, changes, baseline=None, rev=None):
    snap = (baseline or BASELINE)[rid]
    return {
        "row_id": rid,
        "base_rev": rev if rev is not None else snap["rev"],
        "base": {k: snap[k] for k in
                 ("group", "weight", "body_sha256", "active")},
        "changes": changes,
    }


def wadd(d):
    return {"field": "weight", "op": "add", "delta": d}


def wset(n):
    return {"field": "weight", "op": "set", "value": n}


def gset(g):
    return {"field": "group", "op": "set", "value": g}


def bset(text):
    return {"field": "body", "op": "set", "body": text}


def retract():
    return {"field": "active", "op": "retract"}


class TicketValidationTest(unittest.TestCase):
    def test_ticket_id_is_content_hash(self):
        t = ticket("L", [rec("A", [wadd(1)])])
        same = ticket("L", [rec("A", [wadd(1)])])
        other = ticket("L", [rec("A", [wadd(2)])])
        self.assertEqual(t.ticket_id, same.ticket_id)
        self.assertNotEqual(t.ticket_id, other.ticket_id)

    def test_one_field_at_most_once_per_branch(self):
        with self.assertRaises(TicketError):
            ticket("L", [rec("A", [wadd(1), wadd(2)])])

    def test_retract_cannot_pair_with_other_edit(self):
        with self.assertRaises(TicketError):
            ticket("L", [rec("A", [retract(), wadd(1)])])

    def test_weight_limits_checked_at_intake(self):
        with self.assertRaises(TicketError):
            ticket("L", [rec("A", [wset(0)])])
        with self.assertRaises(TicketError):
            ticket("L", [rec("A", [wset(101)])])
        with self.assertRaises(TicketError):
            ticket("L", [rec("A", [wadd(0)])])

    def test_stale_revision_rejected(self):
        t = ticket("L", [rec("A", [wadd(1)], rev=5)])
        with self.assertRaises(TicketError):
            validate_against_baseline(t, BASELINE)

    def test_baseline_content_mismatch_rejected(self):
        snap = copy.deepcopy(BASELINE["A"])
        snap["weight"] = 9
        payload = rec("A", [wadd(1)])
        payload["base"]["weight"] = 9
        t = ticket("L", [payload])
        with self.assertRaises(TicketError):
            validate_against_baseline(t, BASELINE)

    def test_row_not_in_baseline_rejected(self):
        snap = copy.deepcopy(BASELINE)
        del snap["A"]
        t = ticket("L", [rec("A", [wadd(1)])])
        with self.assertRaises(TicketError):
            validate_against_baseline(t, snap)

    def test_divergent_base_versions_rejected(self):
        left = ticket("L", [rec("A", [wadd(1)])], base_version=1)
        right = ticket("R", [rec("B", [wadd(1)])], base_version=2)
        with self.assertRaises(TicketError):
            merge_tickets(left, right, BASELINE)


class MergeRulesTest(unittest.TestCase):
    def test_disjoint_fields_combine(self):
        left = ticket("L", [rec("A", [gset("gX")])])
        right = ticket("R", [rec("A", [bset("new body")])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)
        row = m.rows["A"]
        self.assertEqual(row.group, "gX")
        self.assertEqual(row.body_sha256,
                         hashlib.sha256(b"new body").hexdigest())
        self.assertEqual(sorted(row.sources["group"]), ["L"])
        self.assertEqual(sorted(row.sources["body"]), ["R"])

    def test_parallel_increments_add(self):
        left = ticket("L", [rec("A", [wadd(1)])])
        right = ticket("R", [rec("B", [wadd(2)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)
        self.assertEqual(m.rows["A"].weight, 2)
        self.assertEqual(m.rows["B"].weight, 3)

    def test_two_increments_on_same_row_add_deltas(self):
        left = ticket("L", [rec("A", [wadd(1)])])
        right = ticket("R", [rec("A", [wadd(1)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)
        self.assertEqual(m.rows["A"].weight, 3)
        self.assertEqual(sorted(m.rows["A"].sources["weight"]),
                         ["L", "R"])

    def test_identical_absolute_sets_combine(self):
        left = ticket("L", [rec("A", [wset(7)])])
        right = ticket("R", [rec("A", [wset(7)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)
        self.assertEqual(m.rows["A"].weight, 7)

    def test_distinct_sets_conflict_with_vouchers(self):
        left = ticket("L", [rec("A", [wset(7)])])
        right = ticket("R", [rec("A", [wset(8)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertFalse(m.ok)
        self.assertEqual(m.conflicts[0].reason, "distinct_sets")
        self.assertEqual(m.conflicts[0].left["branch_id"], "L")
        self.assertEqual(m.conflicts[0].right["change"]["value"], 8)

    def test_set_vs_add_conflicts_even_when_end_numbers_equal(self):
        # Left "set 2" leaves the row numerically identical to right "+1",
        # but the operation kinds conflict by rule.
        left = ticket("L", [rec("A", [wset(2)])])
        right = ticket("R", [rec("A", [wadd(1)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertFalse(m.ok)
        c = m.conflicts[0]
        self.assertEqual(c.reason, "set_vs_add")
        self.assertEqual(c.left["branch_end_weight"], 2)
        self.assertEqual(c.right["branch_end_weight"], 2)
        self.assertEqual(c.left["baseline_weight"], 1)

    def test_merged_weight_must_stay_in_old_range(self):
        left = ticket("L", [rec("A", [wadd(99)])])
        right = ticket("R", [rec("A", [wadd(1)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertFalse(m.ok)
        self.assertEqual(m.conflicts[0].reason, "weight_out_of_range")
        # Negative side of the same range rule.
        left2 = ticket("L", [rec("A", [wadd(-1)])])
        m2 = merge_tickets(left2, ticket("R0", [rec("B", [wadd(1)])]),
                           BASELINE)
        self.assertFalse(m2.ok)
        self.assertEqual(m2.conflicts[0].reason, "weight_out_of_range")

    def test_delete_combines_with_untouched_row(self):
        left = ticket("L", [rec("A", [retract()])])
        right = ticket("R", [rec("B", [wadd(1)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)
        self.assertFalse(m.rows["A"].active)
        self.assertTrue(m.rows["B"].active)

    def test_parallel_deletes_combine(self):
        left = ticket("L", [rec("A", [retract()])])
        right = ticket("R", [rec("A", [retract()])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)
        self.assertFalse(m.rows["A"].active)

    def test_delete_vs_real_edit_is_listed_conflict(self):
        left = ticket("L", [rec("A", [retract()])])
        right = ticket("R", [rec("A", [wadd(1)])])
        m = merge_tickets(left, right, BASELINE)
        self.assertFalse(m.ok)
        self.assertEqual(m.conflicts[0].reason, "retract_vs_change")
        # Sides are normalised by ticket id; vouchers are located by branch.
        by_branch = {v["branch_id"]: v for v in
                     (m.conflicts[0].left, m.conflicts[0].right)}
        self.assertEqual(by_branch["L"]["change"]["op"], "retract")
        self.assertEqual(by_branch["R"]["change"]["op"], "add")

    def test_distinct_group_and_body_values_conflict(self):
        left = ticket("L", [rec("A", [gset("gX")])])
        right = ticket("R", [rec("A", [gset("gY")])])
        self.assertFalse(merge_tickets(left, right, BASELINE).ok)
        lb = ticket("L2", [rec("A", [bset("one")])])
        rb = ticket("R2", [rec("A", [bset("two")])])
        m = merge_tickets(lb, rb, BASELINE)
        self.assertFalse(m.ok)
        self.assertEqual(m.conflicts[0].reason, "distinct_body_values")

    def test_identical_body_edits_combine_by_digest(self):
        left = ticket("L", [rec("A", [bset("same text")])])
        right = ticket("R", [rec("A", [bset("same text")])])
        m = merge_tickets(left, right, BASELINE)
        self.assertTrue(m.ok)

    def test_all_conflicts_collected_with_both_vouchers(self):
        left = ticket("L", [rec("A", [wset(2)]), rec("B", [retract()])])
        right = ticket("R", [rec("A", [wadd(1)]), rec("B", [gset("gZ")])])
        m = merge_tickets(left, right, BASELINE)
        self.assertFalse(m.ok)
        rows = {c.row_id for c in m.conflicts}
        self.assertEqual(rows, {"A", "B"})
        for c in m.conflicts:
            self.assertIsInstance(c, MergeConflict)
            self.assertIsNotNone(c.left)
            self.assertIsNotNone(c.right)

    def test_rows_compared_only_by_row_identity(self):
        # Same field kind on different rows never conflicts.
        left = ticket("L", [rec("A", [wset(5)])])
        right = ticket("R", [rec("B", [wset(5)])])
        self.assertTrue(merge_tickets(left, right, BASELINE).ok)

    def test_merge_is_content_addressed_and_deterministic(self):
        left = ticket("L", [rec("A", [wadd(1)])])
        right = ticket("R", [rec("A", [wadd(1)])])
        m1 = merge_tickets(left, right, BASELINE)
        m2 = merge_tickets(left, right, BASELINE)
        self.assertEqual(m1.merge_id, m2.merge_id)
        swapped = merge_tickets(right, left, BASELINE)
        self.assertEqual(swapped.merge_id, m1.merge_id)
        other_right = ticket("R", [rec("A", [wadd(2)])])
        self.assertNotEqual(
            merge_tickets(left, other_right, BASELINE).merge_id,
            m1.merge_id)

    def test_effective_view_and_diff_cover_full_baseline(self):
        left = ticket("L", [rec("A", [wadd(1), gset("gZ")][:1])])
        right = ticket("R", [rec("C", [retract()])])
        m = merge_tickets(left, right, BASELINE)
        view = {r["row_id"]: r for r in
                effective_view(m, list(BASELINE.values()))}
        self.assertEqual(set(view), {"A", "B", "C"})
        self.assertFalse(view["C"]["active"])
        self.assertEqual(view["B"]["weight"], 1)
        diff = merge_diff(m)
        self.assertEqual(diff["deleted"], ["C"])
        self.assertEqual([c["row_id"] for c in diff["changed"]], ["A"])

    def test_conflicted_merge_has_no_effective_view(self):
        left = ticket("L", [rec("A", [wset(2)])])
        right = ticket("R", [rec("A", [wadd(1)])])
        m = merge_tickets(left, right, BASELINE)
        with self.assertRaises(TuningError):
            effective_view(m, list(BASELINE.values()))


class TaskExampleTest(unittest.TestCase):
    """The exact A/B scenario from the task statement."""

    def setUp(self):
        self.baseline = {r["row_id"]: r for r in (
            base("A", group="gAB", weight=1, digest=H_A),
            base("B", group="gAB", weight=1, digest=H_B))}

    def _t(self, branch, rid, changes):
        return ticket(branch, [rec(rid, changes, self.baseline)])

    def test_double_increment_selects_a_and_keep_a_promise_publishes(self):
        left = self._t("L", "A", [wadd(1)])
        right = self._t("R", "A", [wadd(1)])
        m = merge_tickets(left, right, self.baseline)
        self.assertTrue(m.ok)
        self.assertEqual(m.rows["A"].weight, 3)
        reh = run_rehearsal(m, list(self.baseline.values()), ZERO_SEED,
                            {"gAB": 1},
                            [parse_promise({"group": "gAB",
                                            "keep": ["A"]})])
        self.assertEqual(reh.selected["gAB"], ["A"])
        self.assertTrue(reh.promise_ok)

    def test_set_vs_add_conflicts_despite_identical_end_state(self):
        left = self._t("L", "A", [wset(2)])
        right = self._t("R", "A", [wadd(1)])
        m = merge_tickets(left, right, self.baseline)
        self.assertFalse(m.ok)
        self.assertEqual(m.conflicts[0].reason, "set_vs_add")

    def test_merge_ok_but_paired_release_refused_on_broken_promise(self):
        left = self._t("L", "A", [wadd(1)])
        right = self._t("R", "A", [wadd(1)])
        m = merge_tickets(left, right, self.baseline)
        reh = run_rehearsal(m, list(self.baseline.values()), ZERO_SEED,
                            {"gAB": 1},
                            [parse_promise({"group": "gAB",
                                            "keep": ["B"]})])
        self.assertTrue(m.ok)
        self.assertFalse(reh.promise_ok)
        self.assertEqual(reh.violations[0]["kind"], "keep_not_selected")
        self.assertEqual(reh.violations[0]["row_id"], "B")

    def test_baseline_lottery_takes_b(self):
        # Sanity: at weight 1/1 with the zero seed the lottery picks B.
        noop = ticket("L", [rec("B", [bset("still B")], self.baseline)])
        noop2 = ticket("R", [rec("A", [bset("still A")], self.baseline)])
        m = merge_tickets(noop, noop2, self.baseline)
        reh = run_rehearsal(m, list(self.baseline.values()), ZERO_SEED,
                            {"gAB": 1}, [])
        self.assertEqual(reh.selected["gAB"], ["B"])


class RehearsalTest(unittest.TestCase):
    def _merge(self):
        left = ticket("L", [rec("A", [wadd(1)])])
        right = ticket("R", [rec("A", [wadd(1)])])
        return merge_tickets(left, right, BASELINE)

    def test_lottery_uses_full_active_group_not_promise(self):
        m = self._merge()
        reh = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                            {"gAB": 2}, [])
        # Two slots, deterministic exact order; promise cannot alter it.
        self.assertEqual(len(reh.selected["gAB"]), 2)
        again = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                              {"gAB": 2}, [])
        self.assertEqual(again.rehearsal_id, reh.rehearsal_id)
        self.assertEqual(again.selected, reh.selected)

    def test_forbid_violation_detected(self):
        m = self._merge()
        reh = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                            {"gAB": 2},
                            [parse_promise({"group": "gAB",
                                            "forbid": ["A"]})])
        self.assertFalse(reh.promise_ok)
        self.assertEqual(reh.violations[0]["kind"], "forbid_present")

    def test_keep_row_absent_from_group_is_violation(self):
        m = self._merge()
        with self.assertRaises(PromiseError):
            parse_promise({"group": "gAB", "keep": ["A"], "forbid": ["A"]})
        reh = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                            {"gAB": 1},
                            [parse_promise({"group": "gX",
                                            "keep": ["A"]})])
        self.assertEqual(reh.violations[0]["kind"], "unknown_group")

    def test_rehearsal_identity_binds_merge_content(self):
        m = self._merge()
        r1 = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                           {"gAB": 1}, [])
        other_seed = run_rehearsal(m, list(BASELINE.values()), b"\x01" * 32,
                                   {"gAB": 1}, [])
        other_quota = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                                    {"gAB": 2}, [])
        self.assertNotEqual(r1.rehearsal_id, other_seed.rehearsal_id)
        self.assertNotEqual(r1.rehearsal_id, other_quota.rehearsal_id)
        self.assertEqual(r1.merge_content_sha256, m.content_sha256)

    def test_manifest_carries_exact_selection_material(self):
        m = self._merge()
        reh = run_rehearsal(m, list(BASELINE.values()), ZERO_SEED,
                            {"gAB": 1}, [])
        entry = reh.manifest_groups["gAB"]
        self.assertEqual(entry["quota"], 1)
        self.assertEqual(entry["eligible"], 3)
        sel = entry["selected"][0]
        self.assertEqual(sel["row_id"], "A")
        self.assertEqual(sel["weight"], 3)
        self.assertRegex(sel["a"], r"^\d+$")
        self.assertIsNotNone(entry["cut"])


if __name__ == "__main__":
    unittest.main()
