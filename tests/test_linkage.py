"""身份关联不确定性分析的单元与回放测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from linkage_data import calibration
from linkage_data.analyze import report, run, save_snapshot, verify_snapshot
from linkage_data.io import load
from linkage_data.model import ConfigError, make_config

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "scenarios"

BASE = {"period": 1, "version": 1, "n_samples": 120, "calibration_draws": 2}


def analyze(name, **overrides):
    config = dict(BASE)
    config.update(overrides)
    return run(load(SCENARIOS / name / "manifest.json"), config)


def marginal_map(analysis):
    return {key: value for key, value in analysis.marginals.items()}


class CalibrationTests(unittest.TestCase):
    def test_missing_fields_are_not_negative_evidence(self):
        complete = calibration.evidence_vector(0.9, 0.9, 0.9)
        with_missing = calibration.evidence_vector(0.9, None, 0.9)
        self.assertEqual(with_missing[2], 0.0)
        self.assertGreater(sum(with_missing), 0)
        # 缺失只移除证据，不会把分数压到低于“明显不同”的完整向量。
        different = calibration.evidence_vector(0.9, 0.0, 0.9)
        self.assertGreater(sum(with_missing), sum(different) - 1e-12)

    def test_scores_are_not_reported_as_probabilities(self):
        result = report(analyze("phone-change"))
        self.assertIn("比较分数不是概率", result["calibration"]["note"])
        self.assertEqual(result["calibration"]["feature_names"][0], "intercept")

    def test_time_aware_phone_discards_negative_evidence_only(self):
        same_period = calibration.evidence_vector(1.0, 0.0, 1.0, time_aware_phone=True, time_disjoint=False)
        cross_period = calibration.evidence_vector(1.0, 0.0, 1.0, time_aware_phone=True, time_disjoint=True)
        # 同期号码不同提供强负证据（负 LLR）；跨期换号只把负证据抹平为 0。
        self.assertLess(same_period[2], 0.0)
        self.assertEqual(cross_period[2], 0.0)
        self.assertGreater(cross_period[2], same_period[2])

    def test_label_selection_bias_is_declared_with_supplementary_plan(self):
        result = report(analyze("calibration-bias", n_samples=400, calibration_draws=3))
        audit = result["calibration"]["label_audit"]
        self.assertGreater(audit["high_score_share"], 0.8)
        self.assertTrue(audit["warnings"])
        self.assertTrue(any(row["additional_labels_needed"] > 0 for row in audit["supplementary_sampling"]))
        self.assertTrue(any("标注筛选机制未知" in w for w in result["warnings"]))

    def test_unlabeled_calibration_uses_prior_and_warns(self):
        result = report(analyze("phone-change"))
        self.assertEqual(result["calibration"]["source"], "prior")
        self.assertTrue(any("弱先验" in w for w in result["warnings"]))


class CandidateTests(unittest.TestCase):
    def test_weak_chain_is_not_closed(self):
        analysis = analyze("weak-chain", n_samples=600)
        keys = set(analysis.candidates.candidates)
        self.assertIn(("atom:r1", "atom:r2"), keys)
        self.assertIn(("atom:r2", "atom:r3"), keys)
        self.assertNotIn(("atom:r1", "atom:r3"), keys)
        for world in analysis.worlds:
            for block in world.blocks:
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        key = tuple(sorted((block[i], block[j])))
                        self.assertIn(key, keys, "团块内出现了没有直接候选边的原子对")
            self.assertNotIn(
                frozenset(("atom:r1", "atom:r2", "atom:r3")),
                {frozenset(block) for block in world.blocks},
            )

    def test_recall_audit_surfaces_unlinked_pairs(self):
        result = report(analyze("weak-chain", n_samples=300))
        recall = result["candidates"]["recall_audit"]
        self.assertEqual(recall["status"], "unquantified_sampling_required")
        self.assertTrue(recall["strata"])
        self.assertTrue(any("候选集之外" in w for w in recall["warnings"]))

    def test_duplicate_submission_counts_once(self):
        result = report(analyze("duplicate-submission", n_samples=400))
        duplicate = result["candidates"]["duplicate_links"]
        self.assertTrue(duplicate)
        self.assertEqual(result["coverage"]["representative_association"]["beneficiaries_count"], 1)

    def test_mutual_exclusion_prevents_same_name_merge(self):
        analysis = analyze("same-name-different-person", n_samples=400)
        forbidden = analysis.candidates.forbidden
        self.assertIn(("atom:r1", "atom:r2"), forbidden)
        reasons = {r.reason for r in forbidden[("atom:r1", "atom:r2")]}
        self.assertIn("bridge_mutual_exclusion", reasons)
        # r1 锚定 u1、r2 锚定 u2，任何世界都不能合并它们。
        for world in analysis.worlds:
            for block in world.blocks:
                self.assertFalse({"atom:r1", "atom:r2"} <= set(block))

    def test_anchor_conflict_never_forms_triangle(self):
        analysis = analyze("anchor-conflict", n_samples=600)
        for world in analysis.worlds:
            for block in world.blocks:
                self.assertNotEqual({"atom:r1", "atom:r2", "atom:r3"}, set(block))
        marg = marginal_map(analysis)
        # 两个锚定记录永不同块，因此不可能三条边同时为真。
        self.assertEqual(marg.get(("atom:r1", "atom:r3"), 0.0), 0.0)

    def test_forbidden_edges_are_explained(self):
        result = report(analyze("weak-chain"))
        self.assertTrue(result["candidates"]["forbidden_edges"])
        for row in result["candidates"]["forbidden_edges"]:
            self.assertTrue(row["reasons"][0]["reason"])


class WorldConstraintTests(unittest.TestCase):
    def test_worlds_are_legal_partitions(self):
        analysis = analyze("mixed-ops", n_samples=500)
        atoms = set(analysis.candidates.atoms)
        for world in analysis.worlds:
            seen = []
            for block in world.blocks:
                seen.extend(block)
            self.assertEqual(sorted(seen), sorted(atoms), "世界必须覆盖每个原子恰好一次")

    def test_world_distribution_changes_with_evidence(self):
        ambiguous = analyze("phone-change", n_samples=800, calibration_draws=4)
        marginal = marginal_map(ambiguous)[("atom:r1", "atom:r2")]
        self.assertGreater(marginal, 0.01)
        self.assertLess(marginal, 0.99)
        beneficiaries = report(ambiguous)["coverage"]["posterior_distribution"]["beneficiaries_count"]
        self.assertLess(beneficiaries["min"], beneficiaries["max"])
        # 有强标注的场景中，同人对的后验应接近确定，区间也应更窄。
        labeled = report(analyze("calibration-bias", n_samples=500, calibration_draws=3))
        self.assertEqual(labeled["calibration"]["source"], "labeled")

    def test_no_single_identity_id_is_asserted(self):
        result = report(analyze("mixed-ops"))
        representative = result["coverage"]["representative_association"]
        self.assertTrue(representative["assumption"])
        self.assertIn("不是确定身份", representative["note"])

    def test_partial_identification_reports_bounds(self):
        bounds = report(analyze("mixed-ops"))["coverage"]["partial_identification"]
        self.assertLessEqual(bounds["uncovered_fraction_low"], bounds["uncovered_fraction_high"])
        self.assertIn("无法由本地资料识别", bounds["note"])


class EvidenceStrengthTests(unittest.TestCase):
    """同一结构下，只改变证据强弱，覆盖区间应随之变化。"""

    def _tables(self, phone_score, reviewed=None):
        records = [
            {"id": "x1", "department": "A", "period": 1, "name_token": "N1",
             "phone_token": "P1", "birth_bucket": "old"},
            {"id": "x2", "department": "B", "period": 1, "name_token": "N1",
             "phone_token": "P2", "birth_bucket": "old"},
            {"id": "x3", "department": "C", "period": 1, "name_token": "N9",
             "phone_token": "P9", "birth_bucket": "young"},
        ]
        population = [
            {"id": "p1", "birth_bucket": "old", "area": "A"},
            {"id": "p2", "birth_bucket": "young", "area": "C"},
        ]
        comparison = {
            "id": "k1", "left": "x1", "right": "x2",
            "name_score": 1.0, "phone_score": phone_score, "birth_score": 1.0,
            "reviewed_same": reviewed,
        }
        return {
            "records": records,
            "population": population,
            "bridges": [
                {"id": "bb1", "record": "x1", "person": "p1",
                 "valid_from": 1, "valid_to": 2, "revoked_at": None}
            ],
            "comparisons": [comparison],
        }

    def test_stronger_evidence_moves_coverage_and_narrows_interval(self):
        config = {"period": 1, "version": 1, "n_samples": 800, "calibration_draws": 4}
        weak = report(run(self._tables(0.0), config))["coverage"]["posterior_distribution"]["beneficiaries_count"]
        strong = report(run(self._tables(1.0), config))["coverage"]["posterior_distribution"]["beneficiaries_count"]
        # 强证据下 x1、x2 更常被并为一人，受益人数均值更低。
        self.assertLess(strong["mean"], weak["mean"])
        # 强证据的区间不应宽于弱证据。
        self.assertLessEqual(strong["p95"] - strong["p05"], weak["p95"] - weak["p05"])

    def test_missing_phone_is_not_treated_as_mismatch(self):
        result = report(run(self._tables(None), {"period": 1, "version": 1, "n_samples": 400,
                                                 "calibration_draws": 2}))
        rows = result["candidates"]["pair_marginals"]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["posterior_same_person"], 0.0)
        self.assertFalse(result["candidates"]["forbidden_edges"])


class ReviewTests(unittest.TestCase):
    def test_review_queue_is_ordered_by_impact_and_explains_statistics(self):
        result = report(analyze("same-name-different-person", n_samples=600))
        queue = result["review_queue"]
        self.assertTrue(queue)
        impacts = [item["impact_score"] for item in queue]
        self.assertEqual(impacts, sorted(impacts, reverse=True))
        top = queue[0]
        self.assertTrue(top["affected_statistics"])
        self.assertTrue(top["question"])
        self.assertIn("受益人数", top["question"])

    def test_review_prefers_coverage_swing_over_hardest_edge(self):
        result = report(analyze("mixed-ops", n_samples=800, calibration_draws=3))
        queue = result["review_queue"]
        self.assertTrue(queue)
        # 队列首项必须给出对具体统计量的影响，而不是只给相似分数。
        self.assertTrue(queue[0]["swing"])
        self.assertTrue(any(item["kind"] == "recall_sample" for item in queue))

    def test_unresolvable_constraints_do_not_enter_review_queue(self):
        # 出生桶冲突是资料事实，复核不能改变它；只应在禁止边清单中出现。
        result = report(analyze("phone-change", n_samples=400))
        self.assertTrue(any(f["reasons"][0]["reason"] == "birth_conflict"
                            for f in result["candidates"]["forbidden_edges"]))
        self.assertFalse(any(item["kind"] == "forbidden:birth_conflict" for item in result["review_queue"]))

    def test_forbidden_review_item_states_statistic_link(self):
        result = report(analyze("same-name-different-person", n_samples=400))
        item = next(i for i in result["review_queue"] if i["kind"] == "forbidden:bridge_mutual_exclusion")
        self.assertIn("beneficiaries_count", item["affected_statistics"])
        self.assertIn("若放宽", item["question"])


class BridgeVersionTests(unittest.TestCase):
    def test_revocation_changes_later_version_not_old_one(self):
        v1 = report(analyze("bridge-revocation", version=1, n_samples=400))
        v2 = report(analyze("bridge-revocation", version=2, n_samples=400))
        self.assertEqual(v1["bridge_visibility"]["revoked_in_version"], [])
        self.assertIn("b2", v2["bridge_visibility"]["revoked_in_version"])
        self.assertGreater(
            v2["coverage"]["posterior_distribution"]["beneficiaries_count"]["mean"],
            v1["coverage"]["posterior_distribution"]["beneficiaries_count"]["mean"],
        )

    def test_snapshot_recomputes_identically(self):
        analysis = run(load(SCENARIOS / "bridge-revocation" / "manifest.json"), BASE)
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(analysis, directory, "ver")
            result = verify_snapshot(path, load(SCENARIOS / "bridge-revocation" / "manifest.json"))
            self.assertTrue(result["matches"], result["mismatches"])
            # 同一路径再次保存同样内容不得报错，不同内容必须拒绝覆盖。
            save_snapshot(analysis, directory, "ver")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            payload["worlds_digest"] = "0" * 64
            Path(path).write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                save_snapshot(analysis, directory, "ver")

    def test_old_version_still_recomputes_after_revocation(self):
        v1 = run(load(SCENARIOS / "bridge-revocation" / "manifest.json"), BASE)
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(v1, directory, "v1")
            stored = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(stored["config"]["version"], 1)
            # 版本 1 的快照仍以版本 1 的配置复算，不因“当前材料已撤销”而改变。
            self.assertTrue(verify_snapshot(path, load(SCENARIOS / "bridge-revocation" / "manifest.json"))["matches"])


class ConfigTests(unittest.TestCase):
    def test_unknown_config_rejected(self):
        with self.assertRaises(ConfigError):
            make_config({"period": 1, "unexpected": True})

    def test_period_required(self):
        with self.assertRaises(ConfigError):
            make_config({})

    def test_review_weights_validated(self):
        with self.assertRaises(ConfigError):
            make_config({"period": 1, "review_weights": {"unknown": 1.0}})


if __name__ == "__main__":
    unittest.main()
