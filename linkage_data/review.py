"""低置信复核队列：按对覆盖判断的影响排序。

与“只挑最难匹配的边”不同，这里的优先级来自该条边或冲突对业务统计量的
条件影响：连接与不连接两种世界下，受益人数、部门重叠、未覆盖比例分别会
怎样变化。每条复核项都写明会改变哪个统计量、方向以及量级，供业务选择
先复核哪些。

队列包含三类：
* 候选边：后验落在存疑区间的边，按合并/不合并两种世界下统计量之差排序；
* 互斥冲突：同一记录被多条可信桥接指向不同 person 等可裁决冲突；
* 漏配补抽样：候选之外属性相容但未登记比较的记录对，按 strata 抽样。

出生桶冲突、跨期超限这类确定性约束不进入队列：它们由资料或配置直接决定，
复核只能确认规则而不是改变覆盖数字。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidates import CandidateSet
from .prepare import Material
from .worlds import World


@dataclass
class ReviewItem:
    kind: str
    item_id: str
    left: str
    right: str | None
    posterior_same: float | None
    affected_statistics: list[str]
    linked_case: dict[str, Any]
    unlinked_case: dict[str, Any]
    swing: dict[str, float]
    impact: float
    question: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "item_id": self.item_id,
            "left": self.left,
            "right": self.right,
            "posterior_same_person": None if self.posterior_same is None else round(self.posterior_same, 4),
            "affected_statistics": self.affected_statistics,
            "impact_score": round(self.impact, 4),
            "linked_case": self.linked_case,
            "unlinked_case": self.unlinked_case,
            "swing": {k: round(v, 4) for k, v in self.swing.items()},
            "question": self.question,
            "evidence": self.evidence,
        }


class ReviewPlanner:
    def __init__(self, candidates: CandidateSet, material: Material, cfg: dict[str, Any]):
        self.candidates = candidates
        self.material = material
        self.cfg = cfg
        self.weights = cfg["review_weights"]

    def plan(self, worlds: list[World], marginals: dict[tuple[str, str], float],
             metrics_by_world: list[Any], recall: dict[str, Any] | None = None) -> list[ReviewItem]:
        items: list[ReviewItem] = []
        low, high = self.cfg["review_low"], self.cfg["review_high"]
        for key, cand in sorted(self.candidates.candidates.items()):
            posterior = marginals.get(key, 0.0)
            if not (low <= posterior <= high):
                continue
            item = self._edge_item(key, posterior, worlds, metrics_by_world)
            if item is not None:
                items.append(item)
        for conflict in self.candidates.conflicts:
            item = self._conflict_item(conflict, worlds, metrics_by_world)
            if item is not None:
                items.append(item)
        for forbidden_key, reasons in sorted(self.candidates.forbidden.items()):
            for reason in reasons:
                item = self._forbidden_item(forbidden_key, reason)
                if item is not None:
                    items.append(item)
        if recall and recall.get("strata"):
            items.extend(self._recall_items(recall))
        items.sort(key=lambda item: (-item.impact, item.kind, item.item_id))
        return items[: self.cfg["review_limit"]]

    # ------------------------------------------------------------------
    def _edge_item(self, key, posterior, worlds, metrics_by_world) -> ReviewItem | None:
        linked = [m for world, m in zip(worlds, metrics_by_world) if _same(world, key)]
        unlinked = [m for world, m in zip(worlds, metrics_by_world) if not _same(world, key)]
        if not linked or not unlinked:
            return None
        swing = {
            "beneficiaries_count": _mean(m.beneficiaries for m in linked) - _mean(m.beneficiaries for m in unlinked),
            "department_overlap_people": _mean(m.overlapping_people for m in linked) - _mean(m.overlapping_people for m in unlinked),
            "uncovered_fraction": _mean(_uncovered_mid(m) for m in linked) - _mean(_uncovered_mid(m) for m in unlinked),
        }
        # 影响 = 复核改变判断的概率 × 判断改变时业务统计量的移动量。
        # p*(1-p) 是“两种情形都有可能、复核能定夺”的度量；再乘各统计量的
        # 实际差，避免把“中间概率但无关紧要”的边排到前面。
        uncertainty = posterior * (1.0 - posterior)
        impact = uncertainty * (
            abs(swing["beneficiaries_count"]) * self.weights["beneficiaries_count"]
            + abs(swing["department_overlap_people"]) * self.weights["department_overlap_total"]
            + abs(swing["uncovered_fraction"]) * self.weights["uncovered_fraction"]
        )
        a, b = self.candidates.atoms[key[0]], self.candidates.atoms[key[1]]
        cand = self.candidates.candidates[key]
        statistics = []
        thresholds = {"beneficiaries_count": 0.05, "department_overlap_people": 0.05, "uncovered_fraction": 0.001}
        if abs(swing["beneficiaries_count"]) >= thresholds["beneficiaries_count"]:
            statistics.append("beneficiaries_count")
        if abs(swing["department_overlap_people"]) >= thresholds["department_overlap_people"]:
            statistics.append("department_overlap_people")
        if abs(swing["uncovered_fraction"]) >= thresholds["uncovered_fraction"]:
            statistics.append("uncovered_fraction")
        if not statistics:
            statistics = ["beneficiaries_count"]
        question = (
            f"{'/'.join(a.records)} 与 {'/'.join(b.records)} 是否同一人？"
            f"若合并，受益人数会变化 {swing['beneficiaries_count']:+.1f} 人，"
            f"部门重叠会变化 {swing['department_overlap_people']:+.1f} 人，"
            f"未覆盖比例会变化 {swing['uncovered_fraction']*100:+.1f} 个百分点。"
        )
        return ReviewItem(
            kind="candidate_edge",
            item_id=f"edge:{key[0]}:{key[1]}",
            left=key[0],
            right=key[1],
            posterior_same=posterior,
            affected_statistics=statistics,
            linked_case=_case_view(linked),
            unlinked_case=_case_view(unlinked),
            swing=swing,
            impact=impact,
            question=question,
            evidence={
                "scores": cand.scores,
                "sources": cand.sources,
                "comparison_id": cand.comparison_id,
                "departments": sorted(a.departments | b.departments),
                "anchors": [sorted(a.anchors), sorted(b.anchors)],
            },
        )

    def _conflict_item(self, conflict: dict[str, Any], worlds, metrics_by_world) -> ReviewItem | None:
        kind = conflict["kind"]
        records = conflict.get("records") or ([conflict["record"]] if conflict.get("record") else [])
        atom_ids = sorted({self.candidates.record_to_atom[rid] for rid in records if rid in self.candidates.record_to_atom})
        if not atom_ids:
            return None
        atom = self.candidates.atoms[atom_ids[0]]
        # 冲突裁决就是“这些记录是否合并”的二选一。用当前世界中它们是否同块
        # 分成两组，直接计算业务统计量的条件差异。
        linked = [m for world, m in zip(worlds, metrics_by_world) if _all_together(world, atom_ids)]
        unlinked = [m for world, m in zip(worlds, metrics_by_world) if not _all_together(world, atom_ids)]
        if linked and unlinked:
            swing = {
                "beneficiaries_count": _mean(m.beneficiaries for m in linked) - _mean(m.beneficiaries for m in unlinked),
                "department_overlap_people": _mean(m.overlapping_people for m in linked) - _mean(m.overlapping_people for m in unlinked),
                "uncovered_fraction": _mean(_uncovered_mid(m) for m in linked) - _mean(_uncovered_mid(m) for m in unlinked),
            }
            impact = abs(swing["beneficiaries_count"]) + abs(swing["department_overlap_people"]) + abs(
                swing["uncovered_fraction"]) * self.weights["uncovered_fraction"]
            question = (
                f"{'/'.join(atom.records)} 的可信桥接互相冲突（{kind}）。若裁决为同一人，"
                f"受益人数会变化 {swing['beneficiaries_count']:+.1f} 人，部门重叠会变化 "
                f"{swing['department_overlap_people']:+.1f} 人，未覆盖比例会变化 "
                f"{swing['uncovered_fraction']*100:+.1f} 个百分点。"
            )
        else:
            swing = {"beneficiaries_count": 0.0, "department_overlap_people": 0.0, "uncovered_fraction": 0.0}
            impact = 0.5
            question = (
                f"{'/'.join(atom.records)} 的可信桥接互相冲突（{kind}）；当前抽样中该裁决几乎确定，"
                "请确认是否采纳。"
            )
        return ReviewItem(
            kind=f"conflict:{kind}",
            item_id=f"conflict:{kind}:{atom.id}",
            left=atom.id,
            right=conflict.get("person") or None,
            posterior_same=None,
            affected_statistics=["identity_constraint", "beneficiaries_count", "department_overlap_people",
                                 "uncovered_fraction"],
            linked_case=_case_view(linked) if linked else {"worlds": 0},
            unlinked_case=_case_view(unlinked) if unlinked else {"worlds": 0},
            swing=swing,
            impact=impact,
            question=question,
            evidence=conflict,
        )

    def _forbidden_item(self, key, reason) -> ReviewItem | None:
        if reason.reason == "birth_conflict":
            # 出生桶冲突是资料事实，不需要人工复核；保留在禁止边清单即可。
            return None
        if reason.reason == "time_gap":
            # 时间超限也是配置下的确定性约束；若要放宽应先改配置再重跑。
            return None
        return ReviewItem(
            kind=f"forbidden:{reason.reason}",
            item_id=f"forbidden:{reason.reason}:{key[0]}:{key[1]}",
            left=key[0],
            right=key[1],
            posterior_same=0.0,
            affected_statistics=["identity_constraint", "beneficiaries_count", "department_overlap_people"],
            linked_case={"allowed": False, "reason": reason.reason},
            unlinked_case={"allowed": True, "reason": reason.reason},
            swing={"beneficiaries_count": 0.0, "department_overlap_people": 0.0, "uncovered_fraction": 0.0},
            impact=0.3,
            question=(
                f"禁止边 {reason.reason}：{reason.detail}。该约束直接锁定这两个原子不能合并，"
                "复核只能确认约束是否应保持；若放宽，受益人数和部门重叠会按合并情形改变。"
            ),
        )

    def _recall_items(self, recall: dict[str, Any]) -> list[ReviewItem]:
        items = []
        for row in recall["strata"][:10]:
            if not row.get("recommended_sample"):
                continue
            items.append(
                ReviewItem(
                    kind="recall_sample",
                    item_id=f"recall:{row['stratum']}",
                    left=row["stratum"],
                    right=None,
                    posterior_same=None,
                    affected_statistics=["recall_risk", "beneficiaries_count", "uncovered_fraction"],
                    linked_case={"unlinked_pairs": row["unlinked_candidate_pairs"]},
                    unlinked_case={"note": "未抽样前漏配率不可识别"},
                    swing={"beneficiaries_count": float(row["unlinked_candidate_pairs"])},
                    impact=0.25 * row["unlinked_candidate_pairs"],
                    question=(
                        f"层 {row['stratum']} 有 {row['unlinked_candidate_pairs']} 对未登记比较的相容记录对；"
                        "请随机复核其中若干对，用于估计候选之外的漏配率。"
                    ),
                    evidence=row,
                )
            )
        return items


def _same(world: World, key) -> bool:
    for block in world.blocks:
        if key[0] in block and key[1] in block:
            return True
    return False


def _all_together(world: World, atoms: list[str]) -> bool:
    target = set(atoms)
    for block in world.blocks:
        if target & set(block):
            return target <= set(block)
    return False


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _uncovered_mid(metric) -> float:
    return (metric.uncovered_low + metric.uncovered_high) / 2.0 / metric.roster_size


def _case_view(metrics: list[Any]) -> dict[str, Any]:
    return {
        "worlds": len(metrics),
        "beneficiaries_count_mean": round(_mean(m.beneficiaries for m in metrics), 3),
        "department_overlap_people_mean": round(_mean(m.overlapping_people for m in metrics), 3),
        "uncovered_fraction_mid_mean": round(_mean(_uncovered_mid(m) for m in metrics), 4),
    }
