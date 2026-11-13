"""覆盖统计：同一时期受益人数、部门重叠、固定总体未覆盖。

每个关联世界都被转换为业务计数；不把候选分数或边际概率直接当作身份概率，
也不生成单一的“统一ID”作为既成事实。报告包含：

* 代表关联（最高权重世界）——明确标注为一种假设；
* 后验分布（抽样世界）——区间由关联不确定性产生；
* 固定总体未覆盖的部分识别范围——未锚定团块是否属于名单无法由本地资料
  识别，按上下界报告，而不是给未锚定团块硬分配一个概率。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .candidates import CandidateSet
from .prepare import Material
from .worlds import World, _bell_number


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass
class WorldMetrics:
    roster_size: int
    beneficiaries: int
    overlapping_people: int
    department_pairs: int
    anchored_covered: int
    unanchored_blocks: int
    uncovered_low: int
    uncovered_high: int
    blocks: int
    period_blocks: int
    unanchored_ids: list[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    representative: dict[str, Any]
    distribution: dict[str, Any]
    partial_identification: dict[str, Any]
    warnings: list[str]
    world_metrics: list[WorldMetrics] = field(default_factory=list)


def world_metrics(world: World, candidates: CandidateSet, material: Material, cfg: dict[str, Any]) -> WorldMetrics:
    period = cfg["period"]
    roster = [row["id"] for row in material.population]
    roster_set = set(roster)
    beneficiaries = 0
    overlapping = 0
    dept_pairs = 0
    anchored_records = set()
    unanchored: list[str] = []
    period_blocks = 0
    for block in world.blocks:
        members = [candidates.atoms[atom] for atom in block]
        period_departments: set[str] = set()
        in_period = False
        anchors: set[str] = set()
        for atom in members:
            if atom.has_period:
                in_period = True
                period_departments |= atom.period_departments
            anchors |= atom.anchors
        if not in_period:
            continue
        period_blocks += 1
        beneficiaries += 1
        if len(period_departments) >= 2:
            overlapping += 1
            dept_pairs += len(period_departments) * (len(period_departments) - 1) // 2
        # 名单归属只认可信桥接锚点；没有锚点的团块不能声称已覆盖名单中的人，
        # 其归属作为部分识别范围处理，不写回任何身份判定。
        if anchors:
            anchored_records |= (anchors & roster_set)
        else:
            unanchored.append(block[0])
    anchored = len(anchored_records)
    # 未锚定团块的名单归属按部分识别处理：
    # 上界：这些团块全部不在名单中，未覆盖人数 = 名单人数 - 已锚定人数；
    # 下界：这些团块全部对应名单中尚未覆盖的人，但受名单剩余人数限制。
    remaining = max(len(roster) - anchored, 0)
    uncovered_high = remaining
    uncovered_low = max(remaining - min(len(unanchored), remaining), 0)
    return WorldMetrics(
        roster_size=len(roster),
        beneficiaries=beneficiaries,
        overlapping_people=overlapping,
        department_pairs=dept_pairs,
        anchored_covered=anchored,
        unanchored_blocks=len(unanchored),
        uncovered_low=uncovered_low,
        uncovered_high=uncovered_high,
        blocks=len(world.blocks),
        period_blocks=period_blocks,
        unanchored_ids=sorted(unanchored),
    )


def _modal_world(worlds: list[World]) -> World | None:
    """跨校准抽样出现频率最高的划分。

    不直接取 log_weight 最大的世界：不同校准抽样下的对数权重不是同一尺度，
    直接比较会把“概率更极端的那次校准”误当成更可信的假设。
    """
    if not worlds:
        return None
    counts: dict[tuple[tuple[str, ...], ...], int] = {}
    first: dict[tuple[tuple[str, ...], ...], World] = {}
    for world in worlds:
        counts[world.blocks] = counts.get(world.blocks, 0) + 1
        first.setdefault(world.blocks, world)
    best = max(counts, key=lambda key: (counts[key], first[key].log_weight))
    return first[best]


def _world_frequency(worlds: list[World]) -> dict[str, Any]:
    if not worlds:
        return {"samples": 0, "distinct_partitions": 0}
    counts: dict[tuple[tuple[str, ...], ...], int] = {}
    for world in worlds:
        counts[world.blocks] = counts.get(world.blocks, 0) + 1
    top = max(counts.values())
    return {
        "samples": len(worlds),
        "distinct_partitions": len(counts),
        "modal_share": round(top / len(worlds), 4),
    }


def summarize(worlds: list[World], candidates: CandidateSet, material: Material, cfg: dict[str, Any]) -> CoverageReport:
    metrics = [world_metrics(world, candidates, material, cfg) for world in worlds]
    roster_size = len(material.population)
    warnings: list[str] = []
    if roster_size == 0:
        warnings.append("固定总体名单为空，未覆盖比例不可计算")
    representative_world = _modal_world(worlds)
    representative = {}
    if representative_world is not None:
        rep = world_metrics(representative_world, candidates, material, cfg)
        representative = {
            "assumption": True,
            "note": "代表关联是跨校准抽样出现频率最高的单个划分，仅用于展示一种明确标注的"
                    "假设，不是确定身份。",
            "world_frequency": _world_frequency(worlds),
            "blocks": [_block_view(block, candidates, material, cfg) for block in representative_world.blocks],
            "beneficiaries_count": rep.beneficiaries,
            "department_overlap_people": rep.overlapping_people,
            "department_overlap_pairs": rep.department_pairs,
            "uncovered_count_bounds": [rep.uncovered_low, rep.uncovered_high],
            "unanchored_blocks": rep.unanchored_blocks,
        }
    if worlds:
        distinct = len({world.blocks for world in worlds})
        atom_count = len({atom for world in worlds for block in world.blocks for atom in block})
        # 阈值不能超过该原子数下可能的划分数；小场景本来就只有几种划分。
        possible = _bell_number(atom_count)
        min_support = min(cfg.get("min_interval_support", 200), possible)
        if distinct < min_support:
            warnings.append(
                f"抽样世界只出现 {distinct} 种不同划分，少于该原子数下建议的 {min_support}；"
                "这表示后验本身集中，若预期存在更多不确定性，请检查候选覆盖与标注量，"
                "而不是只提高抽样数。"
            )
        if not all(world.exact for world in worlds):
            warnings.append("部分连通分量使用 Gibbs 近似；报告同时给出精确枚举阈值与抽样参数。")
    distribution = _distribution(metrics, roster_size)
    if not worlds:
        warnings.append("没有任何世界样本，覆盖区间为空；请检查资料读取与约束设置")
    bounds = {
        "roster_size": roster_size,
        "uncovered_low": min((m.uncovered_low for m in metrics), default=None) if metrics else None,
        "uncovered_high": max((m.uncovered_high for m in metrics), default=None) if metrics else None,
        "note": (
            "未锚定团块是否对应名单中的具体人无法由本地资料识别；上下界只随身份划分变化，"
            "不代表名单匹配的概率。"
        ),
    }
    if bounds["uncovered_low"] is not None and roster_size:
        bounds["uncovered_fraction_low"] = round(bounds["uncovered_low"] / roster_size, 4)
        bounds["uncovered_fraction_high"] = round(bounds["uncovered_high"] / roster_size, 4)
    return CoverageReport(
        representative=representative,
        distribution=distribution,
        partial_identification=bounds,
        warnings=warnings,
        world_metrics=metrics,
    )


def _distribution(metrics: list[WorldMetrics], roster_size: int) -> dict[str, Any]:
    def summarize_metric(name: str) -> dict[str, float]:
        values = [getattr(m, name) for m in metrics]
        return {
            "mean": round(sum(values) / len(values), 4),
            "median": round(quantile(values, 0.5), 4),
            "p05": round(quantile(values, 0.05), 4),
            "p95": round(quantile(values, 0.95), 4),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    return {
        "samples": len(metrics),
        "interval": "5%-95% 抽样世界分位区间",
        "beneficiaries_count": summarize_metric("beneficiaries"),
        "department_overlap_people": summarize_metric("overlapping_people"),
        "department_overlap_pairs": summarize_metric("department_pairs"),
        "uncovered_count_low": summarize_metric("uncovered_low"),
        "uncovered_count_high": summarize_metric("uncovered_high"),
        "uncovered_fraction_low": {
            k: round(v / roster_size, 4) for k, v in summarize_metric("uncovered_low").items()
        } if roster_size else {},
        "uncovered_fraction_high": {
            k: round(v / roster_size, 4) for k, v in summarize_metric("uncovered_high").items()
        } if roster_size else {},
    }


def _block_view(block: tuple[str, ...], candidates: CandidateSet, material: Material, cfg: dict[str, Any]) -> dict[str, Any]:
    atom_views = []
    for atom_id in block:
        atom = candidates.atoms[atom_id]
        atom_views.append(
            {
                "atom": atom_id,
                "records": atom.records,
                "departments": sorted(atom.departments),
                "period_departments": sorted(atom.period_departments),
                "anchors": sorted(atom.anchors),
                "rationale": sorted(atom.reasons),
            }
        )
    return {"atoms": atom_views}


def pair_summary(candidates: CandidateSet, marginals: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    rows = []
    for key, cand in sorted(candidates.candidates.items()):
        a, b = candidates.atoms[key[0]], candidates.atoms[key[1]]
        rows.append(
            {
                "left": key[0],
                "right": key[1],
                "left_records": a.records,
                "right_records": b.records,
                "departments": sorted(a.departments | b.departments),
                "scores": cand.scores,
                "sources": cand.sources,
                "reviewed_same": cand.reviewed_same,
                "posterior_same_person": round(marginals.get(key, 0.0), 4),
            }
        )
    return rows
