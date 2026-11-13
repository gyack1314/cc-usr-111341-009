"""候选生成：强证据成原子，候选来自已登记比较，弱相似链不闭包。

步骤：
1. 部门内重复上报（同部门、同姓名、同号码、同出生桶）合并为同一原子；
   合并会改变锚定身份时拒绝并登记冲突。
2. 可信桥接在可见版本内把同一 person 的记录合并为原子；同一记录被不同
   person 指向、或同一 person 的记录属性互斥时，桥接不作为合并事实，登记
   互斥冲突。互斥身份约束在后续世界抽样中始终生效。
3. 候选对只来自已登记比较表，不按姓名批量连接。比较表没有登记的对不会
   被静默当作异人；它们进入漏配审计。
4. 明显的禁止边（出生桶冲突、不同可信 person、跨期超限）单独列出。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from .calibration import PairEvidence
from .prepare import Material, Record


@dataclass
class Atom:
    id: str
    records: list[str]
    departments: frozenset[str]
    period_departments: frozenset[str]
    period_min: int
    period_max: int
    names: frozenset[str]
    phones: frozenset[str]
    births: frozenset[str]
    anchors: frozenset[str]
    reasons: frozenset[str]

    @property
    def has_period(self) -> bool:
        return bool(self.period_departments)

@dataclass
class Candidate:
    left: str
    right: str
    scores: dict[str, float | None]
    comparison_id: str | None
    sources: list[str] = field(default_factory=list)
    reviewed_same: bool | None = None

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.left, self.right)))  # type: ignore[return-value]


@dataclass
class Forbidden:
    left: str
    right: str
    reason: str
    detail: str


@dataclass
class CandidateSet:
    atoms: dict[str, Atom]
    candidates: dict[tuple[str, str], Candidate]
    forbidden: dict[tuple[str, str], list[Forbidden]]
    record_to_atom: dict[str, str]
    conflicts: list[dict[str, Any]]
    bridge_states: list[dict[str, Any]]
    duplicate_links: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self, records: dict[str, Record]):
        self.parent = {rid: rid for rid in records}
        self.anchors: dict[str, set[str]] = {rid: set() for rid in records}
        self.births: dict[str, set[str]] = {
            rid: ({record.birth} if record.birth else set())
            for rid, record in records.items()
        }
        self.reasons: dict[str, set[str]] = {rid: {"single"} for rid in records}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_union(self, a: str, b: str) -> str | None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.anchors[ra] and self.anchors[rb] and self.anchors[ra] != self.anchors[rb]:
            return "different_person_anchors"
        if self.births[ra] and self.births[rb] and not (self.births[ra] & self.births[rb]):
            return "birth_conflict"
        return None

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra
        self.anchors[ra] |= self.anchors[rb]
        self.births[ra] |= self.births[rb]
        self.reasons[ra] = (self.reasons[ra] - {"single"}) | (self.reasons[rb] - {"single"})

    def add_anchor(self, record: str, person: str) -> None:
        root = self.find(record)
        self.anchors[root].add(person)


def build(material: Material, cfg: dict[str, Any]) -> CandidateSet:
    warnings: list[str] = []
    conflicts: list[dict[str, Any]] = []
    bridges = material.visible_bridges
    record_persons: dict[str, set[str]] = {}
    person_records: dict[str, set[str]] = {}
    for bridge in bridges:
        if bridge.record not in material.records:
            continue
        record_persons.setdefault(bridge.record, set()).add(bridge.person)
        person_records.setdefault(bridge.person, set()).add(bridge.record)
        # 记录被多条桥接指向不同 person 时，任何一条被采纳都会约束其他记录；
        # 这里不合并，冲突进入复核队列。

    conflicted_records = {rid for rid, persons in record_persons.items() if len(persons) > 1}
    for rid in sorted(conflicted_records):
        conflicts.append(
            {
                "kind": "record_multiple_persons",
                "record": rid,
                "persons": sorted(record_persons[rid]),
                "detail": "同一记录被多条可见桥接指向不同 person；任选一条都会改变身份划分，"
                          "由于互斥身份约束不能同时满足，这些桥接不用于合并，改由复核队列裁决。",
            }
        )

    uf = _UnionFind(material.records)
    bridge_states: list[dict[str, Any]] = []
    for person, recs in sorted(person_records.items()):
        clean = sorted(rid for rid in recs if rid not in conflicted_records)
        if len(clean) < len(recs):
            state = "used_partial"
            detail = "person 的部分记录同时被其他 person 覆盖，仅使用无冲突记录"
        else:
            state = "used"
            detail = ""
        births = {material.records[rid].birth for rid in clean if material.records[rid].birth}
        if len(births) > 1 and clean:
            state = "birth_conflict"
            detail = "同一 person 的桥接记录出生桶互斥"
        anchorable = state in ("used", "used_partial")
        if clean:
            first = clean[0]
            for other in clean[1:]:
                block = uf.can_union(first, other)
                if block:
                    state = "attribute_conflict"
                    detail = block
                    anchorable = False
                    break
                uf.union(first, other)
        # 合并成功后才登记锚点，避免候选生成阶段就把冲突桥接当作事实。
        if anchorable:
            for rid in clean:
                uf.add_anchor(rid, person)
        if state not in ("used", "used_partial"):
            conflicts.append(
                {
                    "kind": "person_bridge_unusable",
                    "person": person,
                    "records": clean,
                    "detail": f"{state}: {detail}",
                }
            )
        bridge_states.append(
            {
                "person": person,
                "records": clean,
                "status": state,
                "detail": detail,
            }
        )

    # 部门内重复上报。仅当合并不会把两个可信 person 并到一起时才合并。
    duplicate_links: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str, str, str], list[str]] = {}
    for rec in material.records.values():
        if rec.name and rec.birth and rec.phone:
            buckets.setdefault((rec.department, rec.name, rec.phone, rec.birth), []).append(rec.id)
    for key, ids in sorted(buckets.items()):
        if len(ids) < 2:
            continue
        base = ids[0]
        for other in sorted(ids[1:]):
            block = uf.can_union(base, other)
            if block:
                conflicts.append(
                    {
                        "kind": "duplicate_anchor_conflict",
                        "records": [base, other],
                        "detail": f"部门内重复上报的属性相同，但合并会违反 {block}",
                    }
                )
                continue
            uf.union(base, other)
            duplicate_links.append({"left": base, "right": other, "basis": "same department, name, phone, birth"})

    groups: dict[str, list[str]] = {}
    for rid in uf.parent:
        groups.setdefault(uf.find(rid), []).append(rid)
    atoms: dict[str, Atom] = {}
    record_to_atom: dict[str, str] = {}
    for root, ids in sorted(groups.items()):
        recs = [material.records[rid] for rid in sorted(ids)]
        atom_id = "atom:" + root
        atoms[atom_id] = Atom(
            id=atom_id,
            records=sorted(ids),
            departments=frozenset(r.department for r in recs),
            period_departments=frozenset(r.department for r in recs if r.period == cfg["period"]),
            period_min=min(r.period for r in recs),
            period_max=max(r.period for r in recs),
            names=frozenset(r.name for r in recs if r.name),
            phones=frozenset(r.phone for r in recs if r.phone),
            births=frozenset(r.birth for r in recs if r.birth),
            anchors=frozenset(uf.anchors[uf.find(root)]),
            reasons=frozenset(uf.reasons[uf.find(root)]),
        )
        for rid in ids:
            record_to_atom[rid] = atom_id

    by_key: dict[tuple[str, str], Candidate] = {}
    for comparison in material.comparisons:
        la = record_to_atom.get(comparison.left)
        ra = record_to_atom.get(comparison.right)
        if la is None or ra is None or la == ra:
            continue
        key = tuple(sorted((la, ra)))
        entry = by_key.get(key)
        if entry is None:
            by_key[key] = Candidate(
                left=key[0],
                right=key[1],
                scores=dict(comparison.scores),
                comparison_id=comparison.id,
                sources=[comparison.id],
                reviewed_same=comparison.reviewed_same,
            )
            continue
        if comparison.id not in entry.sources:
            entry.sources.append(comparison.id)
        for name, value in comparison.scores.items():
            old = entry.scores.get(name)
            if value is not None and (old is None or value > old):
                entry.scores[name] = value
        if comparison.reviewed_same is not None:
            if entry.reviewed_same is not None and entry.reviewed_same != comparison.reviewed_same:
                warnings.append(f"原子对 {key} 的复核结论互相矛盾，按未复核处理")
                entry.reviewed_same = None
            else:
                entry.reviewed_same = comparison.reviewed_same

    forbidden: dict[tuple[str, str], list[Forbidden]] = {}
    for key in sorted(by_key):
        a, b = atoms[key[0]], atoms[key[1]]
        reasons = _forbidden_reasons(a, b, cfg)
        if reasons:
            forbidden[key] = reasons
    candidates = {k: v for k, v in by_key.items() if k not in forbidden}
    return CandidateSet(
        atoms=atoms,
        candidates=candidates,
        forbidden=forbidden,
        record_to_atom=record_to_atom,
        conflicts=conflicts,
        bridge_states=bridge_states,
        duplicate_links=duplicate_links,
        warnings=warnings,
    )


def _forbidden_reasons(a: Atom, b: Atom, cfg: dict[str, Any]) -> list[Forbidden]:
    reasons: list[Forbidden] = []
    if a.anchors and b.anchors and not (a.anchors & b.anchors):
        reasons.append(Forbidden(a.id, b.id, "bridge_mutual_exclusion", "原子锚定到不同的可信 person"))
    if a.births and b.births and not (a.births & b.births):
        reasons.append(Forbidden(a.id, b.id, "birth_conflict", "出生桶冲突，不能是同一人"))
    gap = cfg.get("max_period_gap")
    if gap is not None:
        span = max(a.period_max, b.period_max) - min(a.period_min, b.period_min)
        if span > gap:
            reasons.append(Forbidden(a.id, b.id, "time_gap", f"时期跨度 {span} 超过上限 {gap}"))
    return reasons


def evidence_list(candidates: CandidateSet, cfg: dict[str, Any]) -> list[PairEvidence]:
    result: list[PairEvidence] = []
    for key, cand in sorted(candidates.candidates.items()):
        a, b = candidates.atoms[key[0]], candidates.atoms[key[1]]
        time_disjoint = a.period_max < b.period_min or b.period_max < a.period_min
        result.append(
            PairEvidence(
                left=key[0],
                right=key[1],
                scores=cand.scores,
                reviewed_same=cand.reviewed_same,
                comparison_id=cand.comparison_id or f"bridge:{key[0]}:{key[1]}",
                time_disjoint=time_disjoint,
            )
        )
    return result


@dataclass
class RecallAudit:
    strata: list[dict[str, Any]]
    reviewed_unlinked: int
    reviewed_unlinked_same: int
    warnings: list[str]
    recommended_sample_per_stratum: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reviewed_unlinked": self.reviewed_unlinked,
            "reviewed_unlinked_same": self.reviewed_unlinked_same,
            "recommended_sample_per_stratum": self.recommended_sample_per_stratum,
            "strata": self.strata,
            "warnings": self.warnings,
        }


def recall_audit(material: Material, candidates: CandidateSet, cfg: dict[str, Any], per_stratum: int = 30) -> RecallAudit:
    """候选之外的漏配风险审计。

    候选表只来自已登记比较。没有登记比较、但属性相容的记录对仍可能是同一人。
    本审计按（同部门、共享号码、姓名相同、出生桶相容）分层列出此类对，并建议
    在每层独立随机复核 per_stratum 对。若已有复核结论，用 rule-of-three 给出
    漏配率的粗略上限；否则明确标为未量化，不能假装区间已覆盖此风险。
    """
    linked = set(candidates.candidates)
    linked |= set(candidates.forbidden)
    rows: dict[str, list[tuple[str, str]]] = {}
    for left, right in itertools.combinations(sorted(candidates.atoms), 2):
        key = (left, right)
        if key in linked:
            continue
        a, b = candidates.atoms[left], candidates.atoms[right]
        if a.anchors and b.anchors and not (a.anchors & b.anchors):
            continue
        if a.births and b.births and not (a.births & b.births):
            continue
        shared_phone = bool(a.phones & b.phones) if (a.phones and b.phones) else False
        shared_name = bool(a.names & b.names) if (a.names and b.names) else False
        if not (shared_phone or shared_name):
            continue
        stratum = "|".join(
            [
                "same_dept" if (a.departments & b.departments) else "cross_dept",
                "same_phone" if shared_phone else "no_phone",
                "same_name" if shared_name else "no_name",
            ]
        )
        rows.setdefault(stratum, []).append(key)
    # 已复核对已经进入 comparisons；未进入候选集说明它们被禁止或已合并，
    # 因此这里没有可直接利用的标注。
    reviewed_unlinked = 0
    reviewed_unlinked_same = 0
    strata = []
    warnings: list[str] = []
    for stratum, pairs in sorted(rows.items()):
        opportunity = len(pairs)
        required = min(per_stratum, opportunity)
        upper_rate = 3.0 / required if required else None
        strata.append(
            {
                "stratum": stratum,
                "unlinked_candidate_pairs": opportunity,
                "recommended_sample": required,
                "missed_link_rate_upper_95": None if upper_rate is None else round(upper_rate, 4),
                "example_pairs": [list(pair) for pair in pairs[:3]],
            }
        )
    if not strata:
        status = "no_unlinked_compatible_pairs"
    else:
        status = "unquantified_sampling_required"
        warnings.append(
            "候选集之外可能存在漏配。当前没有针对未登记记录对的独立随机复核，"
            "覆盖区间不包含这一风险；请按 strata 补抽样后再解释区间。"
        )
    if reviewed_unlinked:
        status = "quantified"
    return RecallAudit(
        strata=strata,
        reviewed_unlinked=reviewed_unlinked,
        reviewed_unlinked_same=reviewed_unlinked_same,
        warnings=warnings,
        recommended_sample_per_stratum=per_stratum,
        status=status,
    )


def time_disjoint_map_to_dict(candidates: CandidateSet) -> dict[tuple[str, str], bool]:
    result: dict[tuple[str, str], bool] = {}
    for key in candidates.candidates:
        a, b = candidates.atoms[key[0]], candidates.atoms[key[1]]
        result[key] = a.period_max < b.period_min or b.period_max < a.period_min
    return result
