"""在身份约束下抽样一致的“关联世界”。

每个世界是对原子的一个划分。合法性由以下约束共同决定，任何不满足的划分
都不会进入样本，因此不会出现“各条边独立抽样”导致的身份矛盾：

* 团块内任意两条原子必须存在直接候选边（弱相似链不能闭包）；
* 团块内任意两条原子的校准概率必须达到 cohesion_min_prob；
* 出生桶必须相容；
* 可信 person 锚点必须相容；
* 时间跨度不得超过 max_period_gap；
* 团块大小不得超过 max_block_size。

小原子集精确枚举全部合法划分并按权重抽样（无近似）；较大连通分量用精确
条件分布的 Gibbs 抽样，烧入后按间隔取样本。权重是候选边的连接/断开似然
乘积。没有登记比较的原子对不进入权重，也不能被合并；由此产生的漏配风险
由候选审计单独报告，不会伪装成已覆盖的后验区间。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .candidates import CandidateSet


@dataclass
class World:
    draw: int
    blocks: tuple[tuple[str, ...], ...]
    log_weight: float
    exact: bool


def build_components(atoms: dict, candidates: dict[tuple[str, str], Any]) -> list[list[str]]:
    adj: dict[str, set[str]] = {atom_id: set() for atom_id in atoms}
    for left, right in candidates:
        adj[left].add(right)
        adj[right].add(left)
    seen: set[str] = set()
    components = []
    for atom_id in sorted(atoms):
        if atom_id in seen:
            continue
        stack = [atom_id]
        comp = []
        seen.add(atom_id)
        while stack:
            current = stack.pop()
            comp.append(current)
            for nxt in sorted(adj[current]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _bell_number(n: int) -> int:
    row = [1]
    for _ in range(n):
        new = [row[-1]]
        for value in row:
            new.append(value + new[-1])
        row = new
    return row[0]


def _partitions(n: int):
    """限制增长串枚举集合划分。"""
    labels = [0] * n

    def rec(index: int, maximum: int):
        if index == n:
            yield tuple(labels)
            return
        for value in range(maximum + 2):
            labels[index] = value
            yield from rec(index + 1, max(maximum, value))

    yield from rec(0, -1)


def _legal_block(block: list[str], atoms, edges: set[tuple[str, str]], probs, cfg) -> bool:
    if len(block) > (cfg["max_block_size"] or len(atoms)):
        return False
    if len(block) < 2:
        return True
    if cfg["max_period_gap"] is not None:
        period_min = min(atoms[a].period_min for a in block)
        period_max = max(atoms[a].period_max for a in block)
        if period_max - period_min > cfg["max_period_gap"]:
            return False
    births: set[str] | None = None
    anchors: set[str] | None = None
    for i, a in enumerate(block):
        for b in block[i + 1:]:
            key = (a, b) if a < b else (b, a)
            if key not in edges:
                return False
            if probs is not None and probs.get(key, 0.0) < cfg["cohesion_min_prob"]:
                return False
            if atoms[a].births and atoms[b].births and not (atoms[a].births & atoms[b].births):
                return False
            if atoms[a].anchors and atoms[b].anchors and atoms[a].anchors != atoms[b].anchors:
                return False
        a = block[i]
        if atoms[a].births:
            births = set(atoms[a].births) if births is None else births & atoms[a].births
        if atoms[a].anchors:
            anchors = set(atoms[a].anchors) if anchors is None else anchors & atoms[a].anchors
    if births is not None and not births:
        return False
    if anchors is not None and not anchors:
        return False
    return True


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def _partition_log_weight(blocks, probs, edges: set[tuple[str, str]]) -> float:
    block_of = {atom: index for index, block in enumerate(blocks) for atom in block}
    total = 0.0
    for key in edges:
        a, b = key
        if a not in block_of or b not in block_of:
            continue
        p = probs[key]
        total += math.log(p) if block_of[a] == block_of[b] else math.log1p(-p)
    return total


def _exact_component(component, atoms, candidates, probs, cfg, rng, n_samples, draw) -> list[World]:
    node_set = set(component)
    edges = {key for key in candidates if key[0] in node_set and key[1] in node_set}
    partitions: list[tuple[tuple[str, ...], ...]] = []
    log_weights: list[float] = []
    for labels in _partitions(len(component)):
        by_label: dict[int, list[str]] = {}
        for atom, label in zip(component, labels):
            by_label.setdefault(label, []).append(atom)
        blocks = tuple(tuple(sorted(members)) for _, members in sorted(by_label.items()))
        if not all(_legal_block(list(block), atoms, edges, probs, cfg) for block in blocks):
            continue
        partitions.append(blocks)
        log_weights.append(_partition_log_weight(blocks, probs, edges))
    if not partitions:
        return []
    peak = max(log_weights)
    weights = [math.exp(value - peak) for value in log_weights]
    chosen = rng.choices(range(len(partitions)), weights=weights, k=n_samples)
    return [
        World(draw=draw, blocks=partitions[index], log_weight=log_weights[index], exact=True)
        for index in chosen
    ]


def _gibbs_component(component, atoms, candidates, probs, cfg, rng, n_samples, draw,
                     burn_sweeps: int = 400, thin: int = 2) -> list[World]:
    node_set = set(component)
    edges = {key for key in candidates if key[0] in node_set and key[1] in node_set}
    edge_set = edges

    def logit_for(a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        return _logit(probs[key])

    # 贪心合法初始化：每个原子尝试并入已有团块，若合法则并入，否则单独成块。
    # 初始状态必须合法；否则烧入期间会把违反约束的划分也计入权重。
    state: list[list[str]] = []
    block_of: dict[str, int] = {}
    for atom in component:
        placed = False
        for index, block in enumerate(state):
            if _legal_block(sorted(block + [atom]), atoms, edge_set, probs, cfg):
                block.append(atom)
                block_of[atom] = index
                placed = True
                break
        if not placed:
            block_of[atom] = len(state)
            state.append([atom])

    def move(atom: str, destination_index: int | None) -> None:
        """把 atom 移到目标团块的索引；None 表示独立成新团块。"""
        current_index = block_of[atom]
        current = state[current_index]
        current.remove(atom)
        if destination_index is None:
            state.append([atom])
            block_of[atom] = len(state) - 1
        else:
            state[destination_index].append(atom)
            block_of[atom] = destination_index
        if not current:
            removed_index = current_index
            state.pop(removed_index)
            if removed_index == current_index:
                for index, block in enumerate(state):
                    for member in block:
                        block_of[member] = index

    def sweep() -> None:
        order = list(component)
        rng.shuffle(order)
        for atom in order:
            current_index = block_of[atom]
            current = state[current_index]
            others = [other for other in current if other != atom]
            stay_delta = 0.0
            options: list[list[str] | None] = [current]
            deltas: list[float] = [0.0]
            for other in others:
                stay_delta -= logit_for(atom, other)
            # stay_delta = -sum_{C\{atom}} logit。移动到团块 D 的变化为
            # sum_D logit - sum_C logit，因此 delta = stay_delta + sum_D logit。
            moves: list[int | None] = [current_index]  # 留在原团块
            for index, block in enumerate(state):
                if index == current_index:
                    continue
                merged = sorted(block + [atom])
                if not _legal_block(merged, atoms, edge_set, probs, cfg):
                    continue
                delta = stay_delta
                for other in block:
                    delta += logit_for(atom, other)
                moves.append(index)
                deltas.append(delta)
            if len(current) > 1 and _legal_block([atom], atoms, edge_set, probs, cfg):
                moves.append(None)
                deltas.append(stay_delta)
            if len(moves) == 1:
                continue
            peak = max(deltas)
            weights = [math.exp(value - peak) for value in deltas]
            chosen = rng.choices(range(len(moves)), weights=weights, k=1)[0]
            if chosen == 0:
                continue
            move(atom, moves[chosen])

    for _ in range(burn_sweeps):
        sweep()
    results = []
    for _ in range(n_samples):
        for _ in range(thin):
            sweep()
        blocks = tuple(tuple(sorted(block)) for block in state)
        results.append(World(draw=draw, blocks=blocks, log_weight=_partition_log_weight(blocks, probs, edges), exact=False))
    return results


def sample_worlds(candidates: CandidateSet, edge_probs_per_draw: list[dict[tuple[str, str], float]],
                  cfg: dict[str, Any]) -> list[World]:
    atoms = candidates.atoms
    components = build_components(atoms, candidates.candidates)
    all_worlds: list[World] = []
    for draw, edge_probs in enumerate(edge_probs_per_draw):
        probs = {key: min(max(edge_probs[key], 0.001), 0.999) for key in candidates.candidates}
        rng = random.Random(cfg["seed"] + 1000003 * draw)
        per_component: list[list[World]] = []
        for component in components:
            if len(component) <= cfg["exact_enum_max_atoms"] and _bell_number(len(component)) <= cfg["exact_enum_max_partitions"]:
                sampled = _exact_component(component, atoms, candidates.candidates, probs, cfg, rng, cfg["n_samples"], draw)
            else:
                sampled = _gibbs_component(
                    component, atoms, candidates.candidates, probs, cfg, rng, cfg["n_samples"], draw,
                    burn_sweeps=cfg["gibbs_burn_sweeps"], thin=cfg["gibbs_thin"],
                )
            per_component.append(sampled)
        for sample_index in range(cfg["n_samples"]):
            blocks: list[tuple[str, ...]] = []
            for sampled in per_component:
                if sample_index < len(sampled):
                    blocks.extend(sampled[sample_index].blocks)
            log_weight = sum(
                sampled[sample_index].log_weight for sampled in per_component if sample_index < len(sampled)
            )
            exact = bool(per_component) and all(
                sampled[sample_index].exact for sampled in per_component if sample_index < len(sampled)
            )
            all_worlds.append(World(draw=draw, blocks=tuple(sorted(blocks)), log_weight=log_weight, exact=exact))
    return all_worlds


def pair_marginals(worlds: list[World], pair_keys: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    counts = {key: 0 for key in pair_keys}
    for world in worlds:
        block_of = {atom: index for index, block in enumerate(world.blocks) for atom in block}
        for key in pair_keys:
            if key[0] in block_of and block_of[key[0]] == block_of.get(key[1]):
                counts[key] += 1
    total = len(worlds) or 1
    return {key: counts[key] / total for key in pair_keys}
