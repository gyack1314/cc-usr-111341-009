"""分析编排：校准、约束抽样、覆盖统计、复核队列与版本快照。

一次分析对应一个不可变的资料版本。快照只保存输入摘要、配置、校准依据、
桥接状态、冲突台账和抽样摘要；用快照配置重跑必须得到相同摘要。后续撤销不会
改写旧快照，旧版本也因此仍可复算。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import calibration
from .candidates import CandidateSet, build, evidence_list, recall_audit, time_disjoint_map_to_dict
from .coverage import pair_summary, summarize
from .model import ANALYSIS_SCHEMA, CODE_VERSION, config_for_output, make_config
from .prepare import Material, prepare
from .review import ReviewPlanner
from .worlds import pair_marginals, sample_worlds


@dataclass
class Analysis:
    config: dict[str, Any]
    material: Material
    candidates: CandidateSet
    calibration: calibration.CalibrationFit
    sampled_weights: list[list[float]]
    worlds: list
    marginals: dict[tuple[str, str], float]
    coverage: Any
    recall: Any
    review_items: list
    label_audit: dict[str, Any]
    metrics_by_world: list
    snapshot: dict[str, Any]


def run(tables: dict[str, list[dict[str, Any]]], raw_config: dict[str, Any] | None = None,
        manifest_digest: str | None = None) -> Analysis:
    cfg = make_config(raw_config)
    material = prepare(tables, cfg)
    candidates = build(material, cfg)
    edges = evidence_list(candidates, cfg)
    fit = calibration.fit_calibration(edges, cfg, time_disjoint_map_to_dict(candidates))
    draws = [calibration.sampled_weights(fit, cfg, index) for index in range(cfg["calibration_draws"])]
    edge_probs_per_draw = []
    for weights in draws:
        probs = {}
        for edge, time_disjoint in ((e, e.time_disjoint) for e in edges):
            feature = calibration.feature_vector(edge, time_disjoint, cfg.get("time_aware_phone", False))
            probs[(edge.left, edge.right)] = calibration.sigmoid(
                sum(w * v for w, v in zip(weights, feature))
            )
        edge_probs_per_draw.append(probs)
    worlds = sample_worlds(candidates, edge_probs_per_draw, cfg)
    pair_keys = list(candidates.candidates)
    marginals = pair_marginals(worlds, pair_keys)
    coverage = summarize(worlds, candidates, material, cfg)
    recall = recall_audit(material, candidates, cfg)
    audit = calibration.label_audit(edges, cfg)
    planner = ReviewPlanner(candidates, material, cfg)
    items = planner.plan(worlds, marginals, coverage.world_metrics, recall.to_dict())
    snapshot = _snapshot(cfg, material, candidates, fit, draws, worlds, coverage, recall, audit, items, manifest_digest)
    return Analysis(
        config=cfg,
        material=material,
        candidates=candidates,
        calibration=fit,
        sampled_weights=draws,
        worlds=worlds,
        marginals=marginals,
        coverage=coverage,
        recall=recall,
        review_items=items,
        label_audit=audit,
        metrics_by_world=coverage.world_metrics,
        snapshot=snapshot,
    )


def _snapshot(cfg, material, candidates, fit, draws, worlds, coverage, recall, audit, items, manifest_digest) -> dict[str, Any]:
    return {
        "schema": ANALYSIS_SCHEMA,
        "code_version": CODE_VERSION,
        "manifest_digest": manifest_digest,
        "config": config_for_output(cfg),
        "bridge_visibility": {
            "visible": [b.id for b in material.visible_bridges],
            "not_covering_period": [b.id for b in material.inactive_bridges],
            "revoked_in_version": [
                b.id for b in material.bridges if not b.visible(cfg["version"])
            ],
        },
        "bridge_states": candidates.bridge_states,
        "duplicate_links": candidates.duplicate_links,
        "conflicts": candidates.conflicts,
        "calibration": {
            "source": fit.source,
            "sampling": fit.sampling,
            "weights": [round(w, 6) for w in fit.weights],
            "labeled_edges": fit.labeled_edges,
            "positive": fit.positive,
            "negative": fit.negative,
            "cost": round(fit.cost, 6),
            "signature": fit.signature,
            "warnings": fit.warnings,
            "label_scope": fit.label_scope,
        },
        "sampling": {
            "worlds": len(worlds),
            "independent_draws": len({w.draw for w in worlds}),
            "all_exact": all(w.exact for w in worlds) if worlds else True,
            "seed": cfg["seed"],
        },
        "worlds_digest": _worlds_digest(worlds),
        "review_queue_digest": _items_digest(items),
    }


def _worlds_digest(worlds) -> str:
    encoded = json.dumps([[list(block) for block in world.blocks] for world in worlds], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _items_digest(items) -> str:
    encoded = json.dumps([item.to_dict() for item in items], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def report(analysis: Analysis) -> dict[str, Any]:
    """面向业务的 JSON 报告。"""
    cfg = analysis.config
    fit = analysis.calibration
    coverage = analysis.coverage
    warnings = list(analysis.material.warnings)
    warnings.extend(fit.warnings)
    warnings.extend(analysis.candidates.warnings)
    warnings.extend(coverage.warnings)
    warnings.extend(analysis.recall.warnings)
    forbidden_rows = []
    for key, reasons in sorted(analysis.candidates.forbidden.items()):
        forbidden_rows.append(
            {
                "left": key[0],
                "right": key[1],
                "reasons": [{"reason": r.reason, "detail": r.detail} for r in reasons],
            }
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "code_version": CODE_VERSION,
        "period": cfg["period"],
        "version": cfg["version"],
        "coverage": {
            "representative_association": coverage.representative,
            "posterior_distribution": coverage.distribution,
            "partial_identification": coverage.partial_identification,
            "metric_definitions": {
                "beneficiaries_count": "分析时期内至少出现在一个部门的身份团块数",
                "department_overlap_people": "分析时期内出现在两个及以上部门的身份团块数",
                "department_overlap_pairs": "同一身份团块在分析时期内覆盖的部门对总数",
                "uncovered_fraction_low": "未锚定团块全部计入覆盖时的未覆盖比例下界",
                "uncovered_fraction_high": "未锚定团块完全不覆盖名单时的未覆盖比例上界",
            },
        },
        "calibration": {
            "source": fit.source,
            "sampling": fit.sampling,
            "labeled_edges": fit.labeled_edges,
            "positive": fit.positive,
            "negative": fit.negative,
            "weights": [round(w, 6) for w in fit.weights],
            "feature_names": list(calibration.FEATURE_NAMES),
            "signature": fit.signature,
            "note": "比较分数不是概率；上表是从分数到同人证据的校准参数，抽样中重新计算整条世界分布。",
            "label_audit": analysis.label_audit,
            "warnings": fit.warnings,
        },
        "candidates": {
            "atoms": len(analysis.candidates.atoms),
            "candidate_edges": len(analysis.candidates.candidates),
            "forbidden_edges": forbidden_rows,
            "conflicts": analysis.candidates.conflicts,
            "duplicate_links": analysis.candidates.duplicate_links,
            "bridge_states": analysis.candidates.bridge_states,
            "pair_marginals": pair_summary(analysis.candidates, analysis.marginals),
            "recall_audit": analysis.recall.to_dict(),
        },
        "review_queue": [item.to_dict() for item in analysis.review_items],
        "bridge_visibility": analysis.snapshot["bridge_visibility"],
        "warnings": _dedupe(warnings),
        "snapshot": {
            "schema": ANALYSIS_SCHEMA,
            "code_version": CODE_VERSION,
            "manifest_digest": analysis.snapshot["manifest_digest"],
            "config_hash": hashlib.sha256(
                json.dumps(analysis.snapshot["config"], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "calibration_signature": fit.signature,
            "worlds_digest": analysis.snapshot["worlds_digest"],
            "review_queue_digest": analysis.snapshot["review_queue_digest"],
            "sampling": analysis.snapshot["sampling"],
        },
    }


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def save_snapshot(analysis: Analysis, directory: str | Path, label: str | None = None) -> Path:
    """保存不可变快照。文件名包含版本与签名，后续撤销不会覆盖旧文件。

    只保存“依据”和“配置”，不保存完整报告：用快照配置重跑必须得到相同摘要，
    这样快照本身不能成为无法追溯的既成事实。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(analysis.snapshot)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    name = label or f"v{analysis.config['version']}-p{analysis.config['period']}-{analysis.calibration.signature}-{digest}"
    path = directory / f"{name}.json"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != encoded:
            raise ValueError(f"快照 {path} 已存在且内容不同；拒绝覆盖版本化结果")
    path.write_text(encoded, encoding="utf-8")
    return path


def verify_snapshot(path: str | Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """用快照中的配置对当前资料重算，并逐项比较摘要。

    旧版本快照不会因为后续撤销而失效：只要资料文件与快照登记的内容摘要一致，
    用同一配置重跑必须得到相同的校准签名、世界摘要和复核队列摘要。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("快照缺少配置，不能复算")
    analysis = run(tables, config, manifest_digest=payload.get("manifest_digest"))
    recomputed = {
        "calibration_signature": analysis.calibration.signature,
        "worlds_digest": analysis.snapshot["worlds_digest"],
        "review_queue_digest": analysis.snapshot["review_queue_digest"],
    }
    stored = {
        "calibration_signature": payload.get("calibration", {}).get("signature"),
        "worlds_digest": payload.get("worlds_digest"),
        "review_queue_digest": payload.get("review_queue_digest"),
    }
    mismatches = [key for key in stored if stored[key] != recomputed[key]]
    return {
        "snapshot": str(path),
        "matches": not mismatches,
        "mismatches": mismatches,
        "stored": stored,
        "recomputed": recomputed,
    }


def sensitivity(tables: dict[str, list[dict[str, Any]]], raw_config: dict[str, Any] | None = None,
                cohesion_values: tuple[float, ...] = (0.15, 0.25, 0.35),
                bridge_scopes: tuple[str, ...] = ("analysis", "all")) -> dict[str, Any]:
    """模型设置敏感性：同一资料在不同内聚阈值/桥接范围下的覆盖区间。

    这不是后验的一部分，而是告诉业务“结论有多少依赖模型设置”。如果某个设置下
    覆盖数字明显移动，就应把该设置写进报告假设，而不是只报默认值。
    """
    raw = dict(raw_config or {})
    variants = []
    for cohesion in cohesion_values:
        for scope in bridge_scopes:
            config = dict(raw)
            config["cohesion_min_prob"] = cohesion
            config["bridge_period_scope"] = scope
            analysis = run(tables, config)
            dist = analysis.coverage.distribution
            variants.append(
                {
                    "cohesion_min_prob": cohesion,
                    "bridge_period_scope": scope,
                    "beneficiaries_count": dist["beneficiaries_count"],
                    "department_overlap_people": dist["department_overlap_people"],
                    "uncovered_fraction_low": dist.get("uncovered_fraction_low", {}),
                    "uncovered_fraction_high": dist.get("uncovered_fraction_high", {}),
                    "calibration_source": analysis.calibration.source,
                }
            )
    lows = [v["beneficiaries_count"]["p05"] for v in variants]
    highs = [v["beneficiaries_count"]["p95"] for v in variants]
    return {
        "note": "范围同时包含后验抽样与模型设置敏感性；代表性区间仍在 coverage.posterior_distribution。",
        "beneficiaries_count_across_settings": {
            "min_p05": min(lows) if lows else None,
            "max_p95": max(highs) if highs else None,
        },
        "variants": variants,
    }
