"""把已登记的比较分数校准为同人/异人证据。

比较分数本身不是概率：它只说明属性相似程度。本模块把分数映射为对数似然比
（LLR），在独立标注样本上做 L2 正则 logistic 校准，并把拟合参数的不确定性
随覆盖区间一起传播。

空值处理：任一字段缺失时，该字段不提供证据，贡献恰为 0；空值绝不被当作
“不相同”。缺失只降低分辨力，不降低判定为同一人的证据。

标注获取偏差：
* ``label_sampling="unknown"``：不假设标注如何抽到，仅用启发式截距平移使标注
  基率与样本基率一致，并明确该修正不可识别。
* ``label_sampling="stratified_random"``：按最高字段分分层，层内随机标注；
  用层内纳入率的倒数加权，使校准目标回到全部候选边。层内若只有单一类别，
  该层不能识别真实基率，报告会在 label_audit 中列出覆盖不足的层。
仅抽取高分候选的标注会造成选择性标签偏差，必须在复核台账中登记，不能当成
全量真值。补抽样方案见 :func:`label_audit`。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from .model import ConfigError

FIELDS = ("name_score", "phone_score", "birth_score")
FEATURE_NAMES = ("intercept",) + tuple(f"llr_{name}" for name in FIELDS)
PRIOR_WEIGHTS = [-1.2, 0.55, 0.55, 0.55]


@dataclass
class PairEvidence:
    left: str
    right: str
    scores: dict[str, float | None]
    reviewed_same: bool | None
    comparison_id: str
    time_disjoint: bool = False

    @property
    def max_score(self) -> float | None:
        values = [v for v in self.scores.values() if v is not None]
        return max(values) if values else None


def score_value(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} 必须是数值或空值")
    value = float(value)
    if math.isnan(value) or not (0.0 <= value <= 1.0):
        raise ConfigError(f"{field_name} 必须在 0 到 1 之间")
    return value


def to_logit(score: float | None) -> float | None:
    if score is None:
        return None
    score = min(max(score, 0.002), 0.998)
    return math.log(score / (1.0 - score))


def evidence_vector(name: float | None, phone: float | None, birth: float | None,
                    time_aware_phone: bool = False, time_disjoint: bool = False) -> list[float]:
    """特征向量：截距 + 各字段 LLR；缺失字段严格贡献 0。"""
    vector = [1.0]
    for field_name, value in zip(FIELDS, (name, phone, birth)):
        llr = to_logit(value)
        if llr is None:
            vector.append(0.0)
        elif time_aware_phone and field_name == "phone_score" and time_disjoint:
            # 跨期且号码不同：换号很常见，号码差异不作为异人证据。
            vector.append(max(llr, 0.0))
        else:
            vector.append(llr)
    return vector


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


def feature_vector(edge: PairEvidence, time_disjoint: bool = False, time_aware: bool = False) -> list[float]:
    return evidence_vector(
        edge.scores["name_score"], edge.scores["phone_score"], edge.scores["birth_score"],
        time_aware_phone=time_aware, time_disjoint=time_disjoint,
    )


@dataclass
class CalibrationFit:
    weights: list[float]
    labeled_edges: int
    positive: int
    negative: int
    cost: float
    source: str
    sampling: str
    warnings: list[str] = field(default_factory=list)
    label_scope: dict[str, Any] = field(default_factory=dict)

    def predict(self, feature: list[float]) -> float:
        return sigmoid(sum(w * x for w, x in zip(self.weights, feature)))

    @property
    def signature(self) -> int:
        """稳定签名（不使用 Python 的内置 hash，后者跨进程加盐）。"""
        parts = [self.source, self.sampling, str(self.labeled_edges), str(self.positive)]
        for w in self.weights:
            parts.append(f"{w:.9f}")
        return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def _design(rows: list[tuple[PairEvidence, bool]], cfg: dict[str, Any],
            all_edges: list[PairEvidence] | None = None) -> tuple[list[list[float]], list[int], list[float]]:
    """特征、标签与样本权重。

    stratified_random 时，标注层内的纳入率可能远低于 100%；用候选层总边数 /
    标注层边数作为权重，让校准目标回到全部候选，而不是只拟合被标注的子集。
    """
    strata_total: dict[str, int] = {}
    strata_labeled: dict[str, int] = {}
    for edge in all_edges or []:
        strata_total[_stratum(edge)] = strata_total.get(_stratum(edge), 0) + 1
        if edge.reviewed_same is not None:
            strata_labeled[_stratum(edge)] = strata_labeled.get(_stratum(edge), 0) + 1
    x, y, w = [], [], []
    for edge, time_disjoint in rows:
        x.append(feature_vector(edge, time_disjoint, cfg.get("time_aware_phone", False)))
        y.append(1 if edge.reviewed_same else 0)
        if cfg["label_sampling"] == "stratified_random":
            # 层内纳入率的倒数。若某层只有单一类别，权重只改变截距的尺度，
            # 不提供额外信息；此时退回 1.0，避免把全同层当作有信息。
            total = strata_total.get(_stratum(edge), 0)
            labeled = max(strata_labeled.get(_stratum(edge), 0), 1)
            w.append(max(total / labeled, 1.0))
        else:
            w.append(1.0)
    return x, y, w


def fit_calibration(edges: list[PairEvidence], cfg: dict[str, Any],
                    time_disjoint: dict[tuple[str, str], bool] | None = None) -> CalibrationFit:
    time_disjoint = time_disjoint or {}
    labeled = [e for e in edges if e.reviewed_same is not None]
    positives = sum(1 for e in labeled if e.reviewed_same)
    warnings: list[str] = []
    scope = _label_scope(edges, labeled, cfg)
    if cfg["label_sampling"] == "unknown" and labeled:
        warnings.append(
            "标注筛选机制未知；若标注只来自高分候选，校准会高估弱相似边的同人概率，"
            "覆盖区间可能偏窄。建议按 label_audit 的补抽样方案复核。"
        )
    if len(labeled) < cfg["calibration_min_labels"] or positives == 0 or positives == len(labeled):
        warnings.append("标注不足或类别不均衡，校准使用弱先验并在抽样中传播参数不确定性")
        return CalibrationFit(
            weights=list(PRIOR_WEIGHTS),
            labeled_edges=len(labeled),
            positive=positives,
            negative=len(labeled) - positives,
            cost=0.0,
            source="prior",
            sampling=cfg["label_sampling"],
            warnings=warnings,
            label_scope=scope,
        )
    rows = [(e, time_disjoint.get(_pair_key(e), False)) for e in labeled]
    x, y, sample_weight = _design(rows, cfg, edges)
    weights = [0.0] * len(FEATURE_NAMES)
    weights[0] = math.log(max(positives, 1) / max(len(labeled) - positives, 1))
    cost = _objective(x, y, sample_weight, weights, cfg["calibration_l2"])
    for _ in range(24):
        gradient = [0.0] * len(weights)
        hessian = [[0.0] * len(weights) for _ in weights]
        for row, target, weight in zip(x, y, sample_weight):
            p = sigmoid(sum(w * v for w, v in zip(weights, row)))
            residual = (p - target) * weight
            for i in range(len(weights)):
                gradient[i] += residual * row[i]
                for j in range(len(weights)):
                    hessian[i][j] += weight * p * (1 - p) * row[i] * row[j]
        for i in range(1, len(weights)):
            gradient[i] += cfg["calibration_l2"] * weights[i]
            hessian[i][i] += cfg["calibration_l2"]
        try:
            step = _solve(hessian, gradient)
        except ConfigError:
            break
        candidate = [w - s for w, s in zip(weights, step)]
        candidate_cost = _objective(x, y, sample_weight, candidate, cfg["calibration_l2"])
        if candidate_cost > cost - 1e-12:
            break
        weights, cost = candidate, candidate_cost
    if cfg["label_sampling"] == "unknown":
        # 未知纳入机制下，基率不可识别：做一次启发式截距平移，让标注上的平均
        # 预测概率等于标注阳性率。此修正被明确登记为不可识别。
        mean_p = sum(sigmoid(sum(w * v for w, v in zip(weights, row))) for row in x) / len(x)
        base_rate = positives / len(labeled)
        if 0 < mean_p < 1 and 0 < base_rate < 1:
            weights[0] += math.log(base_rate / (1 - base_rate)) - math.log(mean_p / (1 - mean_p))
        warnings.append("未知标注机制下对截距做了基率对齐；该修正不可识别，抽样区间不覆盖此偏差")
    if positives < 2:
        warnings.append("同人类标注少于 2 条，字段权重不稳定")
    return CalibrationFit(
        weights=weights,
        labeled_edges=len(labeled),
        positive=positives,
        negative=len(labeled) - positives,
        cost=cost,
        source="labeled",
        sampling=cfg["label_sampling"],
        warnings=warnings,
        label_scope=scope,
    )


def _pair_key(edge: PairEvidence) -> tuple[str, str]:
    return tuple(sorted((edge.left, edge.right)))


def _label_scope(edges: list[PairEvidence], labeled: list[PairEvidence], cfg: dict[str, Any]) -> dict[str, Any]:
    """描述标注样本覆盖了候选池的哪些层，生成补抽样建议。"""
    total_strata: dict[str, int] = {}
    labeled_strata: dict[str, int] = {}
    same: dict[str, int] = {}
    for edge in edges:
        key = _stratum(edge)
        total_strata[key] = total_strata.get(key, 0) + 1
    for edge in labeled:
        key = _stratum(edge)
        labeled_strata[key] = labeled_strata.get(key, 0) + 1
        same[key] = same.get(key, 0) + (1 if edge.reviewed_same else 0)
    under = []
    for key, count in sorted(total_strata.items()):
        got = labeled_strata.get(key, 0)
        if count and got / count < 0.05:
            under.append({"stratum": key, "candidate_edges": count, "labeled_edges": got})
    return {
        "sampling": cfg["label_sampling"],
        "labeled_edges": len(labeled),
        "strata": [
            {
                "stratum": key,
                "candidate_edges": total_strata[key],
                "labeled_edges": labeled_strata.get(key, 0),
                "labeled_same": same.get(key, 0),
                "inclusion_rate": round(labeled_strata.get(key, 0) / total_strata[key], 4),
            }
            for key in sorted(total_strata)
        ],
        "under_sampled": under,
    }


def _stratum(edge: PairEvidence) -> str:
    """按最高字段分分层；用于标注覆盖审计与纳入率加权。"""
    value = edge.max_score
    if value is None:
        return "none"
    if value >= 0.9:
        return "high"
    if value >= 0.7:
        return "mid"
    if value >= 0.4:
        return "low"
    return "weak"


def label_audit(edges: list[PairEvidence], cfg: dict[str, Any], min_per_stratum: int = 8) -> dict[str, Any]:
    """标注偏差说明与补抽样方案。

    若已有标注集中于高分候选，不能把它当作全量候选的代表样本。建议按最大字段
    分层，并在每个未充分覆盖的层中独立随机抽取（高分、低分、含空值的候选都要
    有）；抽到的边只用于校准，不直接当作硬身份约束。
    """
    scope = _label_scope(edges, [e for e in edges if e.reviewed_same is not None], cfg)
    plan = []
    for row in scope["strata"]:
        if row["labeled_edges"] >= row["candidate_edges"]:
            # 该层已被完整标注，没有可补的随机样本；若全是同一类，只能靠
            # 扩大候选池或改变比较器覆盖来获得对比。
            if row["labeled_same"] in (0, row["labeled_edges"]):
                plan.append({**row, "additional_labels_needed": 0,
                             "status": "saturated_single_class"})
            continue
        need = max(0, min_per_stratum - row["labeled_edges"])
        if need:
            plan.append({**row, "additional_labels_needed": need, "status": "under_covered"})
    warnings = []
    labeled_high = sum(row["labeled_edges"] for row in scope["strata"] if row["stratum"] == "high")
    total_labeled = sum(row["labeled_edges"] for row in scope["strata"])
    if total_labeled and labeled_high / total_labeled > 0.8:
        warnings.append(
            "现有标注超过八成来自高分候选；弱相似边与含空值边的同人率未知，"
            "校准结果不能直接外推。请按下表补抽样（层内独立随机抽样）后重新校准。"
        )
    return {
        "labeled_edges": total_labeled,
        "high_score_share": round(labeled_high / total_labeled, 4) if total_labeled else None,
        "selection_bias": "labels concentrated on high-score candidates" if warnings else "not detected",
        "warnings": warnings,
        "supplementary_sampling": plan,
        "method": "stratified random sampling by max(field score); weight by inverse inclusion rate",
    }


def _objective(x, y, sample_weight, weights, l2) -> float:
    total = 0.0
    for row, target, weight in zip(x, y, sample_weight):
        z = sum(w * v for w, v in zip(weights, row))
        if z >= 0:
            loss = z + math.log1p(math.exp(-z)) - target * z
        else:
            loss = math.log1p(math.exp(z)) - target * z
        total += weight * loss
    total += 0.5 * l2 * sum(w * w for w in weights[1:])
    return total


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ConfigError("校准矩阵退化")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def sampled_weights(fit: CalibrationFit, cfg: dict[str, Any], draw_index: int) -> list[float]:
    """校准参数后验的一组抽样。

    有标注时用近似标准误（按标注量收缩）加确定性扰动；无标注时从弱先验的
    较宽扰动中抽样。每个校准抽样都会重新计算整条世界分布，因此参数不确定
    性与身份不确定性一起传播，不是各边独立抖动。
    """
    if fit.source != "labeled":
        rng = _draw_rng(cfg["seed"], draw_index, 11)
        return [w + rng.gauss(0.0, 0.45 if i == 0 else 0.30) for i, w in enumerate(PRIOR_WEIGHTS)]
    rng = _draw_rng(cfg["seed"], draw_index, 17)
    scale = 1.0 / math.sqrt(max(fit.labeled_edges, 1))
    weights = list(fit.weights)
    weights[0] += rng.gauss(0.0, 0.35 + 1.5 * scale)
    for i in range(1, len(weights)):
        weights[i] += rng.gauss(0.0, 0.08 + 0.5 * scale)
    return weights


def _draw_rng(seed: int, draw_index: int, salt: int):
    """独立的确定性随机源，保证同一配置、同一抽样编号可复算。

    每次抽样所有参数一起扰动，随后整条世界分布重新计算，因此传播的是参数
    后验，而不是逐条边的独立抖动。
    """
    import random

    return random.Random((seed * 6364136223846793005 + draw_index * 1442695040888963407 + salt * 2654435761) & 0xFFFFFFFF)


@dataclass
class Weights:
    weights: list[float]

    def probability(self, feature: list[float]) -> float:
        return sigmoid(sum(w * x for w, x in zip(self.weights, feature)))
