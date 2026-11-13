"""分析配置与业务字段的本地规范化。

本模块不读取外部身份服务。所有时间（period、版本、撤销时刻）都是本地资料
中的整数刻度；valid_from 包含、valid_to 不包含。
"""
from __future__ import annotations

import copy
import json
from typing import Any

CODE_VERSION = "0.3.0"
ANALYSIS_SCHEMA = "linkage-analysis/1"


class ConfigError(ValueError):
    """配置或业务字段不能被本分析模型解释。"""


DEFAULT_CONFIG: dict[str, Any] = {
    # 分析时期（精确匹配）。只有该时期的记录进入覆盖统计；其他时期记录仍可用于
    # 身份关联，因为它们提供了同一人的跨期证据。
    "period": None,
    # 资料版本。撤销时刻 revoked_at <= version 的桥接在本次版本不可见；版本小于
    # 撤销时刻时仍可见，因此旧版本结果不受后续撤销影响。
    "version": 1,
    "seed": 20261114,
    # 后验计算量：小连通分量精确枚举；较大分量用 Gibbs。每个校准抽样都会
    # 重新计算全部候选边的概率和整条世界分布。
    "n_samples": 4000,
    "calibration_draws": 6,
    "calibration_l2": 1.0,
    "calibration_min_labels": 20,
    # 标注筛选设计。unknown 表示不假设标注如何入选；stratified_random 表示
    # 层内随机标注，可用层内纳入率的倒数加权校准。
    "label_sampling": "unknown",
    # 同一身份团块内的任意两条原子必须存在直接比较边且达到该阈值；
    # 禁止用弱相似链闭包。这是模型设置，不是资料事实；改变它应重跑分析。
    # reviewed_same 只用于校准，不当作可信身份事实；把标注同时当训练标签和
    # 硬约束会循环论证，因此不提供硬约束模式。
    "cohesion_min_prob": 0.25,
    # 时间约束：None 表示不限制跨期合并；整数表示合并团块内记录 period 的
    # 最大跨度。桥接有效期在 prepare 阶段处理。
    "max_period_gap": None,
    "max_block_size": None,
    # 跨期且号码不同时，号码差异不当作异人证据（换号是常见情形）。
    "time_aware_phone": True,
    # 桥接有效期是否必须覆盖分析时期。all 表示所有未撤销桥接都作为身份证据，
    # 仅用于敏感性分析。
    "bridge_period_scope": "analysis",
    # 精确枚举阈值：原子数不超过该值且划分数量可控时枚举全部合法划分。
    "exact_enum_max_atoms": 9,
    "exact_enum_max_partitions": 20000,
    # 单个 Gibbs 链的烧入与抽样间隔。较大连通分量只在烧入后计样本。
    "gibbs_burn_sweeps": 400,
    "gibbs_thin": 2,
    # 覆盖区间的抽样不确定性门槛：有效样本低于该值时报告提醒加宽/重跑。
    "min_interval_support": 200,
    # 复核队列长度与“存疑”区间。
    "review_limit": 40,
    "review_low": 0.10,
    "review_high": 0.90,
    # 复核队列使用的统计量权重（覆盖人数、部门重叠、未覆盖百分点）。
    "review_weights": {
        "beneficiaries_count": 1.0,
        "department_overlap_total": 1.0,
        "uncovered_fraction": 100.0,
    },
    # 业务总体固定名单。未锚定团块是否属于名单按部分识别处理，不强行赋予概率。
    "roster_assignment": "bounds",
}

# 运行配置可以覆盖的键。未登记的键必须拒绝，避免静默改变模型含义。
CONFIG_KEYS = set(DEFAULT_CONFIG)


def make_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if raw:
        unknown = set(raw) - CONFIG_KEYS
        if unknown:
            raise ConfigError(f"未知配置项: {', '.join(sorted(unknown))}")
        for key, value in raw.items():
            if key == "review_weights":
                if not isinstance(value, dict):
                    raise ConfigError("review_weights 必须是对象")
                merged = dict(cfg["review_weights"])
                for k, v in value.items():
                    if k not in merged:
                        raise ConfigError(f"未知复核权重: {k}")
                    merged[k] = _number(v, f"review_weights.{k}")
                cfg[key] = merged
            else:
                cfg[key] = value
    cfg["period"] = _integer(cfg["period"], "period", allow_none=True)
    if cfg["period"] is None:
        raise ConfigError("分析必须指定 period")
    cfg["version"] = _integer(cfg["version"], "version", minimum=0)
    cfg["seed"] = _integer(cfg["seed"], "seed", minimum=0)
    cfg["n_samples"] = _integer(cfg["n_samples"], "n_samples", minimum=10)
    cfg["exact_enum_max_atoms"] = _integer(cfg["exact_enum_max_atoms"], "exact_enum_max_atoms", minimum=2, maximum=11)
    cfg["exact_enum_max_partitions"] = _integer(
        cfg["exact_enum_max_partitions"], "exact_enum_max_partitions", minimum=1
    )
    cfg["calibration_draws"] = _integer(cfg["calibration_draws"], "calibration_draws", minimum=1, maximum=32)
    cfg["gibbs_burn_sweeps"] = _integer(cfg["gibbs_burn_sweeps"], "gibbs_burn_sweeps", minimum=20)
    cfg["gibbs_thin"] = _integer(cfg["gibbs_thin"], "gibbs_thin", minimum=1)
    cfg["min_interval_support"] = _integer(cfg["min_interval_support"], "min_interval_support", minimum=20)
    cfg["calibration_l2"] = _number(cfg["calibration_l2"], "calibration_l2", minimum=0.0)
    cfg["cohesion_min_prob"] = _number(cfg["cohesion_min_prob"], "cohesion_min_prob", minimum=0.0, maximum=1.0)
    cfg["review_low"] = _number(cfg["review_low"], "review_low", minimum=0.0, maximum=1.0)
    cfg["review_high"] = _number(cfg["review_high"], "review_high", minimum=0.0, maximum=1.0)
    if cfg["review_low"] >= cfg["review_high"]:
        raise ConfigError("review_low 必须小于 review_high")
    cfg["review_limit"] = _integer(cfg["review_limit"], "review_limit", minimum=1, maximum=500)
    if cfg["label_sampling"] not in ("unknown", "stratified_random"):
        raise ConfigError("label_sampling 必须是 unknown 或 stratified_random")
    if cfg["bridge_period_scope"] not in ("analysis", "all"):
        raise ConfigError("bridge_period_scope 必须是 analysis 或 all")
    if not isinstance(cfg["time_aware_phone"], bool):
        raise ConfigError("time_aware_phone 必须是布尔值")
    if cfg["max_period_gap"] is not None:
        cfg["max_period_gap"] = _integer(cfg["max_period_gap"], "max_period_gap", minimum=0)
    if cfg["max_block_size"] is not None:
        cfg["max_block_size"] = _integer(cfg["max_block_size"], "max_block_size", minimum=2)
    if cfg["roster_assignment"] != "bounds":
        raise ConfigError("roster_assignment 当前只支持 bounds")
    return cfg


def config_for_output(cfg: dict[str, Any]) -> dict[str, Any]:
    """输出配置时去掉与资料无关的字段，保持稳定排序。"""
    return json.loads(json.dumps(cfg, ensure_ascii=False, sort_keys=True))


def _number(value, name, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} 必须是数值")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} 不得小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} 不得大于 {maximum}")
    return value


def _integer(value, name, minimum=None, maximum=None, allow_none=False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} 必须是整数")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} 不得小于 {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} 不得大于 {maximum}")
    return value


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def token(value: Any, field: str) -> str | None:
    if is_missing(value):
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{field} 必须是字符串或空值")
    return value.strip()
