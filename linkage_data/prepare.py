"""业务资料的内部规范化，以及桥接可见性/有效期的解释。

这里只使用资料中的本地字段，不进行任何外部身份查询。桥接的两种时间概念分开：

* 可见性（revoked_at）：撤销时刻 <= 当前版本时，桥接在本次分析不可见。
  版本小于撤销时刻时仍可见，因此旧版本始终可以复算。
* 有效期（valid_from/valid_to）：默认要求覆盖分析时期；可用配置
  bridge_period_scope="all" 做敏感性分析，把所有可见桥接都视为身份证据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .io import DataError
from .model import token


@dataclass(frozen=True)
class Record:
    id: str
    department: str
    period: int
    name: str | None
    phone: str | None
    birth: str | None


@dataclass(frozen=True)
class Bridge:
    id: str
    record: str
    person: str
    valid_from: int | None
    valid_to: int | None
    revoked_at: int | None

    def visible(self, version: int) -> bool:
        return self.revoked_at is None or self.revoked_at > version

    def covers(self, period: int) -> bool:
        if self.valid_from is not None and period < self.valid_from:
            return False
        if self.valid_to is not None and period >= self.valid_to:
            return False
        return True


@dataclass(frozen=True)
class Comparison:
    id: str
    left: str
    right: str
    scores: dict[str, float | None]
    reviewed_same: bool | None


@dataclass
class Material:
    records: dict[str, Record]
    population: list[dict[str, Any]]
    bridges: list[Bridge]
    comparisons: list[Comparison]
    visible_bridges: list[Bridge]
    inactive_bridges: list[Bridge]
    warnings: list[str] = field(default_factory=list)


def prepare(tables: dict[str, list[dict[str, Any]]], cfg: dict[str, Any]) -> Material:
    period = cfg["period"]
    records: dict[str, Record] = {}
    warnings: list[str] = []
    for row in tables["records"]:
        rid = _id(row, "records")
        department = _text(row.get("department"), "records.department")
        rec_period = _int(row.get("period"), "records.period")
        birth = token(row.get("birth_bucket"), "records.birth_bucket")
        records[rid] = Record(
            id=rid,
            department=department,
            period=rec_period,
            name=token(row.get("name_token"), "records.name_token"),
            phone=token(row.get("phone_token"), "records.phone_token"),
            birth=birth,
        )
    if not any(record.period == period for record in records.values()):
        warnings.append(f"分析时期 {period} 没有记录，覆盖统计为空")
    population = []
    for row in tables["population"]:
        pid = _id(row, "population")
        population.append(
            {
                "id": pid,
                "birth_bucket": token(row.get("birth_bucket"), "population.birth_bucket"),
                "area": token(row.get("area"), "population.area"),
            }
        )
    bridges: list[Bridge] = []
    for row in tables["bridges"]:
        bid = _id(row, "bridges")
        bridges.append(
            Bridge(
                id=bid,
                record=_text(row.get("record"), "bridges.record"),
                person=_text(row.get("person"), "bridges.person"),
                valid_from=_int(row.get("valid_from"), "bridges.valid_from", allow_none=True),
                valid_to=_int(row.get("valid_to"), "bridges.valid_to", allow_none=True),
                revoked_at=_int(row.get("revoked_at"), "bridges.revoked_at", allow_none=True),
            )
        )
    comparisons: list[Comparison] = []
    seen_pairs: dict[tuple[str, str], str] = {}
    for row in tables["comparisons"]:
        cid = _id(row, "comparisons")
        left = _text(row.get("left"), "comparisons.left")
        right = _text(row.get("right"), "comparisons.right")
        if left == right:
            raise DataError(f"comparison {cid} 的两侧是同一记录")
        if left not in records or right not in records:
            raise DataError(f"comparison {cid} 引用不存在的记录")
        scores = {}
        for name in ("name_score", "phone_score", "birth_score"):
            value = row.get(name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise DataError(f"{cid}.{name} 必须是数值或空值")
                value = float(value)
                if not (0.0 <= value <= 1.0):
                    raise DataError(f"{cid}.{name} 必须在 0 到 1 之间")
            scores[name] = value
        reviewed = row.get("reviewed_same")
        if reviewed is not None and not isinstance(reviewed, bool):
            raise DataError(f"{cid}.reviewed_same 必须是布尔值或空值")
        canonical = tuple(sorted((left, right)))
        if canonical in seen_pairs:
            warnings.append(f"comparison {cid} 与 {seen_pairs[canonical]} 是同一记录对，保留先登记者")
            continue
        seen_pairs[canonical] = cid
        comparisons.append(Comparison(id=cid, left=left, right=right, scores=scores, reviewed_same=reviewed))
    version = cfg["version"]
    scope = cfg.get("bridge_period_scope", "analysis")
    visible, inactive = [], []
    for bridge in bridges:
        if not bridge.visible(version):
            continue
        record = records.get(bridge.record)
        # 桥接的有效期按它所关联记录的时期判断：桥接断言的是该记录在该时期
        # 属于该 person。分析时期是否被覆盖由记录本身是否在分析时期决定。
        if scope == "all" or record is None or bridge.covers(record.period):
            visible.append(bridge)
        else:
            inactive.append(bridge)
    if inactive:
        warnings.append(
            f"{len(inactive)} 条桥接未覆盖所关联记录的时期，按 bridge_period_scope={scope} 不作为本次身份证据"
        )
    revoked_later = [bridge.id for bridge in bridges if bridge.visible(version) is False and (bridge.revoked_at or 0) > version]
    if revoked_later:
        warnings.append(f"{len(revoked_later)} 条桥接在版本 {version} 已撤销")
    return Material(
        records=records,
        population=population,
        bridges=bridges,
        comparisons=comparisons,
        visible_bridges=visible,
        inactive_bridges=inactive,
        warnings=warnings,
    )


def _id(row: dict[str, Any], table: str) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value:
        raise DataError(f"{table}.id 必须是非空字符串")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataError(f"{name} 必须是非空字符串")
    return value


def _int(value: Any, name: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"{name} 必须是整数")
    return value
