"""生成确定性合成场景资料包（本地、可复算）。

每个场景一个目录，内含 manifest.json 与四个 JSONL。资历资料全部是合成 token。
运行：python tools/build_scenarios.py scenarios
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FIELDS = {
    "population": ["id", "birth_bucket", "area"],
    "records": ["id", "department", "period", "name_token", "phone_token", "birth_bucket"],
    "bridges": ["id", "record", "person", "valid_from", "valid_to", "revoked_at"],
    "comparisons": ["id", "left", "right", "name_score", "phone_score", "birth_score", "reviewed_same"],
}


def record(rid, dept, period, name, phone, birth):
    return {"id": rid, "department": dept, "period": period, "name_token": name, "phone_token": phone, "birth_bucket": birth}


def bridge(bid, rec, person, start, end, revoked=None):
    return {"id": bid, "record": rec, "person": person, "valid_from": start, "valid_to": end, "revoked_at": revoked}


def comp(cid, left, right, name, phone, birth, reviewed=None):
    return {"id": cid, "left": left, "right": right, "name_score": name, "phone_score": phone,
            "birth_score": birth, "reviewed_same": reviewed}


def scenario(name, description, records, population, bridges, comparisons):
    return {
        "name": name,
        "description": description,
        "tables": {
            "records": records,
            "population": population,
            "bridges": bridges,
            "comparisons": comparisons,
        },
    }


def build() -> list[dict]:
    scenarios = []

    # 1. 改号：同一人跨部门，姓名/出生相同，号码不同；不应因为号码差异被拆成两人。
    scenarios.append(scenario(
        "phone-change",
        "同一人换号后跨部门出现；号码差异不提供异人证据。",
        records=[
            record("r1", "A", 1, "N1", "P1", "old"),
            record("r2", "B", 1, "N1", "P2", "old"),
            record("r3", "A", 1, "N2", None, "young"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "young", "area": "B"},
        ],
        bridges=[bridge("b1", "r1", "u1", 1, 2)],
        comparisons=[
            comp("c1", "r1", "r2", 1.0, 0.0, 1.0),
            comp("c2", "r1", "r3", 0.0, None, 0.0),
            comp("c3", "r2", "r3", 0.0, None, 0.0),
        ],
    ))

    # 2. 同名不同人：姓名与出生相同、号码不同、被不同可信 person 锚定；必须不能合并。
    scenarios.append(scenario(
        "same-name-different-person",
        "同名同生日但由可信桥接锚定到不同 person，互斥约束禁止合并。",
        records=[
            record("r1", "A", 1, "N1", "P1", "old"),
            record("r2", "B", 1, "N1", "P2", "old"),
            record("r3", "A", 1, "N1", "P3", "old"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "old", "area": "B"},
            {"id": "u3", "birth_bucket": "old", "area": "C"},
        ],
        bridges=[bridge("b1", "r1", "u1", 1, 2), bridge("b2", "r2", "u2", 1, 2)],
        comparisons=[
            comp("c1", "r1", "r2", 1.0, 0.0, 1.0),
            comp("c2", "r2", "r3", 1.0, 0.0, 1.0),
        ],
    ))

    # 3. 多次上报：同部门内重复上报，覆盖统计不能重复计数。
    scenarios.append(scenario(
        "duplicate-submission",
        "同部门同一人多次上报（同姓名、同号码、同出生桶），覆盖只计一次。",
        records=[
            record("r1", "A", 1, "N1", "P1", "old"),
            record("r2", "A", 1, "N1", "P1", "old"),
            record("r3", "A", 1, "N1", "P1", "old"),
            record("r4", "B", 1, "N1", "P1", "old"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "young", "area": "B"},
        ],
        bridges=[bridge("b1", "r1", "u1", 1, 2)],
        comparisons=[comp("c1", "r4", "r1", 1.0, 1.0, 1.0)],
    ))

    # 4. 相似链冲突：a-b、b-c 有相似证据，a-c 没有登记比较且属性相容。
    # 并查集会闭包成 a-b-c 一团；本模型要求团块内任意两条都有直接候选边，
    # 因此只能二选一，并把 a-c 列入候选之外的漏配审计。
    scenarios.append(scenario(
        "weak-chain",
        "弱相似链 a-b-c：a-c 无比较登记；团块内聚约束禁止链式传递，a-c 进入漏配审计。",
        records=[
            record("r1", "A", 1, "N1", None, "old"),
            record("r2", "B", 1, "N1", None, "old"),
            record("r3", "C", 1, "N1", None, "old"),
            record("r4", "C", 1, "N9", "P9", "young"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "old", "area": "B"},
            {"id": "u3", "birth_bucket": "old", "area": "C"},
        ],
        bridges=[],
        comparisons=[
            comp("c1", "r1", "r2", 0.72, None, 0.73),
            comp("c2", "r2", "r3", 0.71, None, 0.72),
            comp("c3", "r1", "r4", 0.1, 0.05, 0.2),
        ],
    ))

    # 4b. 锚定冲突：a、c 分别锚定到不同 person，b 与两者都相似。
    # 互斥约束使 b 最多并入一侧；闭包会同时违反两个锚点。
    scenarios.append(scenario(
        "anchor-conflict",
        "相似边 a-b、b-c 均强，但 a、c 锚定不同可信 person；链式闭包违反互斥约束。",
        records=[
            record("r1", "A", 1, "N1", "P1", "old"),
            record("r2", "B", 1, "N1", "P2", "old"),
            record("r3", "C", 1, "N1", "P3", "old"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "old", "area": "C"},
            {"id": "u3", "birth_bucket": "old", "area": "D"},
        ],
        bridges=[bridge("b1", "r1", "u1", 1, 2), bridge("b2", "r3", "u2", 1, 2)],
        comparisons=[
            comp("c1", "r1", "r2", 0.9, 0.8, 0.9),
            comp("c2", "r2", "r3", 0.9, 0.8, 0.9),
            comp("c3", "r1", "r3", 0.9, 0.8, 0.9),
        ],
    ))

    # 5. 可信关系撤销：同一份资料在 version 1 可见、version 2 撤销，旧版结果仍可复算。
    scenarios.append(scenario(
        "bridge-revocation",
        "可信桥接在版本 2 撤销；版本 1 的结果与依据可复算，版本 2 重新分析。",
        records=[
            record("r1", "A", 1, "N1", "P1", "old"),
            record("r2", "B", 1, "N1", "P2", "old"),
            record("r3", "A", 1, "N3", "P3", "young"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "old", "area": "B"},
            {"id": "u3", "birth_bucket": "young", "area": "A"},
        ],
        bridges=[
            bridge("b1", "r1", "u1", 1, 3),
            bridge("b2", "r2", "u1", 1, 3, revoked=2),
        ],
        comparisons=[
            comp("c1", "r1", "r2", 1.0, 0.0, 1.0),
        ],
    ))

    # 6. 标注校准与选择性标注偏差：分数与真实同人关系有正相关，但标注集中在高分候选。
    scenarios.append(scenario(
        "calibration-bias",
        "含 reviewed_same 标注，且标注只在最高分层；说明选择偏差并给出补抽样方案。",
        records=[],
        population=[],
        bridges=[],
        comparisons=[],
    ))
    recs, pops, cmps = [], [], []
    for i in range(1, 57):
        dept = "A" if i % 2 else "B"
        recs.append(record(f"r{i}", dept, 1, f"N{i}", f"P{i}", "old"))
        pops.append({"id": f"u{i}", "birth_bucket": "old", "area": dept})
    # 独立的高分同人对 10 对（号码一致、复核为同人）与高分异人对 10 对
    # （号码不同、复核为异人）。使用不相交的记录对，避免三条记录缺少两两
    # 比较造成的结构性拆分。
    index = 1
    for _ in range(10):
        cmps.append(comp(f"ch{index}", f"r{2*index-1}", f"r{2*index}", 0.95, 0.9, 0.95, reviewed=True))
        index += 1
    for _ in range(10):
        cmps.append(comp(f"ch{index}", f"r{2*index-1}", f"r{2*index}", 0.95, 0.0, 0.95, reviewed=False))
        index += 1
    # 中间分层：4 对没有标注。
    for j in range(4):
        a, b = 41 + 2*j, 42 + 2*j
        cmps.append(comp(f"cm{a}", f"r{a}", f"r{b}", 0.75, 0.2, 0.8))
    # 低分候选 4 对复核为异人，使用独立记录。
    for j, a in enumerate([49, 51, 53, 55]):
        cmps.append(comp(f"cl{a}", f"r{a}", f"r{a+1}", 0.35, 0.1, 0.4, reviewed=False))
    scenarios[-1]["tables"] = {"records": recs, "population": pops, "bridges": [], "comparisons": cmps}

    # 7. 同名不同人 + 改号 + 多次上报的组合，含低置信边与漏配审计。
    scenarios.append(scenario(
        "mixed-ops",
        "组合场景：同名不同人、改号、多次上报、弱链与未登记比较的漏配风险并存。",
        records=[
            record("r1", "A", 1, "N1", "P1", "old"),
            record("r2", "A", 1, "N1", "P1", "old"),   # 重复上报
            record("r3", "B", 1, "N1", "P2", "old"),   # 改号
            record("r4", "C", 1, "N1", "P4", "old"),   # 同名不同人（无锚）
            record("r5", "A", 2, "N2", None, "young"),
            record("r6", "B", 3, "N2", "P6", "young"),
        ],
        population=[
            {"id": "u1", "birth_bucket": "old", "area": "A"},
            {"id": "u2", "birth_bucket": "old", "area": "C"},
            {"id": "u3", "birth_bucket": "young", "area": "B"},
        ],
        bridges=[bridge("b1", "r1", "u1", 1, 4)],
        comparisons=[
            comp("c1", "r1", "r3", 1.0, 0.0, 1.0),
            comp("c2", "r1", "r4", 1.0, 0.0, 1.0),
            comp("c3", "r5", "r6", 0.9, 0.0, 1.0),
        ],
    ))
    return scenarios


def write(root: Path) -> None:
    for item in build():
        directory = root / item["name"]
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": 1, "name": item["name"], "description": item["description"], "tables": {}}
        for table, fields in FIELDS.items():
            rows = item["tables"][table]
            content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
            filename = f"{item['name']}-{table}.jsonl"
            (directory / filename).write_bytes(content)
            manifest["tables"][table] = {
                "file": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "rows": len(rows),
                "fields": fields,
            }
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write(Path(sys.argv[1] if len(sys.argv) > 1 else "scenarios"))
    print("scenarios written")
