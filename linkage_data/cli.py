"""分析命令行入口。

用法：
  python -m linkage_data.cli data/normal.json                     # 原有完整性检查
  python -m linkage_data.cli analyze data/normal.json --period 1  # 不确定关联分析
  python -m linkage_data.cli snapshot data/normal.json --period 1 --version 1 --out out/snapshots
  python -m linkage_data.cli verify data/normal.json out/snapshots/<文件>.json
  python -m linkage_data.cli replay scenarios/bridge-revocation/manifest.json \
      --period 1 --versions 1,2 --out out/replay.json
  python -m linkage_data.cli sensitivity scenarios/phone-change/manifest.json --period 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .analyze import report, run, save_snapshot, sensitivity, verify_snapshot
from .domain import inspect_tables
from .io import DataError, load
from .model import ConfigError


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest")
    parser.add_argument("--period", type=int, required=True, help="分析时期（精确匹配）")
    parser.add_argument("--version", type=int, default=1, help="资料版本，决定撤销可见性")
    parser.add_argument("--samples", type=int, default=None, help="每条链保存的世界数")
    parser.add_argument("--draws", type=int, default=None, help="校准参数抽样数")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-period-gap", type=int, default=None)
    parser.add_argument("--bridge-period-scope", choices=["analysis", "all"], default=None)
    parser.add_argument("--label-sampling", choices=["unknown", "stratified_random"], default=None)
    parser.add_argument("--config", default=None, help="JSON 配置文件（与命令行同时给出时命令行优先）")


def _raw_config(args: argparse.Namespace) -> dict:
    raw = {}
    if args.config:
        raw.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    raw["period"] = args.period
    raw["version"] = args.version
    for cli, key in (
        ("samples", "n_samples"),
        ("draws", "calibration_draws"),
        ("seed", "seed"),
        ("max_period_gap", "max_period_gap"),
        ("bridge_period_scope", "bridge_period_scope"),
        ("label_sampling", "label_sampling"),
    ):
        value = getattr(args, cli)
        if value is not None:
            raw[key] = value
    return raw


def _run(manifest: str, raw: dict):
    path = Path(manifest).resolve()
    tables = load(path)
    inspect_tables(tables)
    digest = _digest(path)
    return run(tables, raw, manifest_digest=digest)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 不指定子命令，或以清单路径/check 开头时，保持原有的完整性检查入口。
    if not argv or argv[0] in ("inspect", "check") or (argv[0].endswith(".json") and argv[0] != "--config"):
        if argv and argv[0] in ("inspect", "check"):
            argv = argv[1:]
        parser = argparse.ArgumentParser(description="检查业务资料的身份与完整性")
        parser.add_argument("manifest")
        args = parser.parse_args(argv)
        try:
            print(json.dumps(inspect_tables(load(args.manifest)), ensure_ascii=False, indent=2))
        except (DataError, ValueError) as exc:
            parser.exit(2, str(exc) + "\n")
        return
    command = argv.pop(0)
    try:
        if command == "analyze":
            parser = argparse.ArgumentParser(prog="linkage_data.cli analyze", description="不确定身份关联下的覆盖分析")
            _common(parser)
            parser.add_argument("--out", default=None, help="同时写出完整 JSON 报告")
            args = parser.parse_args(argv)
            analysis = _run(args.manifest, _raw_config(args))
            payload = report(analysis)
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(text)
        elif command == "snapshot":
            parser = argparse.ArgumentParser(prog="linkage_data.cli snapshot", description="保存版本化分析快照")
            _common(parser)
            parser.add_argument("--out", default="out/snapshots")
            parser.add_argument("--label", default=None)
            args = parser.parse_args(argv)
            analysis = _run(args.manifest, _raw_config(args))
            path = save_snapshot(analysis, args.out, args.label)
            print(json.dumps({"snapshot": str(path), "calibration_signature": analysis.calibration.signature,
                              "worlds_digest": analysis.snapshot["worlds_digest"],
                              "review_queue_digest": analysis.snapshot["review_queue_digest"]}, ensure_ascii=False, indent=2))
        elif command == "verify":
            parser = argparse.ArgumentParser(prog="linkage_data.cli verify", description="用快照配置复算并比较摘要")
            parser.add_argument("manifest")
            parser.add_argument("snapshot")
            args = parser.parse_args(argv)
            tables = load(Path(args.manifest).resolve())
            inspect_tables(tables)
            result = verify_snapshot(args.snapshot, tables)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["matches"]:
                raise SystemExit(1)
        elif command == "sensitivity":
            parser = argparse.ArgumentParser(prog="linkage_data.cli sensitivity", description="模型设置敏感性")
            _common(parser)
            parser.add_argument("--cohesion", default="0.15,0.25,0.35")
            parser.add_argument("--out", default=None)
            args = parser.parse_args(argv)
            tables = load(Path(args.manifest).resolve())
            inspect_tables(tables)
            try:
                values = tuple(float(v) for v in args.cohesion.split(","))
            except ValueError as exc:
                raise ConfigError("--cohesion 必须是逗号分隔的数值") from exc
            result = sensitivity(tables, _raw_config(args), values)
            text = json.dumps(result, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(text)
        elif command == "replay":
            parser = argparse.ArgumentParser(prog="linkage_data.cli replay", description="回放多个资料版本并比较覆盖结果")
            _common(parser)
            parser.add_argument("--versions", default=None, help="逗号分隔的版本号；默认只跑 --version")
            parser.add_argument("--out", default=None)
            args = parser.parse_args(argv)
            versions = [int(v) for v in args.versions.split(",")] if args.versions else [args.version]
            rows = []
            for version in versions:
                raw = _raw_config(args)
                raw["version"] = version
                analysis = _run(args.manifest, raw)
                payload = report(analysis)
                dist = payload["coverage"]["posterior_distribution"]
                rows.append(
                    {
                        "version": version,
                        "visible_bridges": payload["bridge_visibility"]["visible"],
                        "revoked_bridges": payload["bridge_visibility"]["revoked_in_version"],
                        "beneficiaries_count": dist["beneficiaries_count"],
                        "department_overlap_people": dist["department_overlap_people"],
                        "uncovered_fraction_low": dist["uncovered_fraction_low"],
                        "uncovered_fraction_high": dist["uncovered_fraction_high"],
                        "calibration_source": payload["calibration"]["source"],
                        "calibration_signature": payload["calibration"]["signature"],
                        "worlds_digest": payload["snapshot"]["worlds_digest"],
                        "review_queue_digest": payload["snapshot"]["review_queue_digest"],
                        "warnings": payload["warnings"],
                    }
                )
            result = {"manifest": str(Path(args.manifest).resolve()), "period": args.period, "versions": rows}
            text = json.dumps(result, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(text)
        else:
            raise SystemExit(f"未知命令: {command}")
    except (DataError, ConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
