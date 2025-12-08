"""
数据集校验脚本
- 检查 datasets/processed 下的 CSV 是否包含 text/label 字段
- 检查 metadata 是否包含 license/source
运行：python scripts/validate_datasets.py
"""

import csv
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "datasets" / "processed"
METADATA_DIR = ROOT / "datasets" / "metadata"


def validate_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    missing = [c for c in ("text", "label") if c not in headers]
    return missing


def validate_metadata(name: str):
    meta_path = METADATA_DIR / f"{name}.yaml"
    if not meta_path.exists():
        return ["metadata_missing"]
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    missing = []
    if not data.get("license"):
        missing.append("license")
    if not data.get("source"):
        missing.append("source")
    return missing


def main():
    if not PROCESSED_DIR.exists():
        print("⚠️  datasets/processed 不存在")
        sys.exit(1)

    any_error = False
    for csv_file in PROCESSED_DIR.glob("*.csv"):
        name = csv_file.stem
        missing_cols = validate_csv(csv_file)
        meta_missing = validate_metadata(name)

        if missing_cols:
            any_error = True
            print(f"❌ {csv_file.name}: 缺少列 {missing_cols}")
        else:
            print(f"✅ {csv_file.name}: 字段齐全")

        if meta_missing:
            any_error = True
            print(f"⚠️ {name}.yaml: 缺少 {meta_missing}")
        else:
            print(f"✅ {name}.yaml: metadata 完整")

    if any_error:
        sys.exit(1)
    print("🎉 数据集校验通过")


if __name__ == "__main__":
    main()

