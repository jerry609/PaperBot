import csv
import json
import logging
import random
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from paperbot.utils.experiment_metrics import compute_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器：读取配置 -> 加载数据 -> 运行模型 -> 记录结果（轻量版可复现）"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results: List[Dict[str, Any]] = []
        self.seed: int = int(self.config.get("seed", 42))
        self._set_seed(self.seed)

    # -------------------------
    # 配置与数据加载
    # -------------------------
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _set_seed(self, seed: int):
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            # numpy 非必需，存在则设种子，不存在跳过
            pass

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载数据集 (MVP: CSV)"""
        dataset_name = self.config.get("dataset")
        data_path = Path("datasets/processed") / f"{dataset_name}.csv"

        if not data_path.exists():
            data_path = Path(self.config.get("dataset_path", ""))

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        logger.info(f"Loading dataset from {data_path}")

        data: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    # -------------------------
    # 模型执行（占位实现）
    # -------------------------
    def _majority_label(self, labels: Sequence[str]) -> str:
        counter = Counter(labels)
        if not counter:
            return "0"
        return counter.most_common(1)[0][0]

    def _predict_with_model(
        self, model_cfg: Dict[str, Any], texts: Sequence[str], labels: Sequence[str]
    ) -> List[str]:
        """
        占位预测逻辑：
          - rule_keyword: 关键词启发式
          - random: 按标签集合随机
          - majority / 默认: 预测为多数类
        """
        model_type = (model_cfg.get("type") or "majority").lower()
        all_labels = list(sorted(set(labels))) if labels else ["0", "1"]
        positive_label = model_cfg.get("positive_label", all_labels[-1])
        negative_label = model_cfg.get("negative_label", all_labels[0])
        majority = self._majority_label(labels)

        rng = random.Random(self.seed + hash(model_cfg.get("name", "")) % 997)

        if model_type == "rule_keyword":
            pos_kw = model_cfg.get("pos_keywords") or ["好", "棒", "推荐", "赞"]
            neg_kw = model_cfg.get("neg_keywords") or ["差", "不", "糟", "坑"]

            def infer(t: str) -> str:
                if any(k in t for k in pos_kw):
                    return positive_label
                if any(k in t for k in neg_kw):
                    return negative_label
                return majority

            return [infer(t or "") for t in texts]

        if model_type == "random":
            candidates = all_labels or [majority]
            return [rng.choice(candidates) for _ in texts]

        # 默认多数类
        return [majority for _ in texts]

    # -------------------------
    # 主流程
    # -------------------------
    def run(self):
        """执行实验"""
        logger.info(f"Starting experiment: {self.config.get('experiment_name')}")

        data = self.load_dataset()
        texts = [row.get("text", "") for row in data]
        labels = [str(row.get("label", "")).strip() for row in data]

        models = self.config.get("models", [])
        requested_metrics = self.config.get(
            "metrics", ["accuracy", "f1_macro", "f1_0", "f1_1"]
        )

        for model_cfg in models:
            model_name = model_cfg.get("name")
            logger.info(f"Running model: {model_name}")

            y_pred = self._predict_with_model(model_cfg, texts, labels)
            metric_values = compute_metrics(labels, y_pred, requested_metrics)

            self.results.append(
                {
                    "model": model_name,
                    **metric_values,
                }
            )

        self._save_results(labels)
        logger.info("Experiment completed.")

    # -------------------------
    # 结果保存
    # -------------------------
    def _save_results(self, labels: Sequence[str]):
        output_dir = Path(self.config.get("output_dir", "output/experiments"))
        output_dir.mkdir(parents=True, exist_ok=True)

        exp_name = self.config.get("experiment_name", "exp")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = output_dir / f"{exp_name}_{timestamp}_results.csv"
        if self.results:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            logger.info(f"Results saved to {csv_path}")

        dataset_summary = {
            "size": len(labels),
            "label_distribution": Counter(labels),
        }

        meta = {
            "config": self.config,
            "timestamp": timestamp,
            "seed": self.seed,
            "dataset_summary": dataset_summary,
            "results_summary": self.results,
            "git_commit": self._get_git_commit(),
            "pip_freeze": self._get_pip_freeze_summary(),
        }

        with open(output_dir / f"{exp_name}_{timestamp}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _get_git_commit(self) -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
                .decode()
                .strip()
            )
        except Exception:
            return ""

    def _get_pip_freeze_summary(self, limit: int = 30) -> List[str]:
        try:
            out = (
                subprocess.check_output(["pip", "freeze"], cwd=Path.cwd())
                .decode()
                .splitlines()
            )
            return out[:limit]
        except Exception:
            return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to experiment config yaml"
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    runner.run()

