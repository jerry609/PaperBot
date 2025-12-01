import logging
from typing import Dict, Any, List, Optional
import csv
from pathlib import Path
import yaml
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """实验运行器：读取配置 -> 加载数据 -> 运行模型 -> 记录结果"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """加载数据集 (MVP: 只支持 csv，不依赖 pandas 以避免 numpy 冲突)"""
        dataset_name = self.config.get("dataset")
        # 假设数据在 datasets/processed/{name}.csv
        data_path = Path("datasets/processed") / f"{dataset_name}.csv"
        
        if not data_path.exists():
            # 尝试 fallback 到自定义路径
            data_path = Path(self.config.get("dataset_path", ""))
            
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        logger.info(f"Loading dataset from {data_path}")
        
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def run(self):
        """执行实验"""
        logger.info(f"Starting experiment: {self.config.get('experiment_name')}")
        
        data = self.load_dataset()
        models = self.config.get("models", [])
        
        for model_cfg in models:
            model_name = model_cfg.get("name")
            logger.info(f"Running model: {model_name}")
            
            # MVP: 模拟模型预测
            # 实际应调用 model.predict(df['text'])
            import random
            metrics = {
                "accuracy": random.uniform(0.7, 0.95),
                "f1_score": random.uniform(0.6, 0.9),
                "latency_ms": random.uniform(10, 100)
            }
            
            self.results.append({
                "model": model_name,
                **metrics
            })
            
        self._save_results()
        logger.info("Experiment completed.")

    def _save_results(self):
        """保存结果"""
        output_dir = Path(self.config.get("output_dir", "output/experiments"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exp_name = self.config.get("experiment_name", "exp")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 CSV
        csv_path = output_dir / f"{exp_name}_{timestamp}_results.csv"
        if self.results:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            logger.info(f"Results saved to {csv_path}")
        
        # 保存元信息
        meta = {
            "config": self.config,
            "timestamp": timestamp,
            "results_summary": self.results
        }
        import json
        with open(output_dir / f"{exp_name}_{timestamp}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config yaml")
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    runner.run()

