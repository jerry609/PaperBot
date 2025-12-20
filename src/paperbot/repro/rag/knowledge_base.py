# repro/rag/knowledge_base.py
"""
Code Knowledge Base for Paper2Code pipeline.

Provides tag-based retrieval of code patterns and templates.
Uses keyword matching (no embedding dependencies required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """
    A reusable code pattern or template.

    Attributes:
        name: Pattern identifier
        description: What this pattern does
        code: The actual code template
        tags: Keywords for retrieval
        source: Origin (e.g., "PyTorch docs", "ICML 2024")
        language: Programming language
    """
    name: str
    description: str
    code: str
    tags: List[str]
    source: str = "builtin"
    language: str = "python"

    def to_context(self) -> str:
        """Format pattern for LLM context injection."""
        return f"# Pattern: {self.name}\n# {self.description}\n{self.code}"


class CodeKnowledgeBase:
    """
    Code Knowledge Base - tag-based pattern retrieval.

    Uses keyword matching for retrieval without requiring embeddings.
    Patterns are indexed by tags for fast lookup.

    Usage:
        kb = CodeKnowledgeBase.from_builtin()
        patterns = kb.search("pytorch transformer training", k=3)
        for p in patterns:
            print(p.name, p.code)
    """

    def __init__(self):
        self.patterns: Dict[str, CodePattern] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> pattern names

    def add_pattern(self, pattern: CodePattern) -> None:
        """
        Add a pattern and index its tags.

        Args:
            pattern: CodePattern to add
        """
        self.patterns[pattern.name] = pattern

        # Index by tags
        for tag in pattern.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(pattern.name)

    def search(self, query: str, k: int = 3) -> List[CodePattern]:
        """
        Search for relevant patterns using keyword matching.

        Scoring:
        - +2 for tag match
        - +2 for name match
        - +1 for description match

        Args:
            query: Search query (space-separated keywords)
            k: Maximum number of results

        Returns:
            List of matching CodePatterns, sorted by relevance
        """
        query_tokens = set(query.lower().split())
        scores: Dict[str, int] = {}

        for pattern_name, pattern in self.patterns.items():
            score = 0

            # Tag matching (highest weight)
            pattern_tags = set(t.lower() for t in pattern.tags)
            tag_matches = len(query_tokens & pattern_tags)
            score += tag_matches * 2

            # Name matching
            name_tokens = set(pattern.name.lower().replace("_", " ").split())
            if query_tokens & name_tokens:
                score += 2

            # Description matching
            desc_tokens = set(pattern.description.lower().split())
            if query_tokens & desc_tokens:
                score += 1

            if score > 0:
                scores[pattern_name] = score

        # Sort by score and return top-k
        sorted_names = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [self.patterns[name] for name in sorted_names[:k]]

    def get_by_tag(self, tag: str) -> List[CodePattern]:
        """Get all patterns with a specific tag."""
        tag_lower = tag.lower()
        if tag_lower not in self._tag_index:
            return []
        return [self.patterns[name] for name in self._tag_index[tag_lower]]

    def get_pattern(self, name: str) -> Optional[CodePattern]:
        """Get a pattern by name."""
        return self.patterns.get(name)

    def list_tags(self) -> List[str]:
        """List all available tags."""
        return sorted(self._tag_index.keys())

    def list_patterns(self) -> List[str]:
        """List all pattern names."""
        return list(self.patterns.keys())

    @classmethod
    def from_builtin(cls) -> "CodeKnowledgeBase":
        """
        Create a knowledge base with built-in patterns.

        Includes common ML/DL code patterns:
        - PyTorch training loop
        - Transformer architecture
        - Data loading
        - Evaluation metrics
        """
        kb = cls()

        for pattern in BUILTIN_PATTERNS:
            kb.add_pattern(pattern)

        logger.info(f"Loaded {len(kb.patterns)} built-in patterns")
        return kb

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "patterns": {
                name: {
                    "description": p.description,
                    "tags": p.tags,
                    "source": p.source,
                    "code_length": len(p.code),
                }
                for name, p in self.patterns.items()
            },
            "tag_index": {tag: list(names) for tag, names in self._tag_index.items()},
        }


# ============================================================================
# Built-in Patterns
# ============================================================================

BUILTIN_PATTERNS = [
    CodePattern(
        name="pytorch_training_loop",
        description="Standard PyTorch training loop with validation",
        tags=["pytorch", "training", "loop", "deep learning", "neural network"],
        source="PyTorch Tutorials",
        code='''
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)
''',
    ),

    CodePattern(
        name="transformer_encoder_block",
        description="Transformer encoder block with multi-head attention",
        tags=["transformer", "attention", "encoder", "self-attention", "deep learning"],
        source="Attention Is All You Need",
        code='''
class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
''',
    ),

    CodePattern(
        name="pytorch_dataloader",
        description="Custom PyTorch Dataset and DataLoader setup",
        tags=["pytorch", "dataloader", "dataset", "data", "loading"],
        source="PyTorch Tutorials",
        code='''
class CustomDataset(Dataset):
    """Custom dataset for loading data."""

    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load sample paths/data."""
        # Implement based on data format
        samples = []
        # Example: load from CSV, directory, etc.
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load and process sample
        if self.transform:
            sample = self.transform(sample)
        return sample


def create_dataloaders(train_path, val_path, batch_size=32, num_workers=4):
    """Create train and validation dataloaders."""
    train_dataset = CustomDataset(train_path, transform=train_transform)
    val_dataset = CustomDataset(val_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
''',
    ),

    CodePattern(
        name="config_dataclass",
        description="Configuration using dataclass with type hints",
        tags=["config", "configuration", "dataclass", "settings", "hyperparameters"],
        source="Best Practices",
        code='''
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"

    # Device
    device: str = "cuda"
    seed: int = 42


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
        )
''',
    ),

    CodePattern(
        name="mlp_block",
        description="Multi-layer perceptron (MLP) block",
        tags=["mlp", "feedforward", "neural network", "layers", "deep learning"],
        source="Common Pattern",
        code='''
class MLP(nn.Module):
    """Multi-layer perceptron with configurable layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
''',
    ),

    CodePattern(
        name="evaluation_metrics",
        description="Common evaluation metrics for classification and regression",
        tags=["evaluation", "metrics", "accuracy", "f1", "classification"],
        source="scikit-learn",
        code='''
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def compute_classification_metrics(y_true, y_pred, average="macro"):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
    }


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


class MetricTracker:
    """Track metrics during training."""

    def __init__(self):
        self.history = {}

    def update(self, metrics: dict, step: int):
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append((step, value))

    def get_best(self, metric_name: str, mode: str = "max"):
        values = self.history.get(metric_name, [])
        if not values:
            return None
        func = max if mode == "max" else min
        return func(values, key=lambda x: x[1])
''',
    ),

    CodePattern(
        name="checkpoint_manager",
        description="Save and load model checkpoints",
        tags=["checkpoint", "save", "load", "model", "training"],
        source="Best Practices",
        code='''
import torch
from pathlib import Path


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(self, model, optimizer, epoch, metrics, name=None):
        """Save a checkpoint."""
        if name is None:
            name = f"checkpoint_epoch_{epoch}.pt"

        path = self.checkpoint_dir / name

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, path)

        self.checkpoints.append(path)
        self._cleanup_old_checkpoints()

        return path

    def load(self, path, model, optimizer=None):
        """Load a checkpoint."""
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})

    def load_best(self, model, optimizer=None, metric="loss", mode="min"):
        """Load the best checkpoint by metric."""
        # Implementation depends on tracking
        pass

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
''',
    ),

    CodePattern(
        name="cnn_encoder",
        description="Convolutional neural network encoder for images",
        tags=["cnn", "convolutional", "encoder", "image", "vision"],
        source="ResNet-style",
        code='''
class CNNEncoder(nn.Module):
    """CNN encoder for image feature extraction."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: List[int] = [64, 128, 256, 512],
        output_dim: int = 512,
    ):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for channels in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            prev_channels = channels

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
''',
    ),

    CodePattern(
        name="learning_rate_scheduler",
        description="Learning rate scheduling with warmup",
        tags=["scheduler", "learning rate", "warmup", "training", "optimizer"],
        source="Transformer Training",
        code='''
import math


class WarmupCosineScheduler:
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )


def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear schedule with warmup (Hugging Face style)."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
''',
    ),

    CodePattern(
        name="argparse_cli",
        description="Command-line argument parsing",
        tags=["argparse", "cli", "arguments", "main", "entry"],
        source="Best Practices",
        code='''
import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Training script")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Path to output directory")

    # Model arguments
    parser.add_argument("--model-type", type=str, default="transformer",
                       choices=["transformer", "cnn", "mlp"],
                       help="Model architecture")
    parser.add_argument("--hidden-size", type=int, default=768,
                       help="Hidden dimension size")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Misc
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
''',
    ),
]
