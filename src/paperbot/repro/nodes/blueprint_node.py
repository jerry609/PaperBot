# repro/nodes/blueprint_node.py
"""
Blueprint Distillation Node for Paper2Code pipeline.
Inspired by DeepCode's Blueprint Distillation approach.

Extracts structured information from paper context into a compressed
Blueprint format that maximizes signal while minimizing token usage.
"""

import logging
import json
import re
from typing import Any, Optional, List, Dict
from .base_node import BaseNode, NodeResult
from ..models import PaperContext, Blueprint, AlgorithmSpec

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


class BlueprintDistillationNode(BaseNode[Blueprint]):
    """
    Distill paper context into a structured Blueprint.

    This node extracts and compresses paper information into a format
    optimized for downstream code generation:
    - Architecture type and module hierarchy
    - Core algorithms with pseudocode
    - Key hyperparameters
    - I/O specifications

    Input: PaperContext
    Output: Blueprint
    """

    DISTILLATION_PROMPT = """You are an expert at analyzing machine learning papers and extracting implementation details.

Analyze the following paper and extract structured information for code reproduction.

Paper Title: {title}
Abstract: {abstract}
Method Section: {method}
Algorithm Blocks: {algorithms}
Hyperparameters: {hyperparams}

Extract the following information and return as JSON:
{{
    "architecture_type": "transformer|cnn|gnn|rnn|mlp|hybrid|unknown",
    "module_hierarchy": {{
        "model": ["encoder", "decoder", "attention"],
        ...
    }},
    "data_flow": [
        ["input", "encoder"],
        ["encoder", "attention"],
        ...
    ],
    "core_algorithms": [
        {{
            "name": "Algorithm Name",
            "pseudocode": "Step-by-step pseudocode",
            "complexity": "O(n^2)",
            "inputs": ["input1", "input2"],
            "outputs": ["output1"]
        }}
    ],
    "loss_functions": ["CrossEntropyLoss", "MSELoss", ...],
    "optimization_strategy": "AdamW with cosine schedule, warmup 1000 steps",
    "key_hyperparameters": {{
        "hidden_size": 768,
        "num_layers": 12,
        "learning_rate": 1e-4,
        ...
    }},
    "input_output_spec": {{
        "input": "tensor of shape (batch, seq_len)",
        "output": "logits of shape (batch, num_classes)"
    }},
    "framework_hints": ["pytorch", "transformers"],
    "domain": "nlp|cv|audio|multimodal|rl|other"
}}

Focus on extracting concrete implementation details. If something is not mentioned, omit it or use reasonable defaults.
"""

    # Common architecture patterns for detection
    ARCHITECTURE_PATTERNS = {
        "transformer": [
            "attention", "self-attention", "multi-head", "transformer",
            "encoder-decoder", "bert", "gpt", "llama", "qkv"
        ],
        "cnn": [
            "convolutional", "conv2d", "resnet", "vgg", "inception",
            "pooling", "kernel", "stride", "feature map"
        ],
        "gnn": [
            "graph neural", "message passing", "node embedding", "edge",
            "gcn", "gat", "graphsage", "aggregation"
        ],
        "rnn": [
            "recurrent", "lstm", "gru", "hidden state", "sequence",
            "bidirectional", "seq2seq"
        ],
        "mlp": [
            "fully connected", "feedforward", "dense layer", "perceptron"
        ],
        "diffusion": [
            "diffusion", "denoising", "noise schedule", "ddpm", "ddim",
            "score matching", "sde"
        ],
        "vae": [
            "variational", "latent", "kl divergence", "encoder decoder",
            "reparameterization"
        ],
        "gan": [
            "generative adversarial", "discriminator", "generator",
            "adversarial training"
        ],
    }

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        "nlp": [
            "text", "language", "token", "embedding", "bert", "gpt",
            "translation", "sentiment", "ner", "parsing", "llm"
        ],
        "cv": [
            "image", "vision", "pixel", "object detection", "segmentation",
            "classification", "cnn", "resnet", "vit"
        ],
        "audio": [
            "audio", "speech", "spectrogram", "mel", "wav", "asr",
            "tts", "music"
        ],
        "multimodal": [
            "multimodal", "vision-language", "clip", "image-text",
            "cross-modal"
        ],
        "rl": [
            "reinforcement", "reward", "policy", "value function",
            "action", "environment", "agent", "q-learning", "ppo"
        ],
    }

    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        "pytorch": ["torch", "pytorch", "nn.Module", ".cuda()", "tensor"],
        "tensorflow": ["tensorflow", "tf.", "keras", "tf.keras"],
        "jax": ["jax", "flax", "optax", "jnp."],
        "transformers": ["huggingface", "transformers", "AutoModel", "from_pretrained"],
    }

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(node_name="BlueprintDistillationNode", **kwargs)
        self.llm_client = llm_client

    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is a PaperContext."""
        if not isinstance(input_data, PaperContext):
            return "Input must be a PaperContext"
        if not input_data.title:
            return "PaperContext must have a title"
        return None

    async def _execute(self, input_data: PaperContext, **kwargs) -> Blueprint:
        """Distill paper context into Blueprint."""
        paper_context = input_data

        # Try LLM-based distillation first
        if query and ClaudeAgentOptions:
            try:
                prompt = self.DISTILLATION_PROMPT.format(
                    title=paper_context.title,
                    abstract=paper_context.abstract or "",
                    method=paper_context.method_section or "",
                    algorithms="\n".join(paper_context.algorithm_blocks) if paper_context.algorithm_blocks else "",
                    hyperparams=json.dumps(paper_context.hyperparameters) if paper_context.hyperparameters else "{}"
                )

                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=2000)
                )

                blueprint = self._parse_blueprint(result.response, paper_context)
                if blueprint:
                    return blueprint
            except Exception as e:
                logger.warning(f"LLM distillation failed, using heuristic: {e}")

        # Fallback to heuristic-based distillation
        return self._heuristic_distillation(paper_context)

    def _parse_blueprint(self, response: str, paper_context: PaperContext) -> Optional[Blueprint]:
        """Parse LLM response into Blueprint."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])

                # Parse algorithms
                algorithms = []
                for algo_data in data.get("core_algorithms", []):
                    algorithms.append(AlgorithmSpec(
                        name=algo_data.get("name", ""),
                        pseudocode=algo_data.get("pseudocode", ""),
                        complexity=algo_data.get("complexity", ""),
                        inputs=algo_data.get("inputs", []),
                        outputs=algo_data.get("outputs", []),
                    ))

                # Parse data_flow as list of tuples
                data_flow = [
                    (flow[0], flow[1])
                    for flow in data.get("data_flow", [])
                    if len(flow) >= 2
                ]

                return Blueprint(
                    architecture_type=data.get("architecture_type", "unknown"),
                    module_hierarchy=data.get("module_hierarchy", {}),
                    data_flow=data_flow,
                    core_algorithms=algorithms,
                    loss_functions=data.get("loss_functions", []),
                    optimization_strategy=data.get("optimization_strategy", ""),
                    key_hyperparameters=data.get("key_hyperparameters", {}),
                    input_output_spec=data.get("input_output_spec", {}),
                    paper_title=paper_context.title,
                    paper_year=self._extract_year(paper_context),
                    framework_hints=data.get("framework_hints", []),
                    domain=data.get("domain", ""),
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse blueprint JSON: {e}")

        return None

    def _heuristic_distillation(self, paper_context: PaperContext) -> Blueprint:
        """
        Extract blueprint using pattern matching when LLM is unavailable.

        Uses keyword detection to identify:
        - Architecture type
        - Domain
        - Framework hints
        - Hyperparameters from text
        """
        # Combine all text for analysis
        full_text = " ".join([
            paper_context.title or "",
            paper_context.abstract or "",
            paper_context.method_section or "",
        ]).lower()

        # Detect architecture
        architecture_type = self._detect_architecture(full_text)

        # Detect domain
        domain = self._detect_domain(full_text)

        # Detect frameworks
        framework_hints = self._detect_frameworks(full_text)

        # Extract hyperparameters from text
        hyperparameters = self._extract_hyperparameters(
            full_text,
            paper_context.hyperparameters
        )

        # Extract algorithms from code blocks
        algorithms = self._extract_algorithms(paper_context.algorithm_blocks)

        # Build module hierarchy based on architecture
        module_hierarchy = self._infer_module_hierarchy(architecture_type, domain)

        return Blueprint(
            architecture_type=architecture_type,
            module_hierarchy=module_hierarchy,
            data_flow=self._infer_data_flow(module_hierarchy),
            core_algorithms=algorithms,
            loss_functions=self._detect_loss_functions(full_text),
            optimization_strategy=self._detect_optimizer(full_text),
            key_hyperparameters=hyperparameters,
            input_output_spec={},
            paper_title=paper_context.title,
            paper_year=self._extract_year(paper_context),
            framework_hints=framework_hints,
            domain=domain,
        )

    def _detect_architecture(self, text: str) -> str:
        """Detect architecture type from text."""
        scores = {}
        for arch, patterns in self.ARCHITECTURE_PATTERNS.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                scores[arch] = score

        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def _detect_domain(self, text: str) -> str:
        """Detect domain from text."""
        scores = {}
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = sum(1 for p in patterns if p in text)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "other"

    def _detect_frameworks(self, text: str) -> List[str]:
        """Detect mentioned frameworks."""
        detected = []
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            if any(p.lower() in text for p in patterns):
                detected.append(framework)

        # Default to pytorch if nothing detected
        if not detected:
            detected = ["pytorch"]

        return detected

    def _extract_hyperparameters(
        self,
        text: str,
        existing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract hyperparameters from text."""
        params = dict(existing) if existing else {}

        # Common hyperparameter patterns
        patterns = [
            (r"learning.?rate[:\s]+([0-9e\-\.]+)", "learning_rate", float),
            (r"batch.?size[:\s]+(\d+)", "batch_size", int),
            (r"hidden.?size[:\s]+(\d+)", "hidden_size", int),
            (r"hidden.?dim(?:ension)?[:\s]+(\d+)", "hidden_size", int),
            (r"num.?layers?[:\s]+(\d+)", "num_layers", int),
            (r"num.?heads?[:\s]+(\d+)", "num_heads", int),
            (r"dropout[:\s]+([0-9\.]+)", "dropout", float),
            (r"epochs?[:\s]+(\d+)", "epochs", int),
            (r"warmup.?steps?[:\s]+(\d+)", "warmup_steps", int),
            (r"weight.?decay[:\s]+([0-9e\-\.]+)", "weight_decay", float),
            (r"max.?seq(?:uence)?.?len(?:gth)?[:\s]+(\d+)", "max_seq_length", int),
            (r"embedding.?dim(?:ension)?[:\s]+(\d+)", "embedding_dim", int),
        ]

        for pattern, key, dtype in patterns:
            if key not in params:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        params[key] = dtype(match.group(1))
                    except ValueError:
                        pass

        return params

    def _extract_algorithms(self, algorithm_blocks: List[str]) -> List[AlgorithmSpec]:
        """Extract algorithm specifications from pseudocode blocks."""
        algorithms = []

        for i, block in enumerate(algorithm_blocks or []):
            # Try to extract algorithm name from first line
            lines = block.strip().split('\n')
            name = f"Algorithm_{i + 1}"

            if lines:
                first_line = lines[0].strip()
                # Common patterns: "Algorithm 1: Name" or "def name(" or "function name"
                name_match = re.search(
                    r"(?:Algorithm\s+\d+[:\s]+)?([A-Za-z][A-Za-z0-9_\-\s]+)",
                    first_line
                )
                if name_match:
                    name = name_match.group(1).strip()

            algorithms.append(AlgorithmSpec(
                name=name,
                pseudocode=block,
                complexity="",
                inputs=[],
                outputs=[],
            ))

        return algorithms

    def _detect_loss_functions(self, text: str) -> List[str]:
        """Detect mentioned loss functions."""
        losses = []

        loss_patterns = [
            ("cross.?entropy", "CrossEntropyLoss"),
            ("mse|mean.?squared", "MSELoss"),
            ("mae|mean.?absolute", "L1Loss"),
            ("bce|binary.?cross.?entropy", "BCELoss"),
            ("kl.?div(?:ergence)?", "KLDivLoss"),
            ("contrastive", "ContrastiveLoss"),
            ("triplet", "TripletLoss"),
            ("focal", "FocalLoss"),
            ("dice", "DiceLoss"),
            ("huber", "HuberLoss"),
        ]

        for pattern, loss_name in loss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                losses.append(loss_name)

        return losses or ["CrossEntropyLoss"]  # Default

    def _detect_optimizer(self, text: str) -> str:
        """Detect optimization strategy."""
        parts = []

        # Optimizer
        if "adamw" in text:
            parts.append("AdamW")
        elif "adam" in text:
            parts.append("Adam")
        elif "sgd" in text:
            parts.append("SGD")
        else:
            parts.append("Adam")

        # Schedule
        if "cosine" in text:
            parts.append("cosine schedule")
        elif "linear" in text and "schedule" in text:
            parts.append("linear schedule")
        elif "step" in text and ("decay" in text or "schedule" in text):
            parts.append("step decay")

        # Warmup
        if "warmup" in text:
            parts.append("with warmup")

        return ", ".join(parts)

    def _infer_module_hierarchy(
        self,
        architecture: str,
        domain: str
    ) -> Dict[str, List[str]]:
        """Infer module hierarchy based on architecture."""
        hierarchies = {
            "transformer": {
                "model": ["encoder", "decoder", "embedding"],
                "encoder": ["attention", "ffn", "norm"],
                "decoder": ["self_attention", "cross_attention", "ffn"],
            },
            "cnn": {
                "model": ["backbone", "neck", "head"],
                "backbone": ["conv_layers", "pooling"],
                "head": ["fc_layers", "classifier"],
            },
            "gnn": {
                "model": ["encoder", "message_passing", "readout"],
                "message_passing": ["aggregate", "update"],
            },
            "rnn": {
                "model": ["embedding", "encoder", "decoder"],
                "encoder": ["rnn_layers", "attention"],
            },
            "diffusion": {
                "model": ["unet", "noise_schedule", "sampler"],
                "unet": ["down_blocks", "mid_block", "up_blocks"],
            },
        }

        return hierarchies.get(architecture, {
            "model": ["encoder", "decoder", "head"]
        })

    def _infer_data_flow(
        self,
        hierarchy: Dict[str, List[str]]
    ) -> List[tuple]:
        """Infer data flow from module hierarchy."""
        flow = []

        # Input to first module
        if "model" in hierarchy:
            components = hierarchy["model"]
            flow.append(("input", components[0] if components else "model"))

            # Sequential flow through components
            for i in range(len(components) - 1):
                flow.append((components[i], components[i + 1]))

            # Last component to output
            if components:
                flow.append((components[-1], "output"))

        return flow

    def _extract_year(self, paper_context: PaperContext) -> int:
        """Extract publication year from paper context."""
        # Try to find year in title or abstract
        text = f"{paper_context.title} {paper_context.abstract or ''}"
        year_match = re.search(r"20[12]\d", text)
        if year_match:
            return int(year_match.group())
        return 2024  # Default to current year
