# repro/nodes/analysis_node.py
"""
Analysis Node for Paper2Code pipeline.
Phase 2: Extract implementation specifications from paper context.

Enhanced with:
- Detailed hyperparameter extraction from Appendix/Experiment sections
- Config.yaml generation
- Structured training specification parsing
"""

import re
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from .base_node import BaseNode, NodeResult
from ..models import PaperContext, ReproductionPlan, ImplementationSpec

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
except Exception:
    query = None
    ClaudeAgentOptions = None


class AnalysisNode(BaseNode[ImplementationSpec]):
    """
    Extract implementation specifications from paper context.
    
    Enhanced Features:
    - Hyperparameter extraction from Appendix/Experiment sections
    - Config.yaml generation for reproducibility
    - Detailed training specification parsing
    
    Input: (PaperContext, ReproductionPlan) or (PaperContext, ReproductionPlan, EnvironmentSpec)
    Output: ImplementationSpec
    """
    
    # Base analysis prompt
    ANALYSIS_PROMPT = """Analyze this paper and extract implementation specifications.

Paper: {title}
Abstract: {abstract}
Method: {method}
Planned Components: {components}

For each component, provide:
1. Class/function name
2. Key methods or attributes
3. Input/output types

Output JSON:
{{
    "specs": [
        {{"name": "ModelName", "type": "class", "methods": ["forward", "train"], "description": "..."}},
        ...
    ],
    "hyperparameters": {{"learning_rate": 0.001, ...}},
    "data_format": "csv/json/..."
}}
"""

    # Enhanced hyperparameter extraction prompt
    HYPERPARAMETER_PROMPT = """Analyze the Appendix and Experiment sections of this paper.

Paper: {title}
Method: {method}
Hyperparameters mentioned in context: {existing_hyperparams}
Code snippets: {code_snippets}

Extract ALL hyperparameters mentioned, being as precise as possible:

1. **Optimizer Settings**:
   - Type (adam, sgd, adamw, etc.)
   - Learning rate (including schedule if mentioned)
   - Weight decay / L2 regularization
   - Momentum (for SGD)
   - Beta values (for Adam variants)

2. **Training Details**:
   - Batch size
   - Number of epochs
   - Warmup steps/epochs
   - Gradient clipping value
   - Dropout rate
   - Early stopping patience

3. **Model Architecture**:
   - Hidden dimension(s)
   - Number of layers/blocks
   - Attention heads (for transformers)
   - Activation function

4. **Learning Rate Scheduler**:
   - Type (step, cosine, warmup_cosine, etc.)
   - Milestones/decay steps
   - Gamma/decay rate

5. **Data Augmentation**:
   - Techniques used
   - Augmentation probabilities

Output valid JSON:
{{
    "optimizer": {{
        "type": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999]
    }},
    "training": {{
        "batch_size": 32,
        "epochs": 100,
        "warmup_epochs": 5,
        "gradient_clip": 1.0,
        "dropout": 0.1
    }},
    "model": {{
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "activation": "gelu"
    }},
    "scheduler": {{
        "type": "cosine",
        "min_lr": 1e-6
    }},
    "augmentation": {{
        "techniques": ["random_crop", "horizontal_flip"],
        "probability": 0.5
    }},
    "config_yaml": "# Full config.yaml content here..."
}}
"""

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(node_name="AnalysisNode", **kwargs)
        self.llm_client = llm_client
    
    def _validate_input(self, input_data: Any, **kwargs) -> Optional[str]:
        """Validate input is (PaperContext, ReproductionPlan) or with EnvironmentSpec."""
        if not isinstance(input_data, tuple) or len(input_data) < 2:
            return "Input must be tuple with at least (PaperContext, ReproductionPlan)"
        paper_context, plan = input_data[0], input_data[1]
        if not isinstance(paper_context, PaperContext):
            return "First element must be PaperContext"
        if not isinstance(plan, ReproductionPlan):
            return "Second element must be ReproductionPlan"
        return None
    
    async def _execute(self, input_data: tuple, **kwargs) -> ImplementationSpec:
        """Extract implementation specifications with enhanced hyperparameter extraction."""
        paper_context = input_data[0]
        plan = input_data[1]
        env_spec = input_data[2] if len(input_data) > 2 else None
        
        # Step 1: Extract basic structure specs
        basic_spec = await self._extract_basic_specs(paper_context, plan)
        
        # Step 2: Enhanced hyperparameter extraction
        hyperparams = await self._extract_hyperparameters(paper_context, basic_spec)
        
        # Step 3: Merge and generate config
        final_spec = self._merge_specs(basic_spec, hyperparams, env_spec)
        
        # Step 4: Generate config.yaml content
        final_spec.extra_params["config_yaml"] = self._generate_config_yaml(final_spec)
        
        return final_spec
    
    async def _extract_basic_specs(
        self, paper_context: PaperContext, plan: ReproductionPlan
    ) -> ImplementationSpec:
        """Extract basic implementation structure."""
        if query and ClaudeAgentOptions:
            try:
                prompt = self.ANALYSIS_PROMPT.format(
                    title=paper_context.title,
                    abstract=paper_context.abstract or "",
                    method=paper_context.method_section or "",
                    components=", ".join(plan.components) if hasattr(plan, 'components') else ", ".join(plan.key_components)
                )
                
                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=1500)
                )
                
                return self._parse_basic_spec(result.response)
            except Exception as e:
                logger.warning(f"LLM analysis failed, using fallback: {e}")
        
        return self._fallback_spec(plan)
    
    async def _extract_hyperparameters(
        self, paper_context: PaperContext, basic_spec: ImplementationSpec
    ) -> Dict[str, Any]:
        """Enhanced hyperparameter extraction from paper context."""
        # First, try regex-based extraction for common patterns
        regex_hyperparams = self._extract_hyperparams_regex(paper_context)
        
        # Then, try LLM-based extraction for comprehensive analysis
        if query and ClaudeAgentOptions:
            try:
                code_snippets = "\n".join(paper_context.algorithm_blocks[:2])
                prompt = self.HYPERPARAMETER_PROMPT.format(
                    title=paper_context.title,
                    method=paper_context.method_section[:2000] if paper_context.method_section else "",
                    existing_hyperparams=json.dumps(paper_context.hyperparameters),
                    code_snippets=code_snippets,
                )
                
                result = query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(max_tokens=2000)
                )
                
                llm_hyperparams = self._parse_hyperparams(result.response)
                
                # Merge regex and LLM results (LLM takes precedence)
                return self._merge_hyperparams(regex_hyperparams, llm_hyperparams)
            except Exception as e:
                logger.warning(f"LLM hyperparameter extraction failed: {e}")
        
        return regex_hyperparams
    
    def _extract_hyperparams_regex(self, ctx: PaperContext) -> Dict[str, Any]:
        """Extract hyperparameters using regex patterns."""
        text = " ".join([
            ctx.abstract,
            ctx.method_section or "",
            " ".join(ctx.algorithm_blocks),
        ])
        
        hyperparams = {
            "optimizer": {},
            "training": {},
            "model": {},
            "scheduler": {},
        }
        
        # Learning rate patterns
        lr_patterns = [
            r'learning\s*rate\s*(?:of|=|:)?\s*([\d.e-]+)',
            r'lr\s*[=:]\s*([\d.e-]+)',
            r'η\s*[=:]\s*([\d.e-]+)',
        ]
        for pattern in lr_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    hyperparams["optimizer"]["learning_rate"] = float(match.group(1))
                    break
                except ValueError:
                    pass
        
        # Batch size patterns
        batch_patterns = [
            r'batch\s*size\s*(?:of|=|:)?\s*(\d+)',
            r'mini-?batch\s*(?:of|=|:)?\s*(\d+)',
        ]
        for pattern in batch_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hyperparams["training"]["batch_size"] = int(match.group(1))
                break
        
        # Epochs
        epoch_match = re.search(r'(\d+)\s*epochs?|epochs?\s*[=:]\s*(\d+)', text, re.IGNORECASE)
        if epoch_match:
            hyperparams["training"]["epochs"] = int(epoch_match.group(1) or epoch_match.group(2))
        
        # Dropout
        dropout_match = re.search(r'dropout\s*(?:rate|prob)?\s*(?:of|=|:)?\s*([\d.]+)', text, re.IGNORECASE)
        if dropout_match:
            hyperparams["training"]["dropout"] = float(dropout_match.group(1))
        
        # Hidden dimension
        hidden_match = re.search(r'hidden\s*(?:dim|dimension|size)\s*(?:of|=|:)?\s*(\d+)', text, re.IGNORECASE)
        if hidden_match:
            hyperparams["model"]["hidden_dim"] = int(hidden_match.group(1))
        
        # Optimizer type
        optimizer_patterns = [
            (r'\bAdam\b', "adam"),
            (r'\bAdamW\b', "adamw"),
            (r'\bSGD\b', "sgd"),
            (r'\bRMSprop\b', "rmsprop"),
        ]
        for pattern, opt_type in optimizer_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hyperparams["optimizer"]["type"] = opt_type
                break
        
        return hyperparams
    
    def _parse_hyperparams(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for hyperparameters."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            logger.warning("Failed to parse hyperparameter JSON")
        return {}
    
    def _merge_hyperparams(
        self, regex_hp: Dict[str, Any], llm_hp: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge regex and LLM hyperparameters, LLM takes precedence."""
        merged = {}
        for key in set(list(regex_hp.keys()) + list(llm_hp.keys())):
            if key in llm_hp and llm_hp[key]:
                merged[key] = llm_hp[key]
            elif key in regex_hp and regex_hp[key]:
                merged[key] = regex_hp[key]
        return merged
    
    def _merge_specs(
        self, basic: ImplementationSpec, hyperparams: Dict[str, Any], env_spec
    ) -> ImplementationSpec:
        """Merge basic spec with extracted hyperparameters."""
        # Update optimizer settings
        opt = hyperparams.get("optimizer", {})
        if opt.get("learning_rate"):
            basic.learning_rate = opt["learning_rate"]
        if opt.get("type"):
            basic.optimizer = opt["type"]
        
        # Update training settings
        train = hyperparams.get("training", {})
        if train.get("batch_size"):
            basic.batch_size = train["batch_size"]
        if train.get("epochs"):
            basic.epochs = train["epochs"]
        
        # Store full hyperparams for config generation
        basic.extra_params["hyperparameters"] = hyperparams
        
        # Add environment info if available
        if env_spec:
            basic.extra_params["environment"] = env_spec.to_dict() if hasattr(env_spec, 'to_dict') else {}
        
        return basic
    
    def _generate_config_yaml(self, spec: ImplementationSpec) -> str:
        """Generate a config.yaml file from the specification."""
        import yaml
        
        config = {
            "model": {
                "type": spec.model_type or "custom",
            },
            "training": {
                "optimizer": spec.optimizer,
                "learning_rate": spec.learning_rate,
                "batch_size": spec.batch_size,
                "epochs": spec.epochs,
            },
            "data": {
                "format": spec.data_format,
            },
        }
        
        # Add extracted hyperparameters
        if "hyperparameters" in spec.extra_params:
            hp = spec.extra_params["hyperparameters"]
            
            # Merge model settings
            if hp.get("model"):
                config["model"].update(hp["model"])
            
            # Merge training settings
            if hp.get("training"):
                config["training"].update(hp["training"])
            
            # Add optimizer details
            if hp.get("optimizer"):
                config["optimizer"] = hp["optimizer"]
            
            # Add scheduler
            if hp.get("scheduler"):
                config["scheduler"] = hp["scheduler"]
            
            # Add augmentation
            if hp.get("augmentation"):
                config["augmentation"] = hp["augmentation"]
        
        # Add environment info
        if "environment" in spec.extra_params:
            config["environment"] = spec.extra_params["environment"]
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def _parse_basic_spec(self, response: str) -> ImplementationSpec:
        """Parse LLM response into ImplementationSpec."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                hyperparams = data.get("hyperparameters", {})
                return ImplementationSpec(
                    layers=data.get("specs", []),
                    learning_rate=hyperparams.get("learning_rate", 1e-4),
                    batch_size=hyperparams.get("batch_size", 32),
                    epochs=hyperparams.get("epochs", 10),
                    data_format=data.get("data_format", "csv"),
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse spec JSON, using fallback")
        
        return self._fallback_spec(None)
    
    def _fallback_spec(self, plan: Optional[ReproductionPlan]) -> ImplementationSpec:
        """Generate fallback spec."""
        return ImplementationSpec(
            layers=[
                {"name": "Model", "type": "class", "methods": ["forward", "train"]},
                {"name": "DataLoader", "type": "class", "methods": ["load", "preprocess"]},
            ],
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            data_format="csv",
        )

