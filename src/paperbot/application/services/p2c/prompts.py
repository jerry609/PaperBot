"""LLM prompt templates for P2C extraction stages."""

from __future__ import annotations

from typing import Dict, Tuple


def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def _paper_context(title: str, abstract: str, sections: Dict[str, str]) -> str:
    parts = [f"Paper: {title}", f"Abstract: {abstract}"]
    for key in ("method", "introduction", "experiment", "results", "conclusion"):
        section_text = sections.get(key, "")
        if section_text:
            parts.append(f"{key.title()} section: {_truncate(section_text)}")
    return "\n\n".join(parts)


def literature_distill_prompt(
    title: str, abstract: str, sections: Dict[str, str]
) -> Tuple[str, str]:
    system = (
        "You are a research paper analysis expert. "
        "Extract the core problem definition, proposed method, and key limitations. "
        "Return a JSON array of 2-3 observations."
    )
    user = f"""{_paper_context(title, abstract, sections)}

Return a JSON array of 2-3 observations. Each observation must have:
- "type": "method" or "limitation"
- "title": short descriptive title (5-10 words)
- "narrative": 2-3 sentence description
- "concepts": array from ["core_method", "gotcha", "baseline"]
- "confidence": 0.0-1.0 based on how clearly the paper describes this
- "structured_data": dict with relevant key-value pairs (e.g. "problem_definition", "core_approach", "limitation")

First observation should cover the problem and core idea (type=method, concepts=["core_method"]).
Second observation should cover key limitations or gotchas (type=limitation, concepts=["gotcha"]).

Return ONLY the JSON array, no other text."""
    return system, user


def blueprint_extract_prompt(
    title: str, abstract: str, sections: Dict[str, str]
) -> Tuple[str, str]:
    system = (
        "You are a research paper analysis expert specializing in system architecture. "
        "Extract the system/model architecture from the given paper. Return a JSON array."
    )
    user = f"""{_paper_context(title, abstract, sections)}

Return a JSON array of 2-3 observations about the system architecture. Each observation must have:
- "type": "architecture" or "method"
- "title": short descriptive title (5-10 words)
- "narrative": 2-3 sentence description
- "concepts": array from ["core_method", "architecture", "trade_off", "reproduction_hint", "gotcha"]
- "confidence": 0.0-1.0 based on evidence strength
- "structured_data": dict with relevant key-value pairs (e.g. "architecture_type", "key_modules", "design_pattern")

Focus on:
1. Architecture pattern and key components (type=architecture, concepts include "architecture", "core_method")
2. Key design trade-offs (type=architecture, concepts include "trade_off")
3. Implementation details important for reproduction (type=method, concepts include "reproduction_hint")

Return ONLY the JSON array, no other text."""
    return system, user


def environment_extract_prompt(
    title: str, abstract: str, sections: Dict[str, str]
) -> Tuple[str, str]:
    system = (
        "You are a research paper analysis expert specializing in software environments. "
        "Identify the runtime environment, programming languages, frameworks, and dependencies. "
        "Return a JSON array."
    )
    user = f"""{_paper_context(title, abstract, sections)}

Return a JSON array of 1-2 observations about the runtime environment. Each observation must have:
- "type": "environment"
- "title": short descriptive title (5-10 words)
- "narrative": 2-3 sentence description
- "concepts": array from ["environment", "reproduction_hint", "gotcha"]
- "confidence": 0.0-1.0 (higher if explicitly stated, lower if inferred)
- "structured_data": dict with keys like "language", "framework", "dependencies", "python_version", "hardware", "os"

Focus on:
1. Runtime stack: programming language, frameworks, key libraries (concepts include "environment", "reproduction_hint")
2. Build/deployment gotchas if any (concepts include "gotcha")

If the paper doesn't explicitly mention the environment, infer from the domain and methods described.
Set confidence lower (0.5-0.65) for inferred information vs higher (0.75-0.90) for explicitly stated.

Return ONLY the JSON array, no other text."""
    return system, user


def spec_extract_prompt(
    title: str, abstract: str, sections: Dict[str, str]
) -> Tuple[str, str]:
    system = (
        "You are a research paper analysis expert specializing in experimental details. "
        "Extract key hyperparameters, configurations, and experimental settings. "
        "Return a JSON array."
    )
    user = f"""{_paper_context(title, abstract, sections)}

Return a JSON array of 1-3 observations about hyperparameters and specs. Each observation must have:
- "type": "hyperparameter"
- "title": short descriptive title (5-10 words)
- "narrative": 2-3 sentence description listing the key parameters
- "concepts": array from ["hyperparameter", "gotcha", "reproduction_hint"]
- "confidence": 0.0-1.0 (higher if values explicitly stated, lower if typical/inferred)
- "structured_data": dict mapping parameter names to their values (e.g. "learning_rate": "1e-4", "batch_size": "32")

Extract all hyperparameters you can find: learning rate, batch size, epochs, optimizer, regularization, etc.
If values are not explicitly stated, note "not specified" and set lower confidence.

Return ONLY the JSON array, no other text."""
    return system, user


def roadmap_planning_prompt(
    title: str, abstract: str, sections: Dict[str, str]
) -> Tuple[str, str]:
    system = (
        "You are a research paper reproduction expert. "
        "Generate a paper-specific step-by-step reproduction roadmap. "
        "Return a JSON array."
    )
    user = f"""{_paper_context(title, abstract, sections)}

Generate a reproduction roadmap with 4-6 steps tailored to THIS specific paper.
Return a JSON array where each step has:
- "id": "T1", "T2", etc.
- "title": concise step title
- "description": what this step involves (1-2 sentences)
- "acceptance_criteria": array of 1-2 verifiable criteria
- "depends_on": array of step IDs this depends on (empty for first step)
- "estimated_difficulty": "low", "medium", or "high"

Steps should be specific to the paper's method, not generic.
For example, for a SLAM paper: setup camera/sensor pipeline, implement feature extraction, etc.
For a transformer paper: implement attention mechanism, prepare tokenizer, etc.

Return ONLY the JSON array, no other text."""
    return system, user


def success_criteria_prompt(
    title: str, abstract: str, sections: Dict[str, str]
) -> Tuple[str, str]:
    system = (
        "You are a research paper analysis expert specializing in evaluation metrics. "
        "Extract the specific evaluation metrics, target values, datasets, and success criteria. "
        "Return a JSON array."
    )
    user = f"""{_paper_context(title, abstract, sections)}

Return a JSON array of 1-3 observations about success criteria and metrics. Each observation must have:
- "type": "metric"
- "title": short descriptive title (5-10 words)
- "narrative": 2-3 sentence description of the metrics and targets
- "concepts": array from ["baseline", "reproduction_hint", "gotcha"]
- "confidence": 0.0-1.0 (higher if explicit numbers given)
- "structured_data": dict with keys like "metrics" (list), "datasets" (list), "target_values" (dict), "evaluation_protocol"

Focus on:
1. Primary evaluation metrics with target values if available
2. Datasets and splits used for evaluation
3. Comparison baselines mentioned

Return ONLY the JSON array, no other text."""
    return system, user
