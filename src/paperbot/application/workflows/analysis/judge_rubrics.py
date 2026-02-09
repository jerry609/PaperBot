from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class JudgeDimension:
    key: str
    label: str
    weight: float
    rubric: Dict[int, str]


@dataclass(frozen=True)
class JudgeRubric:
    dimensions: List[JudgeDimension]

    def weights(self) -> Dict[str, float]:
        return {dim.key: dim.weight for dim in self.dimensions}


def default_judge_rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[
            JudgeDimension(
                key="relevance",
                label="Relevance",
                weight=0.30,
                rubric={
                    5: "Directly targets the query core problem with meaningful contribution.",
                    4: "Highly related to important sub-problems or close techniques.",
                    3: "Moderately related but not central to the query focus.",
                    2: "Weak or peripheral relation.",
                    1: "Not relevant.",
                },
            ),
            JudgeDimension(
                key="novelty",
                label="Novelty",
                weight=0.25,
                rubric={
                    5: "Paradigm-level or clearly groundbreaking idea.",
                    4: "Substantial innovation over prior methods.",
                    3: "Incremental but useful improvement.",
                    2: "Limited innovation, mostly reuse/application.",
                    1: "No clear novelty.",
                },
            ),
            JudgeDimension(
                key="rigor",
                label="Technical Rigor",
                weight=0.20,
                rubric={
                    5: "Strong methodology, ablations, and comprehensive validation.",
                    4: "Methodologically sound with decent evidence.",
                    3: "Acceptable but missing notable checks.",
                    2: "Weak validation or methodological gaps.",
                    1: "Fundamentally flawed methodology.",
                },
            ),
            JudgeDimension(
                key="impact",
                label="Impact Potential",
                weight=0.15,
                rubric={
                    5: "Likely to shape future direction of the field.",
                    4: "Likely to influence follow-up work significantly.",
                    3: "Useful contribution with moderate influence.",
                    2: "Niche contribution with limited influence.",
                    1: "Low expected impact.",
                },
            ),
            JudgeDimension(
                key="clarity",
                label="Clarity",
                weight=0.10,
                rubric={
                    5: "Very clear and easy to follow.",
                    4: "Mostly clear with minor issues.",
                    3: "Understandable but uneven clarity.",
                    2: "Hard to follow in key parts.",
                    1: "Very unclear.",
                },
            ),
        ]
    )
