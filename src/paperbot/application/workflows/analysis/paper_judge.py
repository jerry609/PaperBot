from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from paperbot.application.services.llm_service import LLMService, get_llm_service
from paperbot.application.workflows.analysis.judge_prompts import (
    PAPER_JUDGE_SYSTEM,
    build_paper_judge_user_prompt,
    dimension_keys,
)
from paperbot.application.workflows.analysis.judge_rubrics import (
    JudgeRubric,
    default_judge_rubric,
)


@dataclass
class DimensionScore:
    score: int
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {"score": int(self.score), "rationale": self.rationale}


@dataclass
class PaperJudgment:
    relevance: DimensionScore
    novelty: DimensionScore
    rigor: DimensionScore
    impact: DimensionScore
    clarity: DimensionScore
    overall: float
    one_line_summary: str
    recommendation: str
    judge_model: str = ""
    judge_cost_tier: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance": self.relevance.to_dict(),
            "novelty": self.novelty.to_dict(),
            "rigor": self.rigor.to_dict(),
            "impact": self.impact.to_dict(),
            "clarity": self.clarity.to_dict(),
            "overall": round(float(self.overall), 4),
            "one_line_summary": self.one_line_summary,
            "recommendation": self.recommendation,
            "judge_model": self.judge_model,
            "judge_cost_tier": int(self.judge_cost_tier),
        }


class PaperJudge:
    """LLM-as-Judge scorer for DailyPaper ranking."""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        rubric: Optional[JudgeRubric] = None,
    ) -> None:
        self._llm = llm_service or get_llm_service()
        self._rubric = rubric or default_judge_rubric()
        self._dim_keys = list(dimension_keys(self._rubric))

    def judge_single(self, *, paper: Dict[str, Any], query: str) -> PaperJudgment:
        prompt = build_paper_judge_user_prompt(query=query, paper=paper, rubric=self._rubric)
        raw = self._llm.complete(
            task_type="analysis",
            system=PAPER_JUDGE_SYSTEM,
            user=prompt,
            temperature=0.1,
        )
        payload = self._parse_payload(raw)
        provider_info = self._llm.describe_task_provider("analysis")
        return self._to_judgment(payload=payload, provider_info=provider_info)

    def judge_with_calibration(
        self,
        *,
        paper: Dict[str, Any],
        query: str,
        n_runs: int = 3,
    ) -> PaperJudgment:
        runs = max(1, int(n_runs))
        judgments = [self.judge_single(paper=paper, query=query) for _ in range(runs)]
        if len(judgments) == 1:
            return judgments[0]

        def pick_recommendation(values: Sequence[str]) -> str:
            rank = {"must_read": 4, "worth_reading": 3, "skim": 2, "skip": 1}
            return sorted(values, key=lambda item: rank.get(item, 0), reverse=True)[0]

        dim_medians: Dict[str, int] = {}
        for key in self._dim_keys:
            dim_medians[key] = int(
                median([int(getattr(j, key).score) for j in judgments])
            )

        payload = {
            key: {"score": score, "rationale": "Median-calibrated from multiple judge runs."}
            for key, score in dim_medians.items()
        }
        payload["overall"] = self._weighted_overall(dim_medians)
        payload["one_line_summary"] = judgments[0].one_line_summary
        payload["recommendation"] = pick_recommendation([j.recommendation for j in judgments])

        provider_info = {
            "model_name": judgments[0].judge_model,
            "cost_tier": judgments[0].judge_cost_tier,
        }
        return self._to_judgment(payload=payload, provider_info=provider_info)

    def judge_batch(
        self,
        *,
        papers: Sequence[Dict[str, Any]],
        query: str,
        n_runs: int = 1,
    ) -> List[PaperJudgment]:
        out: List[PaperJudgment] = []
        for paper in papers:
            if n_runs > 1:
                out.append(self.judge_with_calibration(paper=paper, query=query, n_runs=n_runs))
            else:
                out.append(self.judge_single(paper=paper, query=query))
        return out

    def _parse_payload(self, raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return {}

    def _to_judgment(self, *, payload: Dict[str, Any], provider_info: Dict[str, Any]) -> PaperJudgment:
        dims: Dict[str, DimensionScore] = {}
        for key in self._dim_keys:
            raw_dim = payload.get(key) if isinstance(payload.get(key), dict) else {}
            raw_score = raw_dim.get("score", 3)
            try:
                score = int(raw_score)
            except Exception:
                score = 3
            score = max(1, min(score, 5))
            rationale = str(raw_dim.get("rationale") or "")
            dims[key] = DimensionScore(score=score, rationale=rationale)

        if "overall" in payload:
            try:
                overall = float(payload.get("overall") or 0)
            except Exception:
                overall = 0.0
            if overall <= 0:
                overall = self._weighted_overall({k: v.score for k, v in dims.items()})
        else:
            overall = self._weighted_overall({k: v.score for k, v in dims.items()})
        overall = max(1.0, min(5.0, float(overall)))

        recommendation = (str(payload.get("recommendation") or "") or self._recommendation(overall)).strip().lower()
        if recommendation not in {"must_read", "worth_reading", "skim", "skip"}:
            recommendation = self._recommendation(overall)

        one_line_summary = str(payload.get("one_line_summary") or "")
        if not one_line_summary:
            one_line_summary = self._fallback_summary(dims)

        return PaperJudgment(
            relevance=dims["relevance"],
            novelty=dims["novelty"],
            rigor=dims["rigor"],
            impact=dims["impact"],
            clarity=dims["clarity"],
            overall=overall,
            one_line_summary=one_line_summary,
            recommendation=recommendation,
            judge_model=str(provider_info.get("model_name") or ""),
            judge_cost_tier=int(provider_info.get("cost_tier") or 0),
        )

    def _weighted_overall(self, scores: Dict[str, int]) -> float:
        weights = self._rubric.weights()
        total = 0.0
        for key, weight in weights.items():
            total += float(scores.get(key, 3)) * float(weight)
        return round(total, 4)

    def _recommendation(self, overall: float) -> str:
        if overall >= 4.3:
            return "must_read"
        if overall >= 3.6:
            return "worth_reading"
        if overall >= 2.8:
            return "skim"
        return "skip"

    def _fallback_summary(self, dims: Dict[str, DimensionScore]) -> str:
        return (
            "Judge summary: relevance={relevance}, novelty={novelty}, rigor={rigor}, "
            "impact={impact}, clarity={clarity}."
        ).format(
            relevance=dims["relevance"].score,
            novelty=dims["novelty"].score,
            rigor=dims["rigor"].score,
            impact=dims["impact"].score,
            clarity=dims["clarity"].score,
        )
