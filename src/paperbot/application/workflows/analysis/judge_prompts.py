from __future__ import annotations

from typing import Any, Dict, Sequence

from paperbot.application.workflows.analysis.judge_rubrics import JudgeRubric


PAPER_JUDGE_SYSTEM = (
    "You are an expert research paper evaluator. "
    "Reason carefully and return strict JSON only."
)


def build_paper_judge_user_prompt(*, query: str, paper: Dict[str, Any], rubric: JudgeRubric) -> str:
    title = paper.get("title") or ""
    abstract = paper.get("snippet") or paper.get("abstract") or ""
    authors = ", ".join(paper.get("authors") or [])
    venue = paper.get("subject_or_venue") or paper.get("venue") or ""
    keywords = ", ".join(paper.get("keywords") or [])

    rubric_blocks = []
    for idx, dim in enumerate(rubric.dimensions, start=1):
        lines = [
            f"### {idx}. {dim.label} (weight: {int(dim.weight * 100)}%)",
            *[f"- {score}: {text}" for score, text in sorted(dim.rubric.items(), reverse=True)],
        ]
        rubric_blocks.append("\n".join(lines))

    dims_json = ",\n    ".join(
        [f'"{dim.key}": {{"score": <1-5>, "rationale": "<1-2 sentences>"}}' for dim in rubric.dimensions]
    )

    return (
        "Evaluate the following paper against the research query.\n\n"
        f"## Research Query\n{query}\n\n"
        "## Paper Information\n"
        f"- Title: {title}\n"
        f"- Abstract: {abstract}\n"
        f"- Authors: {authors}\n"
        f"- Venue/Subject: {venue}\n"
        f"- Keywords: {keywords}\n\n"
        "Use integer scores 1-5. Abstract length should not affect scoring.\n\n"
        "## Rubric\n"
        f"{'\n\n'.join(rubric_blocks)}\n\n"
        "## Output Format (strict JSON)\n"
        "{\n"
        f"    {dims_json},\n"
        '    "overall": <weighted float 1.0-5.0>,\n'
        '    "one_line_summary": "<one sentence takeaway>",\n'
        '    "recommendation": "<must_read|worth_reading|skim|skip>"\n'
        "}\n"
    )


def dimension_keys(rubric: JudgeRubric) -> Sequence[str]:
    return [dim.key for dim in rubric.dimensions]
