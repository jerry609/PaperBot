from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

from .models import NormalizedInput, PaperIdentity, PaperType, RawPaperData

_ARXIV_ID_RE = re.compile(r"(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


class PaperInputAdapter(ABC):
    name: str = ""

    @abstractmethod
    def can_handle(self, paper_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def fetch(self, paper_id: str) -> RawPaperData:
        raise NotImplementedError


class SemanticScholarAdapter(PaperInputAdapter):
    name = "semantic_scholar"

    def can_handle(self, paper_id: str) -> bool:
        pid = paper_id.strip()
        return pid.startswith("s2:") or bool(_HEX40_RE.match(pid))

    async def fetch(self, paper_id: str) -> RawPaperData:
        raise NotImplementedError(
            "SemanticScholarAdapter fetch is implemented in Module 2 integration"
        )


class ArXivAdapter(PaperInputAdapter):
    name = "arxiv"

    def can_handle(self, paper_id: str) -> bool:
        return bool(_ARXIV_ID_RE.search(paper_id.strip()))

    async def fetch(self, paper_id: str) -> RawPaperData:
        raise NotImplementedError("ArXivAdapter fetch is implemented in Module 2 integration")


class LocalFileAdapter(PaperInputAdapter):
    name = "local_file"

    def can_handle(self, paper_id: str) -> bool:
        path = Path(paper_id).expanduser()
        return path.exists() and path.is_file()

    async def fetch(self, paper_id: str) -> RawPaperData:
        path = Path(paper_id).expanduser()
        text = ""
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")

        return RawPaperData(
            paper_id=str(path),
            title=path.stem,
            full_text=text or None,
            source_adapter=self.name,
        )


class PaperInputRouter:
    def __init__(self, adapters: Optional[Iterable[PaperInputAdapter]] = None):
        self.adapters: List[PaperInputAdapter] = list(
            adapters or [LocalFileAdapter(), ArXivAdapter(), SemanticScholarAdapter()]
        )

    async def fetch(self, paper_id: str) -> RawPaperData:
        for adapter in self.adapters:
            if adapter.can_handle(paper_id):
                return await adapter.fetch(paper_id)
        raise ValueError(f"No adapter can handle paper id: {paper_id}")


class PaperSectionExtractor:
    """Heuristic section extractor for module1 baseline."""

    _heading_patterns = {
        "introduction": ["introduction", "background"],
        "method": ["method", "approach", "model", "methodology"],
        "results": ["results", "experiments", "evaluation"],
        "discussion": ["discussion", "conclusion"],
    }

    def _is_heading(self, line: str) -> Optional[str]:
        normalized = line.strip().lower().strip(":")
        if not normalized:
            return None
        for section, aliases in self._heading_patterns.items():
            if normalized in aliases:
                return section
        return None

    def _split_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, List[str]] = {k: [] for k in self._heading_patterns.keys()}
        current: Optional[str] = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            heading = self._is_heading(line)
            if heading:
                current = heading
                continue
            if current:
                sections[current].append(raw_line)

        compact = {name: "\n".join(lines).strip() for name, lines in sections.items() if lines}
        return compact

    async def extract(self, raw: RawPaperData) -> NormalizedInput:
        full_text = (raw.full_text or "").strip()
        abstract = raw.abstract.strip()

        sections = self._split_sections(full_text)
        if not sections and full_text:
            # Fallback: we only know we have text, so keep a best-effort method slice.
            method_hint = re.search(
                r"(method|approach|model|methodology)([\s\S]{0,1200})",
                full_text,
                flags=re.IGNORECASE,
            )
            if method_hint:
                sections["method"] = method_hint.group(2).strip()

        identity = PaperIdentity(
            paper_id=raw.paper_id,
            title=raw.title,
            year=raw.year,
            authors=list(raw.authors),
            identifiers=dict(raw.identifiers),
        )

        return NormalizedInput(
            paper=identity,
            abstract=abstract,
            full_text=full_text,
            sections=sections,
        )


class PaperTypeClassifier:
    def classify(self, normalized: NormalizedInput) -> PaperType:
        text = " ".join(
            [
                normalized.paper.title.lower(),
                normalized.abstract.lower(),
                normalized.sections.get("method", "").lower(),
            ]
        )

        if any(token in text for token in ["survey", "systematic review", "overview"]):
            return PaperType.SURVEY
        if any(token in text for token in ["theorem", "proof", "bound", "convergence"]):
            return PaperType.THEORETICAL
        if any(token in text for token in ["benchmark", "leaderboard", "sota"]):
            return PaperType.BENCHMARK
        if any(token in text for token in ["system", "platform", "serving", "deployment"]):
            return PaperType.SYSTEM
        return PaperType.EXPERIMENTAL
