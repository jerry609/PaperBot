from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional

import aiohttp

from paperbot.domain.paper_identity import normalize_arxiv_id, normalize_doi
from paperbot.infrastructure.api_clients.semantic_scholar import SemanticScholarClient
from paperbot.infrastructure.connectors.arxiv_connector import ArxivConnector

from .models import NormalizedInput, PaperIdentity, PaperType, RawPaperData

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

    _DEFAULT_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "authors",
        "externalIds",
    ]

    def __init__(
        self,
        *,
        client: Optional[SemanticScholarClient] = None,
        api_key: Optional[str] = None,
    ):
        self._client = client
        self._api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")

    def can_handle(self, paper_id: str) -> bool:
        pid = paper_id.strip()
        pid_upper = pid.upper()
        return (
            pid.startswith("s2:")
            or bool(_HEX40_RE.match(pid))
            or pid_upper.startswith("DOI:")
            or pid_upper.startswith("ARXIV:")
            or pid_upper.startswith("CORPUSID:")
            or normalize_doi(pid) is not None
        )

    @staticmethod
    def _normalize_lookup_id(paper_id: str) -> str:
        pid = paper_id.strip()
        if pid.startswith("s2:"):
            return pid.split(":", 1)[1].strip()

        pid_upper = pid.upper()
        if pid_upper.startswith(("DOI:", "ARXIV:", "CORPUSID:")):
            prefix, value = pid.split(":", 1)
            return f"{prefix.upper()}:{value.strip()}"

        doi = normalize_doi(pid)
        if doi:
            return f"DOI:{doi}"

        arxiv_id = normalize_arxiv_id(pid)
        if arxiv_id:
            return f"ARXIV:{arxiv_id}"

        return pid

    async def fetch(self, paper_id: str) -> RawPaperData:
        lookup_id = self._normalize_lookup_id(paper_id)
        client = self._client
        owns_client = False
        if client is None:
            client = SemanticScholarClient(api_key=self._api_key)
            owns_client = True

        try:
            record = await client.get_paper(lookup_id, fields=self._DEFAULT_FIELDS)
        finally:
            if owns_client:
                await client.close()

        if not record:
            raise ValueError(f"Semantic Scholar paper not found: {lookup_id}")

        authors = [
            str(author.get("name")).strip()
            for author in (record.get("authors") or [])
            if isinstance(author, dict) and author.get("name")
        ]

        year = 0
        year_raw = record.get("year")
        if year_raw is not None:
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                year = 0

        external_ids = (
            record.get("externalIds") if isinstance(record.get("externalIds"), dict) else {}
        )
        identifiers: dict[str, str] = {}
        paper_id_value = str(record.get("paperId") or "").strip()
        if paper_id_value:
            identifiers["semantic_scholar"] = paper_id_value

        doi = normalize_doi(external_ids.get("DOI"))
        if doi:
            identifiers["doi"] = doi

        arxiv_id = normalize_arxiv_id(external_ids.get("ArXiv") or external_ids.get("ARXIV"))
        if arxiv_id:
            identifiers["arxiv"] = arxiv_id

        normalized_paper_id = f"s2:{paper_id_value}" if paper_id_value else lookup_id
        return RawPaperData(
            paper_id=normalized_paper_id,
            title=str(record.get("title") or "").strip(),
            abstract=str(record.get("abstract") or "").strip(),
            authors=authors,
            year=year,
            identifiers=identifiers,
            source_adapter=self.name,
        )


class ArXivAdapter(PaperInputAdapter):
    name = "arxiv"
    API_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        connector: Optional[ArxivConnector] = None,
        timeout_seconds: float = 30.0,
        fetch_atom_xml: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        self._connector = connector or ArxivConnector(timeout_s=timeout_seconds)
        self._timeout_seconds = timeout_seconds
        self._fetch_atom_xml_override = fetch_atom_xml

    def can_handle(self, paper_id: str) -> bool:
        return normalize_arxiv_id(paper_id) is not None

    async def _fetch_atom_xml(self, arxiv_id: str) -> str:
        if self._fetch_atom_xml_override is not None:
            return await self._fetch_atom_xml_override(arxiv_id)

        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        headers = {"User-Agent": "PaperBot/2.0"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(self.API_URL, params=params) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"arXiv API error {response.status}: {body[:200]}")
                return await response.text()

    async def fetch(self, paper_id: str) -> RawPaperData:
        arxiv_id = normalize_arxiv_id(paper_id)
        if not arxiv_id:
            raise ValueError(f"Invalid arXiv paper id: {paper_id}")

        xml_text = await self._fetch_atom_xml(arxiv_id)
        records = self._connector.parse_atom(xml_text)
        if not records:
            raise ValueError(f"arXiv paper not found: {arxiv_id}")

        record = records[0]
        year = 0
        published = str(record.published or "").strip()
        if len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                year = 0

        normalized_id = normalize_arxiv_id(record.arxiv_id) or arxiv_id
        return RawPaperData(
            paper_id=f"arxiv:{normalized_id}",
            title=str(record.title or "").strip(),
            abstract=str(record.summary or "").strip(),
            authors=[name.strip() for name in record.authors if str(name).strip()],
            year=year,
            identifiers={"arxiv": normalized_id},
            source_adapter=self.name,
        )


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


class InternalPaperStoreAdapter(PaperInputAdapter):
    name = "paper_store"

    def __init__(self) -> None:
        from paperbot.infrastructure.stores.paper_store import PaperStore

        self._store = PaperStore()

    def can_handle(self, paper_id: str) -> bool:
        return paper_id.strip().isdigit()

    async def fetch(self, paper_id: str) -> RawPaperData:
        from paperbot.utils.logging_config import Logger, LogFiles

        paper_db_id = int(paper_id)
        row = self._store.get_paper_by_id(paper_db_id)
        if row is None:
            Logger.error(
                f"[M1] paper_store_not_found paper_id={paper_id}",
                file=LogFiles.ERROR,
            )
            raise ValueError(f"Paper not found in store: {paper_id}")

        identifiers: dict[str, str] = {}
        if row.semantic_scholar_id:
            identifiers["semantic_scholar"] = row.semantic_scholar_id
        if row.arxiv_id:
            identifiers["arxiv"] = row.arxiv_id
        if row.doi:
            identifiers["doi"] = row.doi
        if row.openalex_id:
            identifiers["openalex"] = row.openalex_id

        Logger.info(
            f"[M1] paper_store_fetch paper_id={paper_id} title={row.title}",
            file=LogFiles.API,
        )

        return RawPaperData(
            paper_id=str(row.id),
            title=row.title or "",
            abstract=row.abstract or "",
            authors=row.get_authors(),
            year=int(row.year or 0),
            identifiers=identifiers,
            source_adapter=self.name,
        )


class PaperInputRouter:
    def __init__(self, adapters: Optional[Iterable[PaperInputAdapter]] = None):
        self.adapters: List[PaperInputAdapter] = list(
            adapters
            or [
                InternalPaperStoreAdapter(),
                LocalFileAdapter(),
                ArXivAdapter(),
                SemanticScholarAdapter(),
            ]
        )

    async def fetch(self, paper_id: str) -> RawPaperData:
        from paperbot.utils.logging_config import Logger, LogFiles

        Logger.info(
            f"[M1] input_router_fetch paper_id={paper_id} adapters={[a.name for a in self.adapters]}",
            file=LogFiles.API,
        )
        for adapter in self.adapters:
            if adapter.can_handle(paper_id):
                Logger.info(
                    f"[M1] adapter_selected paper_id={paper_id} adapter={adapter.name}",
                    file=LogFiles.API,
                )
                return await adapter.fetch(paper_id)
        Logger.error(
            f"[M1] no_adapter_found paper_id={paper_id} adapters={[a.name for a in self.adapters]}",
            file=LogFiles.ERROR,
        )
        raise ValueError(f"No adapter can handle paper id: {paper_id}")


class PaperSectionExtractor:
    """Heuristic section extractor for module1 baseline."""

    _heading_patterns = {
        "introduction": ["introduction", "background", "related work"],
        "method": ["method", "methods", "approach", "model", "methodology"],
        "results": [
            "results",
            "experiments",
            "experimental setup",
            "evaluation",
            "analysis",
        ],
        "discussion": ["discussion", "conclusion", "future work", "limitations"],
    }
    _heading_prefix_re = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\).\s:-]+", re.IGNORECASE)

    def _normalize_heading_text(self, line: str) -> str:
        normalized = line.strip().lower()
        normalized = self._heading_prefix_re.sub("", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" :.-")
        return normalized

    def _is_heading(self, line: str) -> Optional[str]:
        original = line.strip()
        if not original:
            return None
        if len(original) > 120:
            return None
        if original.endswith((".", "?", "!")) and ":" not in original:
            return None
        if len(original.split()) > 8:
            return None

        normalized = self._normalize_heading_text(line)
        if not normalized:
            return None
        for section, aliases in self._heading_patterns.items():
            for alias in aliases:
                if (
                    normalized == alias
                    or normalized.startswith(f"{alias} and ")
                    or normalized.endswith(f" {alias}")
                ):
                    return section
        return None

    @staticmethod
    def _merge_offset_range(
        offsets: dict[str, tuple[int, int]],
        *,
        section: str,
        start: int,
        end: int,
    ) -> None:
        if start >= end:
            return
        if section in offsets:
            old_start, old_end = offsets[section]
            offsets[section] = (min(old_start, start), max(old_end, end))
        else:
            offsets[section] = (start, end)

    def _split_sections(self, text: str) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
        sections: dict[str, List[str]] = {k: [] for k in self._heading_patterns.keys()}
        offsets: dict[str, tuple[int, int]] = {}
        current: Optional[str] = None
        current_start: Optional[int] = None
        cursor = 0

        for raw_line in text.splitlines(keepends=True):
            line = raw_line.strip()
            heading = self._is_heading(line)
            if heading:
                if current and current_start is not None:
                    self._merge_offset_range(
                        offsets,
                        section=current,
                        start=current_start,
                        end=cursor,
                    )
                current = heading
                current_start = cursor + len(raw_line)
                cursor += len(raw_line)
                continue
            if current:
                sections[current].append(raw_line.rstrip("\n"))
            cursor += len(raw_line)

        if current and current_start is not None:
            self._merge_offset_range(
                offsets,
                section=current,
                start=current_start,
                end=cursor,
            )

        compact = {name: "\n".join(lines).strip() for name, lines in sections.items() if lines}
        offsets = {name: span for name, span in offsets.items() if name in compact}
        return compact, offsets

    def _segment_noisy_text(self, text: str) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
        lowered = text.lower()
        anchors: list[tuple[int, int, str]] = []

        for section, aliases in self._heading_patterns.items():
            best_match: Optional[tuple[int, int, str]] = None
            for alias in aliases:
                match = re.search(rf"\b{re.escape(alias)}\b", lowered)
                if match:
                    candidate = (match.start(), match.end(), section)
                    if best_match is None or candidate[0] < best_match[0]:
                        best_match = candidate
            if best_match:
                anchors.append(best_match)

        if len(anchors) < 2:
            return {}, {}

        anchors.sort(key=lambda item: item[0])
        sections: dict[str, str] = {}
        offsets: dict[str, tuple[int, int]] = {}
        for idx, (_, anchor_end, section) in enumerate(anchors):
            next_start = anchors[idx + 1][0] if idx + 1 < len(anchors) else len(text)
            snippet = text[anchor_end:next_start].strip()
            if not snippet:
                continue
            if section in sections:
                sections[section] = f"{sections[section]}\n{snippet}".strip()
                old_start, _ = offsets[section]
                offsets[section] = (old_start, next_start)
            else:
                sections[section] = snippet
                offsets[section] = (anchor_end, next_start)
        return sections, offsets

    async def extract(self, raw: RawPaperData) -> NormalizedInput:
        full_text = (raw.full_text or "").strip()
        abstract = raw.abstract.strip()

        sections, section_offsets = self._split_sections(full_text)
        if not sections and full_text:
            sections, section_offsets = self._segment_noisy_text(full_text)

        if not sections and full_text:
            # Last fallback: keep a best-effort method slice.
            method_hint = re.search(
                r"(method|approach|model|methodology)([\s\S]{0,1200})",
                full_text,
                flags=re.IGNORECASE,
            )
            if method_hint:
                sections["method"] = method_hint.group(2).strip()
                section_offsets["method"] = (method_hint.start(2), method_hint.end(2))

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
            section_offsets=section_offsets,
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
