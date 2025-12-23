#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory import extract_memories, parse_chat_log


def main() -> int:
    parser = argparse.ArgumentParser(description="Import chat logs and extract long-term memory into PaperBot DB.")
    parser.add_argument("--file", required=True, help="Path to a chat export (json/txt/md)")
    parser.add_argument("--user-id", default="default", help="Memory namespace (one per person/team)")
    parser.add_argument("--platform", default=None, help="Hint: chatgpt/gemini/claude/...")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM for extraction (fallback on heuristics)")
    parser.add_argument("--no-redact", action="store_true", help="Disable basic PII redaction")
    parser.add_argument("--db-url", default=None, help="Override PAPERBOT_DB_URL")
    args = parser.parse_args()

    path = Path(args.file)
    raw = path.read_bytes()

    parsed = parse_chat_log(raw, filename=path.name, platform_hint=args.platform)
    candidates = extract_memories(parsed.messages, use_llm=args.use_llm, redact=(not args.no_redact))

    store = SqlAlchemyMemoryStore(db_url=args.db_url)
    src = store.upsert_source(
        user_id=args.user_id,
        platform=(args.platform or parsed.platform or "unknown"),
        filename=path.name,
        raw_bytes=raw,
        message_count=len(parsed.messages),
        conversation_count=int(parsed.metadata.get("conversation_count") or 0),
        metadata=parsed.metadata,
    )
    created, skipped, _ = store.add_memories(user_id=args.user_id, memories=candidates, source_id=src.id)

    print(f"parsed_messages={len(parsed.messages)} extracted={len(candidates)} created={created} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

