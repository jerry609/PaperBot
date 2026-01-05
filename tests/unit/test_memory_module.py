from __future__ import annotations

import json

from paperbot.infrastructure.stores.memory_store import SqlAlchemyMemoryStore
from paperbot.memory.extractor import extract_memories
from paperbot.memory.parsers.common import parse_chat_log


def test_parse_chatgpt_conversations_export_minimal():
    export = [
        {
            "id": "c1",
            "mapping": {
                "n1": {
                    "id": "n1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"parts": ["我叫 Jerry"]},
                        "create_time": 1700000000,
                    },
                },
                "n2": {
                    "id": "n2",
                    "message": {
                        "id": "m2",
                        "author": {"role": "assistant"},
                        "content": {"parts": ["你好！"]},
                        "create_time": 1700000001,
                    },
                },
            },
        }
    ]
    parsed = parse_chat_log(json.dumps(export, ensure_ascii=False).encode("utf-8"), filename="conversations.json")
    assert parsed.platform == "chatgpt"
    assert len(parsed.messages) == 2
    assert parsed.messages[0].role == "user"
    assert "Jerry" in parsed.messages[0].content


def test_plaintext_parser_and_heuristic_extraction():
    raw = "User: 我计划做一个中间件\nAssistant: 好的\nUser: 我不喜欢太长的回答\n".encode("utf-8")
    parsed = parse_chat_log(raw, filename="chat.txt")
    assert len(parsed.messages) >= 2

    mems = extract_memories(parsed.messages, use_llm=False, redact=True)
    kinds = {m.kind for m in mems}
    assert "goal" in kinds
    assert "preference" in kinds


def test_loose_json_gemini_api_style():
    obj = {
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": "我想做一个中间件"}]},
            ]
        },
        "response": {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "可以，先定义数据模型"}]}},
            ]
        },
    }
    parsed = parse_chat_log(json.dumps(obj, ensure_ascii=False).encode("utf-8"), filename="gemini.json", platform_hint="gemini")
    assert len(parsed.messages) >= 2
    assert parsed.messages[0].role in {"user", "unknown"}
    assert any("中间件" in m.content for m in parsed.messages)


def test_memory_store_dedup_by_user_and_hash(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'mem.db'}"
    store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

    raw = "User: 我叫 Jerry".encode("utf-8")
    parsed = parse_chat_log(raw, filename="a.txt")
    src = store.upsert_source(
        user_id="u1",
        platform="test",
        filename="a.txt",
        raw_bytes=raw,
        message_count=len(parsed.messages),
        conversation_count=0,
        metadata={},
    )

    mems = extract_memories(parsed.messages, use_llm=False)
    created1, skipped1, _ = store.add_memories(user_id="u1", memories=mems, source_id=src.id)
    created2, skipped2, _ = store.add_memories(user_id="u1", memories=mems, source_id=src.id)

    assert created1 >= 1
    assert created2 == 0
    assert skipped2 >= 1


def test_memory_store_soft_delete_and_status_filter(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'mem2.db'}"
    store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

    raw = "User: 我不喜欢太长的回答".encode("utf-8")
    parsed = parse_chat_log(raw, filename="a.txt")
    src = store.upsert_source(
        user_id="u1",
        platform="test",
        filename="a.txt",
        raw_bytes=raw,
        message_count=len(parsed.messages),
        conversation_count=0,
        metadata={},
    )
    mems = extract_memories(parsed.messages, use_llm=False)
    created, _, rows = store.add_memories(user_id="u1", memories=mems, source_id=src.id)
    assert created >= 1

    # Find the preference item and mark it pending; it should no longer be returned by search_memories.
    pref = next((r for r in rows if r.kind == "preference"), None)
    assert pref is not None
    store.update_item(user_id="u1", item_id=int(pref.id), status="pending")
    assert not any("太长" in i["content"] for i in store.search_memories(user_id="u1", query="太长", limit=10))

    # Approve then soft-delete; it should stay out of results.
    store.update_item(user_id="u1", item_id=int(pref.id), status="approved")
    assert any("太长" in i["content"] for i in store.search_memories(user_id="u1", query="太长", limit=10))
    store.soft_delete_item(user_id="u1", item_id=int(pref.id), reason="user request")
    assert not any("太长" in i["content"] for i in store.search_memories(user_id="u1", query="太长", limit=10))


def test_memory_store_get_items_by_ids(tmp_path):
    """Test get_items_by_ids method retrieves correct items."""
    from paperbot.memory.schema import MemoryCandidate

    db_url = f"sqlite:///{tmp_path / 'mem_ids.db'}"
    store = SqlAlchemyMemoryStore(db_url=db_url, auto_create_schema=True)

    # Create test memories
    candidates = [
        MemoryCandidate(kind="preference", content="Prefers Python", confidence=0.8),
        MemoryCandidate(kind="goal", content="Learn Rust", confidence=0.7),
        MemoryCandidate(kind="fact", content="Uses VSCode", confidence=0.6),
    ]
    created, _, rows = store.add_memories(user_id="u1", memories=candidates)
    assert created == 3

    # Get IDs from list_memories since rows are detached
    all_items = store.list_memories(user_id="u1", limit=10)
    item_ids = [item["id"] for item in all_items]
    assert len(item_ids) == 3

    # Test get_items_by_ids
    retrieved = store.get_items_by_ids(user_id="u1", item_ids=item_ids[:2])
    assert len(retrieved) == 2

    # Test with empty list
    empty = store.get_items_by_ids(user_id="u1", item_ids=[])
    assert len(empty) == 0

    # Test user isolation - u2 can't see u1's items
    other_user = store.get_items_by_ids(user_id="u2", item_ids=item_ids)
    assert len(other_user) == 0
