from paperbot.infrastructure.swarm.paper_slug import paper_slug


def test_slug_from_title_and_id():
    slug = paper_slug("abc123", "Attention Is All You Need")
    assert slug.startswith("attention-is-all-you-need-")
    assert len(slug) <= 40


def test_slug_without_title():
    slug = paper_slug("def456")
    assert slug.startswith("paper-")
    assert len(slug) <= 40


def test_slug_uniqueness():
    slug_a = paper_slug("id-aaa", "Same Title")
    slug_b = paper_slug("id-bbb", "Same Title")
    assert slug_a != slug_b


def test_slug_deterministic():
    assert paper_slug("x", "Y") == paper_slug("x", "Y")


def test_slug_strips_special_characters():
    slug = paper_slug("id1", "ResNet: Deep Residual Learning (2015)!")
    assert ":" not in slug
    assert "(" not in slug
    assert "!" not in slug


def test_slug_truncation():
    long_title = "A " * 100
    slug = paper_slug("id2", long_title, max_length=30)
    assert len(slug) <= 30


def test_slug_no_double_hyphens():
    slug = paper_slug("id3", "Hello --- World")
    assert "--" not in slug


def test_slug_filesystem_safe():
    slug = paper_slug("id4", "What/Why\\How?*<>|")
    for ch in "/\\?*<>|":
        assert ch not in slug
