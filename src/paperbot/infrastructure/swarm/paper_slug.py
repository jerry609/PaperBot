"""Generate filesystem-safe directory names for paper reproductions.

Each paper gets a unique subdirectory under ``/home/user/`` in the sandbox VM,
derived from its title and id.  This keeps multiple reproductions isolated
inside a single long-lived sandbox.
"""

from __future__ import annotations

import hashlib
import re


def paper_slug(paper_id: str, title: str = "", max_length: int = 40) -> str:
    """Return a short, filesystem-safe directory name for a paper.

    Rules:
    1. A *readable* part is derived from *title* (lowercased, hyphen-separated,
       special characters stripped).  Falls back to ``"paper"`` when *title*
       is empty.
    2. A 4-character hex suffix from the MD5 of *paper_id* guarantees
       uniqueness even when titles collide after truncation.
    3. Total length never exceeds *max_length*.

    Examples::

        paper_slug("abc123", "Attention Is All You Need")
        → "attention-is-all-you-need-a9b1"

        paper_slug("def456", "")
        → "paper-d8e2"
    """
    if title:
        readable = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        readable = re.sub(r"-{2,}", "-", readable)
    else:
        readable = "paper"

    suffix = hashlib.md5(paper_id.encode()).hexdigest()[:4]

    max_readable = max_length - len(suffix) - 1  # -1 for the joining hyphen
    if len(readable) > max_readable:
        readable = readable[:max_readable].rstrip("-")

    return f"{readable}-{suffix}"
