from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    id: int
    email: Optional[str]
    github_id: Optional[str]
    github_username: Optional[str]
    display_name: Optional[str]
    avatar_url: Optional[str]
    is_active: bool
    created_at: datetime

