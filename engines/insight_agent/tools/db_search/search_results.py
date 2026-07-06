from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class SearchRecord:
    mysql_pk: int
    platform: str
    source_table: str
    title_or_content: str
    published_at: datetime
    source_keyword: str

    engagement: dict[str, int] = field(default_factory=dict)
    hotness_score: float = None


@dataclass
class SearchResponse:
    retrieval_channel: str
    search_results: list[SearchRecord] = field(default_factory=list)
    search_results_count: int = 0
    search_error_message: Optional[str] = None
