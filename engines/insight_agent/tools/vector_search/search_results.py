from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorDocument:

    doc_id: str
    platform: str
    source_table: str
    mysql_pk: int
    record_type: str
    content: str    # 向量的内容（两个平台对应内容表【1.douyin:title 2.weibo:content】以及评论表[1.douyin:content 2.weibo:content]的数据）

    published_at: str
    source_keyword: str

    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    collect_count: int = 0
    reply_count: int = 0

    def to_milvus_record(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
    ) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "platform": self.platform,
            "source_table": self.source_table,
            "mysql_pk": self.mysql_pk,
            "record_type": self.record_type,
            "content": self.content,
            "published_at": self.published_at,
            "source_keyword": self.source_keyword,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "share_count": self.share_count,
            "collect_count": self.collect_count,
            "reply_count": self.reply_count,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }


@dataclass(frozen=True)
class SearchHit:
    """向量检索命中结果。"""

    doc_id: str
    score: float
    channel: str
    entity: dict[str, Any] = field(default_factory=dict)   # output_filed
