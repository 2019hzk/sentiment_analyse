from typing import Any

from pymilvus import DataType

MILVUS_OUTPUT_FIELDS = [
    "doc_id", "platform", "source_table", "mysql_pk", "record_type",
    "content", "published_at", "source_keyword",
    "like_count", "comment_count", "share_count", "collect_count", "reply_count",
]


def build_collection_schema(client: Any, dense_dim: int) -> Any:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("platform", DataType.VARCHAR, max_length=32)
    schema.add_field("source_table", DataType.VARCHAR, max_length=64)
    schema.add_field("mysql_pk", DataType.INT64)
    schema.add_field("record_type", DataType.VARCHAR, max_length=32)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("published_at", DataType.VARCHAR, max_length=64)
    schema.add_field("source_keyword", DataType.VARCHAR, max_length=512)

    for metric in ["like_count", "comment_count", "share_count", "collect_count", "reply_count"]:
        schema.add_field(metric, DataType.INT64)

    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    return schema


def build_index_params(client: Any) -> Any:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    return index_params
