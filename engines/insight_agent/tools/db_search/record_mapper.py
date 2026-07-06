from datetime import datetime
from typing import Any, Mapping

from engines.insight_agent.tools.db_search.search_results import SearchRecord
from engines.insight_agent.tools.platform_mappings import PLATFORM_MAPPING


def db_row_to_search_record(result_row: dict[str, Any]) -> SearchRecord:
    """将 db_search 查询行映射为统一的 SearchRecord。"""
    platform_name = result_row.get("platform")
    platform_mapping = PLATFORM_MAPPING.get(platform_name)

    is_comment = result_row.get("source_table") == platform_mapping.comment_mapping.table_name
    engagement_column_map = (
        platform_mapping.comment_mapping.engagement_cols
        if is_comment
        else platform_mapping.content_mapping.engagement_cols
    )

    return SearchRecord(
        platform=result_row.get("platform"),
        source_table=result_row.get("source_table"),
        mysql_pk=result_row.get("mysql_pk"),
        title_or_content=result_row.get("title"),
        published_at=_to_datetime(result_row.get("published_at")),
        source_keyword=result_row.get("source_keyword"),
        engagement=_extract_engagement(result_row, engagement_column_map),
        hotness_score=result_row.get("hotness_score"),
    )


def _to_datetime(value: Any) -> datetime:
    """把一个未知的时间戳不管是秒级还是毫秒级，转换成datetime 时间对象"""
    timestamp = float(value)
    return datetime.fromtimestamp(
        timestamp / 1000 if timestamp > 1_000_000_000_000 else timestamp
    )


def _extract_engagement(result_row: dict[str, Any],
                        metric_to_column_map: Mapping[str, str]) -> dict[str, int]:
    extracted_metrics = {}
    for metric_name in metric_to_column_map:
        aliased_column_name = f"eng_{metric_name}"
        metric_value = result_row.get(aliased_column_name)
        if metric_value is not None:
            extracted_metrics[metric_name] = metric_value
    return extracted_metrics
