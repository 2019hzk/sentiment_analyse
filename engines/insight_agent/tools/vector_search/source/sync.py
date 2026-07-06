import argparse
import asyncio

from loguru import logger

from engines.contracts import config
from engines.insight_agent.tools.vector_search.repository import VectorRepository
from engines.insight_agent.tools.vector_search.source.reader import SourceDocumentReader


async def sync_data(drop_existing: bool = False, batch_size: int | None = None) -> int:
    milvus_repository = VectorRepository()
    source_reader = SourceDocumentReader()
    batch_size = batch_size or config.get_settings().INSIGHT_SYNC_BATCH_SIZE

    milvus_repository.ensure_collection(drop_existing=drop_existing)

    total = 0
    async for documents in source_reader.iter_documents(batch_size=batch_size):
        count = milvus_repository.upsert_documents(documents)
        total += count
        logger.info(f"同步Milvus 批次={count} 总计={total}")

    logger.info(f"Milvus 同步完成 总计={total}")
    return total



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同步MySQL数据到Milvus")
    asyncio.run(sync_data())
