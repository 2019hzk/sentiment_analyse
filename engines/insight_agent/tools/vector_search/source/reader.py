from typing import Any, AsyncIterator

from engines.insight_agent.tools.connection import get_session_factory
from engines.insight_agent.tools.vector_search.search_results import VectorDocument
from engines.insight_agent.tools.vector_search.source.register import VECTOR_SYNC_REGISTRY, VectorSyncMapper


class SourceDocumentReader:

    def __init__(self) -> None:
        self.session_factory = get_session_factory()
        self.registry = VECTOR_SYNC_REGISTRY

    async def iter_documents(self, batch_size: int = 100) -> AsyncIterator[list[VectorDocument]]:
        for mapper in self.registry:
            offset = 0
            while True:
                rows = await self._fetch_table_rows(mapper, batch_size, offset)
                if not rows:
                    break
                raw_docs = [mapper.to_document(row) for row in rows]
                clean_docs = [doc for doc in raw_docs if doc.content.strip()]
                if clean_docs:
                    yield clean_docs

                offset += batch_size

    async def _fetch_table_rows(self, mapper: VectorSyncMapper, limit: int, offset: int) -> list[Any]:
        async with self.session_factory() as session:
            result = await session.execute(mapper.select_batch(limit, offset))
            return list(result.scalars().all())
