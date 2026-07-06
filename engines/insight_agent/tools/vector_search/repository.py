from __future__ import annotations

from typing import Any

from loguru import logger
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus import MilvusClient
from engines.contracts import config
from engines.insight_agent.tools.vector_search.embedder import BgeM3Embedder, VectorEmbedding
from engines.insight_agent.tools.vector_search.schemas import (
    MILVUS_OUTPUT_FIELDS,
    build_collection_schema,
    build_index_params,
)
from engines.insight_agent.tools.vector_search.search_results import SearchHit, VectorDocument


class VectorRepository:

    def __init__(self) -> None:
        settings = config.get_settings()
        self.collection_name = settings.MILVUS_INSIGHT_COLLECTION
        self.embedder = BgeM3Embedder(
            model_name=settings.INSIGHT_EMBEDDING_MODEL,
            device=settings.INSIGHT_EMBEDDING_DEVICE,
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            settings = config.get_settings()
            kwargs: dict[str, Any] = {
                "uri": settings.MILVUS_URI,
                "db_name": settings.MILVUS_DB_NAME
            }
            self._client = MilvusClient(**kwargs)
        return self._client

    def ensure_collection(self, drop_existing: bool = False) -> None:
        if drop_existing and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        if self.client.has_collection(self.collection_name):
            return

        schema = build_collection_schema(self.client, config.get_settings().INSIGHT_DENSE_DIM)
        index_params = build_index_params(self.client)

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"创建Milvus集合 {self.collection_name}")

    def upsert_documents(self, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0

        self.ensure_collection()
        embeddings = self.embedder.encode_documents([doc.content for doc in documents])

        entities = [
            doc.to_milvus_record(
                dense_vector=embedding.dense_vector,
                sparse_vector=embedding.sparse_vector,
            )
            for doc, embedding in zip(documents, embeddings)
            if embedding.dense_vector and embedding.sparse_vector
        ]
        if not entities:
            return 0

        self.client.upsert(collection_name=self.collection_name, data=entities)
        return len(entities)

    def search(self, query: str, limit: int, filter_expr: str | None = None) -> list[SearchHit]:
        self.ensure_collection()

        query_embedding = self.embedder.encode_query(query)
        if not query_embedding or not query_embedding.dense_vector or not query_embedding.sparse_vector:
            return []

        return self._hybrid_search(query_embedding, limit, filter_expr)

    def _hybrid_search(
            self,
            query_embedding: VectorEmbedding,
            limit: int,
            filter_expr: str | None = None,
    ) -> list[SearchHit]:
        request_filter = {"expr": filter_expr} if filter_expr else {}
        dense_request = AnnSearchRequest(
            data=[query_embedding.dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE"},
            limit=limit,
            **request_filter,
        )
        sparse_request = AnnSearchRequest(
            data=[query_embedding.sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=limit,
            **request_filter,
        )
        result = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=MILVUS_OUTPUT_FIELDS,
        )

        return self._map_hits(result, "semantic_recall")

    @staticmethod
    def _map_hits(raw_result: Any, channel: str) -> list[SearchHit]:
        if not raw_result:
            return []
        first_query_hits = raw_result[0] if isinstance(raw_result, list) else raw_result
        hits: list[SearchHit] = []
        for hit_dict in first_query_hits:
            entity_data = hit_dict.get("entity") or hit_dict
            raw_doc_id = hit_dict.get("doc_id") or entity_data.get("doc_id")
            if not raw_doc_id:
                continue
            hits.append(SearchHit(
                doc_id=str(raw_doc_id),
                score=float(hit_dict.get("distance") or 0.0),
                channel=channel,
                entity=dict(entity_data),
            ))
        return hits




if __name__ == '__main__':

    rep= VectorRepository()


    response=rep.search(query="高考难吗?",limit=10)

    for hit in response:
        print(hit)

