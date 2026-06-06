import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType


class QdrantStorage:
    def __init__(self, collection="docs", dim=384):
        qdrant_url = os.getenv("QDRANT_URL", os.getenv("QDRANT_HOST_URL", "http://localhost:6333"))
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

        if "io:6333" in qdrant_url and qdrant_url.startswith("https://"):
            qdrant_url = qdrant_url.replace("https://", "")
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as index_err:
            # If it already exists from an earlier handshake, bypass gracefully
            print(f"Payload index initialization status: {str(index_err)}")

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5, source_id: str = None):
        # Build an isolated structural scope filter if a source file constraint is present
        query_filter = None
        if source_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_id)
                    )
                ]
            )

        # Executes query point matching passing the filter argument payload
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,  # Restricts lookup entirely to this single file ID
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}