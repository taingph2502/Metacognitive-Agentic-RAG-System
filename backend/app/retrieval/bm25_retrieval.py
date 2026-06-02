"""
ElasticSearch-backed lexical (BM25) retrieval.
"""

from elasticsearch import Elasticsearch

from app.config import settings

_es_client: Elasticsearch | None = None

ES_INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "standard"},
            "source": {"type": "keyword"},
            "document_id": {"type": "integer"},
            "chunk_index": {"type": "integer"},
        }
    }
}


def get_es_client() -> Elasticsearch:
    """Return a singleton ElasticSearch client."""
    global _es_client
    if _es_client is None:
        _es_client = Elasticsearch(settings.elasticsearch_url)
    return _es_client


def ensure_es_index() -> None:
    """Create the ES index if it does not exist."""
    client = get_es_client()
    if not client.indices.exists(index=settings.elasticsearch_index):
        client.indices.create(
            index=settings.elasticsearch_index, body=ES_INDEX_SETTINGS
        )


def index_chunks(chunks: list[dict], document_id: int | None = None) -> None:
    """
    Index chunks into ElasticSearch incrementally.
    Called during document ingestion alongside Qdrant upsert.
    """
    from app.retrieval.dense import text_to_id

    ensure_es_index()
    client = get_es_client()

    operations: list[dict] = []
    for chunk in chunks:
        doc_id = str(text_to_id(chunk["text"], chunk["source"]))
        operations.append(
            {"index": {"_index": settings.elasticsearch_index, "_id": doc_id}}
        )
        operations.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk.get("chunk_index", 0),
                "document_id": document_id,
            }
        )

    if operations:
        client.bulk(operations=operations, refresh=True)


def bm25_search(
    query: str, top_k: int = 5, document_ids: list[int] | None = None
) -> list[dict]:
    """Search using ElasticSearch BM25 scoring."""
    ensure_es_index()
    client = get_es_client()

    if document_ids is not None and len(document_ids) == 0:
        return []

    es_query: dict = {"match": {"text": query}}

    if document_ids is not None:
        es_query = {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": [{"terms": {"document_id": document_ids}}],
            }
        }

    response = client.search(
        index=settings.elasticsearch_index,
        query=es_query,
        size=top_k,
    )

    return [
        {
            "id": int(hit["_id"]),
            "text": hit["_source"]["text"],
            "source": hit["_source"]["source"],
            "score": hit["_score"],
        }
        for hit in response["hits"]["hits"]
    ]


def wipe_es_index() -> None:
    """Delete and recreate the ES index."""
    client = get_es_client()
    if client.indices.exists(index=settings.elasticsearch_index):
        client.indices.delete(index=settings.elasticsearch_index)
    ensure_es_index()
