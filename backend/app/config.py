from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_model_strong: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "research_papers"

    # ElasticSearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "research_chunks"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Meta-Agent-RAG defaults
    retrieval_top_k: int = 5
    monitor_similarity_threshold: float = 0.4

    # Metacognitive regulation (Paper §4, §6.5)
    max_metacognitive_rounds: int = 5
    metacognitive_convergence_threshold: float = 0.85


settings = Settings()
