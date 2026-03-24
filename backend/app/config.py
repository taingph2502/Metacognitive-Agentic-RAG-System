from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # LLM provider: "deepseek" or "gemini"
    llm_provider: str = "deepseek"

    # DeepSeek (OpenAI-compatible)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Gemini (kept for side-by-side comparison)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://ara:ara_secret@localhost:5432/ara_db"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "research_papers"

    # ElasticSearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "research_chunks"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Self-improvement thresholds
    faithfulness_threshold: float = 0.7
    completeness_threshold: float = 0.6
    citation_precision_threshold: float = 0.4
    lambda_cost: float = 0.3
    lambda_latency: float = 0.2

    # Constraints
    max_latency_seconds: float = 15.0

    # Retrieval intelligence / control
    default_rewrite_count: int = 4
    evidence_coverage_threshold: float = 0.45
    min_retrieval_diversity: float = 0.25
    evaluator_confidence_threshold: float = 0.4

    # Multi-metric reward weights
    reward_w1_faithfulness: float = 0.45
    reward_w2_citation_precision: float = 0.3
    reward_w3_answer_completeness: float = 0.2
    reward_w4_latency_penalty: float = 0.15

    # Metacognitive regulation (Paper §4, §6.5)
    max_metacognitive_rounds: int = 3
    metacognitive_convergence_threshold: float = 0.85

    # Phase 3 latency optimizations (default OFF for reproducibility)
    fast_path_threshold: float | None = None  # evaluator confidence for skipping diagnosis; None = disabled
    disable_parallel: bool = False  # disable parallel post-write verification


settings = Settings()
