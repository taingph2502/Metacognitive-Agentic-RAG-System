from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow_naive() -> datetime:
    """Match the current PostgreSQL schema, which stores naive timestamps."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RunLog(Base):
    __tablename__ = "run_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String)
    query_type: Mapped[str] = mapped_column(String)
    config_name: Mapped[str] = mapped_column(String)
    faithfulness: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float)
    filename: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="processing")  # processing, indexed, failed
    chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)


class RetrievalDiagnostic(Base):
    __tablename__ = "retrieval_diagnostics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String)
    query_type: Mapped[str] = mapped_column(String)
    config_name: Mapped[str] = mapped_column(String)
    query_coverage: Mapped[float] = mapped_column(Float, default=0.0)
    document_diversity: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_redundancy: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_recall_proxy: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)


class ProvenanceSnapshot(Base):
    __tablename__ = "provenance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String)
    query_type: Mapped[str] = mapped_column(String)
    config_name: Mapped[str] = mapped_column(String)
    diagnosis: Mapped[str] = mapped_column(String, default="satisfactory")
    faithfulness: Mapped[float] = mapped_column(Float, default=0.0)
    citation_precision: Mapped[float] = mapped_column(Float, default=0.0)
    hops: Mapped[int] = mapped_column(Integer, default=0)
    metacognitive_rounds: Mapped[int] = mapped_column(Integer, default=0)
    answer_preview: Mapped[str] = mapped_column(String)
    provenance_payload: Mapped[dict] = mapped_column(JSONB)
    retrieval_diagnostics: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
