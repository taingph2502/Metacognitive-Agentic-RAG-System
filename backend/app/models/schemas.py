from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    query: str


class CitedChunk(BaseModel):
    index: int
    text: str
    source: str


class EvidenceGraphLinkRead(BaseModel):
    claim_index: int
    claim: str
    supporting_docs: list[int] = Field(default_factory=list)
    confidence: float = 0.0


class EvidenceGraphRead(BaseModel):
    total_claims: int = 0
    supported_claims: int = 0
    coverage_ratio: float = 0.0
    links: list[EvidenceGraphLinkRead] = Field(default_factory=list)


class RetrievedDocumentRead(BaseModel):
    rank: int
    source: str
    text: str
    score: float = 0.0
    cited: bool = False
    supporting_claims: list[int] = Field(default_factory=list)


class QueryProvenance(BaseModel):
    query_variants: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    followup_query: str | None = None
    retrieved_documents: list[RetrievedDocumentRead] = Field(default_factory=list)
    evidence_graph: EvidenceGraphRead = Field(default_factory=EvidenceGraphRead)


class RunMetrics(BaseModel):
    faithfulness: float
    citation_precision: float = 0.0
    unsupported_claim_rate: float = 1.0
    answer_completeness: float = 0.0
    cost: float
    latency: float
    utility: float
    config: str
    query_type: str
    hops: int
    retrieval_diagnostics: dict[str, float] = Field(default_factory=dict)
    # Metacognitive diagnosis (Paper §4)
    diagnosis: str = "satisfactory"
    diagnosis_reasoning: str = ""
    error_types: list[str] = Field(default_factory=list)
    metacognitive_rounds: int = 0


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitedChunk]
    metrics: RunMetrics
    provenance: QueryProvenance = Field(default_factory=QueryProvenance)
    abstained: bool = False


class DocumentUploadResponse(BaseModel):
    message: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    database: bool


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    source: str
    status: str
    chunks_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
