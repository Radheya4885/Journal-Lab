from typing import Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    artifacts_ready: bool
    paper_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)

class DomainListResponse(BaseModel):
    domains: list[str]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Research query or arXiv ID/URL")
    domain: str = Field(default="Auto", description="Domain name, e.g. 'Machine Learning' or 'Auto'")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")

class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: Optional[str] = None
    categories: Optional[str] = None
    primary_category: Optional[str] = None
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    domain_tags: list[str] = Field(default_factory=list)
    rank: Optional[int] = None
    rrf_score: Optional[float] = None
    tfidf_score: Optional[float] = None
    semantic_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    query: str
    domain: str
    is_arxiv_id_query: bool = False
    seed_paper: Optional[PaperMetadata] = None
    total_results: int
    results: list[PaperMetadata]

class GapAnalysisRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    domain: str = Field(default="Auto")
    top_k: int = Field(default=15, ge=5, le=30)

class GapAnalysisResponse(BaseModel):
    topic: str
    domain: str
    report: str
    retrieved_papers_count: int
    cited_paper_ids: list[str] = Field(default_factory=list)
    unsupported_citation_ids: list[str] = Field(default_factory=list)
    disclaimer: str

class CounterEvidenceRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    candidate_gap: str = Field(..., min_length=1)
    domain: str = Field(default="Auto")
    papers_per_search: int = Field(default=10, ge=3, le=20)

class CounterEvidenceResponse(BaseModel):
    topic: str
    candidate_gap: str
    report: str
    unique_papers_examined: int
    unmatched_ids: list[str] = Field(default_factory=list)

class ExperimentPlanRequest(BaseModel):
    research_topic: str = Field(..., min_length=1)
    candidate_gap: str = Field(..., min_length=1)
    domain: str = Field(default="Auto")
    top_k: int = Field(default=10, ge=3, le=20)

class ExperimentPlanResponse(BaseModel):
    research_topic: str
    candidate_gap: str
    plan: str
    supporting_papers_count: int
    unmatched_ids: list[str] = Field(default_factory=list)

class FigureResponse(BaseModel):
    arxiv_id: str
    caption: str
    image_data_uri: Optional[str] = None
    fetch_mode: str
    paper_url: str
    explanation: Optional[str] = None
