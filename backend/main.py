import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from backend.config import BASE_DIR, settings
from backend.figure_service import figure_service
from backend.rag_service import rag_service
from backend.retrieval import generate_bibtex, retriever
from backend.schemas import (
    CounterEvidenceRequest,
    CounterEvidenceResponse,
    DomainListResponse,
    ExperimentPlanRequest,
    ExperimentPlanResponse,
    FigureResponse,
    GapAnalysisRequest,
    GapAnalysisResponse,
    HealthResponse,
    PaperMetadata,
    RecommendationResponse,
    SearchRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("journallab.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load retrieval indices on startup."""
    logger.info("Initializing %s v%s...", settings.PROJECT_NAME, settings.VERSION)
    try:
        retriever.load_artifacts()
        logger.info("Retriever loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load retriever artifacts at startup: %s. "
            "Search endpoints will return 503 until 'python backend/build_index.py' is executed.",
            exc,
        )
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)


app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    version=settings.VERSION,
    description="Production backend API for JournalLab: AI-Based Research Paper Recommendation Platform.",
    lifespan=lifespan,
)

# Enable CORS for local development and file:// origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/api/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Report readiness status of the server, dataset, and retrieval indices."""
    if not retriever.is_ready:
        try:
            retriever.load_artifacts()
            logger.info("Retriever artifacts successfully loaded on health probe.")
        except Exception as exc:
            logger.debug("Artifact reload attempt: %s", exc)

    is_ready = retriever.is_ready
    paper_count = len(retriever.papers) if retriever.papers is not None else 0

    return HealthResponse(
        status="ready" if is_ready else "initializing_or_artifacts_missing",
        artifacts_ready=is_ready,
        paper_count=paper_count,
        details={
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "rrf_k": settings.RRF_K,
            "groq_configured": bool(settings.GROQ_API_KEY),
            "data_exists": settings.DATA_PATH.exists(),
            "models_dir": str(settings.MODELS_DIR),
        },
    )


@app.get("/api/domains", response_model=DomainListResponse, tags=["Search"])
def get_supported_domains() -> DomainListResponse:
    """Return all supported research domain categories."""
    return DomainListResponse(domains=retriever.get_supported_domains())


@app.post("/api/recommend", response_model=RecommendationResponse, tags=["Search"])
def recommend_papers(req: SearchRequest) -> RecommendationResponse:
    """
    Search and recommend papers using hybrid TF-IDF + MiniLM retrieval and Reciprocal Rank Fusion.
    Accepts natural-language queries or arXiv IDs/URLs.
    """
    if not retriever.is_ready:
        try:
            retriever.load_artifacts()
        except Exception:
            pass

    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval artifacts are not loaded yet. Run 'python backend/build_index.py' to generate model indices.",
        )

    try:
        result = retriever.search(
            query=req.query,
            domain=req.domain,
            top_k=req.top_k,
            rrf_k=settings.RRF_K,
        )
        return RecommendationResponse(**result)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.exception("Error during paper recommendation: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search failed: {str(exc)}")


@app.get("/api/papers/{arxiv_id}", response_model=PaperMetadata, tags=["Papers"])
def get_paper_details(arxiv_id: str) -> PaperMetadata:
    """Fetch metadata for a specific paper by its arXiv ID."""
    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Catalog is not loaded yet.",
        )

    paper = retriever.get_paper(arxiv_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with arXiv ID '{arxiv_id}' was not found in the local catalog.",
        )
    return paper


@app.get("/api/papers/{arxiv_id}/bibtex", response_class=PlainTextResponse, tags=["Papers"])
def get_paper_bibtex(arxiv_id: str) -> PlainTextResponse:
    """Generate and return clean BibTeX citation format for an arXiv paper."""
    paper = retriever.get_paper(arxiv_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with arXiv ID '{arxiv_id}' was not found.",
        )
    return PlainTextResponse(generate_bibtex(paper), media_type="application/x-bibtex")


@app.post("/api/research-gaps/stream", tags=["AI Research"])
def stream_research_gaps(req: GapAnalysisRequest) -> StreamingResponse:
    """Stream grounded research gap analysis token-by-token."""
    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not ready.",
        )
    return StreamingResponse(
        rag_service.stream_find_research_gaps(topic=req.topic, domain=req.domain, top_k=req.top_k),
        media_type="text/plain",
    )


@app.post("/api/research-gaps", response_model=GapAnalysisResponse, tags=["AI Research"])
def analyze_research_gaps(req: GapAnalysisRequest) -> GapAnalysisResponse:
    """Perform grounded research gap analysis using top retrieved papers and Groq RAG."""
    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not ready.",
        )

    try:
        return rag_service.find_research_gaps(topic=req.topic, domain=req.domain, top_k=req.top_k)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.exception("Gap analysis error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/api/counter-evidence", response_model=CounterEvidenceResponse, tags=["AI Research"])
def test_counter_evidence(req: CounterEvidenceRequest) -> CounterEvidenceResponse:
    """Stress-test a proposed research gap against potential counter-evidence and overlapping prior art."""
    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not ready.",
        )

    try:
        return rag_service.stress_test_research_gap(
            research_topic=req.topic,
            candidate_gap=req.candidate_gap,
            domain=req.domain,
            papers_per_search=req.papers_per_search,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.exception("Counter-evidence test error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/api/plan-experiment/stream", tags=["AI Research"])
def stream_plan_experiment(req: ExperimentPlanRequest) -> StreamingResponse:
    """Stream 10-section experiment plan token-by-token."""
    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not ready.",
        )
    return StreamingResponse(
        rag_service.stream_plan_research_experiment(
            research_topic=req.research_topic,
            candidate_gap=req.candidate_gap,
            domain=req.domain,
            top_k=req.top_k,
        ),
        media_type="text/plain",
    )


@app.post("/api/plan-experiment", response_model=ExperimentPlanResponse, tags=["AI Research"])
def plan_experiment(req: ExperimentPlanRequest) -> ExperimentPlanResponse:
    """Generate a reproducible, 10-section experimental research plan for a candidate gap."""
    if not retriever.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not ready.",
        )

    try:
        return rag_service.plan_research_experiment(
            research_topic=req.research_topic,
            candidate_gap=req.candidate_gap,
            domain=req.domain,
            top_k=req.top_k,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.exception("Experiment planning error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.get("/api/papers/{arxiv_id}/figures", response_model=FigureResponse, tags=["Multimodal"])
def get_paper_figure(
    arxiv_id: str,
    title: Optional[str] = Query(default=None),
    abstract: Optional[str] = Query(default=None),
) -> FigureResponse:
    """Extract or retrieve the architecture/framework figure for an arXiv paper."""
    fig = figure_service.get_paper_figure(arxiv_id, title=title or "", abstract=abstract or "")
    if fig is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not extract a diagram or preview page for paper arXiv:{arxiv_id}.",
        )
    return fig


class FigureExplainRequest(BaseModel):
    title: str
    abstract: str

@app.post("/api/papers/{arxiv_id}/figures/explain", tags=["Multimodal"])
def explain_paper_figure(arxiv_id: str, req: FigureExplainRequest) -> Dict[str, str]:
    """Provide a multimodal vision explanation for the paper's extracted figure."""
    explanation = figure_service.explain_figure(arxiv_id, req.title, req.abstract)
    return {"arxiv_id": arxiv_id, "explanation": explanation}


# --- Frontend Direct Delivery Routes ---
@app.api_route("/", methods=["GET", "HEAD"], response_class=FileResponse, include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the JournalLab frontend application directly from FastAPI."""
    return FileResponse(settings.BASE_DIR / "index.html")


@app.get("/styles.css", response_class=FileResponse, include_in_schema=False)
def serve_styles() -> FileResponse:
    """Serve frontend CSS styling."""
    return FileResponse(settings.BASE_DIR / "styles.css")


@app.get("/app.js", response_class=FileResponse, include_in_schema=False)
def serve_app_js() -> FileResponse:
    """Serve frontend client interaction logic."""
    return FileResponse(settings.BASE_DIR / "app.js")


@app.get("/gradient-wave-bg.js", response_class=FileResponse, include_in_schema=False)
def serve_gradient_wave_js() -> FileResponse:
    """Serve WebGL gradient wave background animation script."""
    return FileResponse(settings.BASE_DIR / "gradient-wave-bg.js")


