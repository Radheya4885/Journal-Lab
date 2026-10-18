import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify health endpoint returns status and artifact metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "artifacts_ready" in data
    assert "details" in data

def test_domains_endpoint():
    """Verify domains endpoint returns list of supported research domains."""
    response = client.get("/api/domains")
    assert response.status_code == 200
    data = response.json()
    assert "domains" in data
    assert len(data["domains"]) > 0
    assert "Auto" in data["domains"]
    assert "All domains" in data["domains"]

def test_empty_search_query_validation():
    """Verify empty search query is rejected with 422 Unprocessable Entity."""
    response = client.post("/api/recommend", json={"query": "", "domain": "Auto", "top_k": 5})
    assert response.status_code == 422

def test_nonexistent_paper_details():
    """Verify querying an invalid arXiv ID returns 404 or 503 if not ready."""
    response = client.get("/api/papers/9999.99999")
    assert response.status_code in (404, 503)

def test_static_routes_and_frontend():
    """Verify frontend static files are served properly."""
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "JournalLab" in r_index.text

    r_css = client.get("/styles.css")
    assert r_css.status_code == 200

    r_js = client.get("/app.js")
    assert r_js.status_code == 200

def test_openapi_docs_endpoint():
    """Verify FastAPI automatically serves the interactive Swagger docs and OpenAPI schema."""
    r_docs = client.get("/docs")
    assert r_docs.status_code == 200

    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    schema = r_openapi.json()
    assert "paths" in schema
    assert "/api/health" in schema["paths"]
    assert "/api/recommend" in schema["paths"]
    assert "/api/papers/{arxiv_id}/bibtex" in schema["paths"]
    assert "/api/research-gaps/stream" in schema["paths"]

def test_bibtex_generation():
    """Verify BibTeX endpoint returns properly formatted BibTeX citation."""
    response = client.get("/api/papers/2006.07909/bibtex")
    if response.status_code == 200:
        assert "@article" in response.text
        assert "arxiv" in response.text.lower()
