import pytest
from backend.retrieval import retriever, normalize_arxiv_id

@pytest.fixture(scope="module", autouse=True)
def ensure_retriever_loaded():
    """Ensure retriever artifacts are loaded for tests."""
    if not retriever.is_ready:
        retriever.load_artifacts()

def test_real_hybrid_retrieval_ranking():
    """Verify hybrid search returns top-10 ranked papers with strictly decreasing RRF scores."""
    query = "transformer models for natural language processing and text understanding"
    res = retriever.search(query=query, domain="All domains", top_k=10)

    assert res["total_results"] == 10
    assert len(res["results"]) == 10
    assert not res["is_arxiv_id_query"]
    assert res["seed_paper"] is None

    # Check score monotonically decreasing
    scores = [p.rrf_score for p in res["results"]]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"RRF scores not sorted: {scores[i]} < {scores[i+1]}"

    # Check essential fields are populated
    for paper in res["results"]:
        assert paper.arxiv_id
        assert paper.title
        assert paper.abstract
        assert paper.rrf_score is not None
        assert paper.tfidf_score is not None
        assert paper.semantic_score is not None
        assert paper.rank is not None

def test_domain_filtering():
    """Verify that domain filtering restricts results to the chosen domain."""
    query = "convolutional neural network visual recognition object detection"
    res = retriever.search(query=query, domain="Computer Vision", top_k=5)

    assert res["total_results"] > 0
    for paper in res["results"]:
        assert "Computer Vision" in paper.domain_tags, f"Paper {paper.arxiv_id} domain tags {paper.domain_tags} missing 'Computer Vision'"

def test_arxiv_id_seed_resolution_and_exclusion():
    """Verify searching by arXiv ID uses seed paper context and excludes seed paper from results."""
    seed_id = "2006.07909"
    res = retriever.search(query=seed_id, domain="Auto", top_k=5)

    assert res["is_arxiv_id_query"]
    assert res["seed_paper"] is not None
    assert normalize_arxiv_id(res["seed_paper"].arxiv_id) == seed_id

    # Seed paper MUST NOT appear in recommendations
    recommended_ids = [normalize_arxiv_id(p.arxiv_id) for p in res["results"]]
    assert seed_id not in recommended_ids, f"Seed paper {seed_id} was improperly included in recommendations"

def test_arxiv_url_resolution():
    """Verify searching by full arXiv URL resolves correctly."""
    url = "https://arxiv.org/abs/2006.07909v2"
    res = retriever.search(query=url, domain="Auto", top_k=5)

    assert res["is_arxiv_id_query"]
    assert res["seed_paper"] is not None
    assert normalize_arxiv_id(res["seed_paper"].arxiv_id) == "2006.07909"
