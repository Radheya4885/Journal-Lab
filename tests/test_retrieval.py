import pytest
from backend.retrieval import (
    CATEGORY_TO_DOMAIN,
    ALL_DOMAINS,
    AUTO_DOMAIN,
    OTHER_DOMAIN,
    extract_domains,
    normalize_arxiv_id,
)

def test_normalize_arxiv_id():
    """Verify various arXiv ID and URL inputs normalize to canonical ID string."""
    assert normalize_arxiv_id("2006.07909") == "2006.07909"
    assert normalize_arxiv_id("2006.07909v2") == "2006.07909"
    assert normalize_arxiv_id("arXiv:2006.07909v1") == "2006.07909"
    assert normalize_arxiv_id("https://arxiv.org/abs/2006.07909v2") == "2006.07909"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2006.07909.pdf") == "2006.07909"
    assert normalize_arxiv_id(" 2002.01464 ") == "2002.01464"

def test_extract_domains():
    """Verify arXiv category strings map correctly to high-level research domains."""
    row_ml = {"primary_category": "cs.LG", "categories": "cs.LG stat.ML"}
    domains_ml = extract_domains(row_ml)
    assert "Machine Learning" in domains_ml

    row_nlp = {"primary_category": "cs.CL", "categories": "cs.CL cs.AI"}
    domains_nlp = extract_domains(row_nlp)
    assert "Natural Language Processing" in domains_nlp
    assert "Artificial Intelligence" in domains_nlp

    row_unknown = {"primary_category": "quant-ph", "categories": "quant-ph"}
    domains_unknown = extract_domains(row_unknown)
    assert domains_unknown == [OTHER_DOMAIN]

def test_taxonomy_coverage():
    """Verify core AI domains are represented in category mapping."""
    domains = set(CATEGORY_TO_DOMAIN.values())
    expected = {
        "Machine Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Artificial Intelligence",
        "Robotics",
        "Information Retrieval",
        "Cybersecurity",
        "Databases",
    }
    assert expected.issubset(domains)
