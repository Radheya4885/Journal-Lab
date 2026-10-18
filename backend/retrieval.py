import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.sparse import load_npz
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.schemas import PaperMetadata

logger = logging.getLogger("journallab.retrieval")

CATEGORY_TO_DOMAIN: Dict[str, str] = {
    "cs.LG": "Machine Learning",
    "stat.ML": "Machine Learning",
    "cs.CV": "Computer Vision",
    "eess.IV": "Computer Vision",
    "cs.CL": "Natural Language Processing",
    "cs.AI": "Artificial Intelligence",
    "cs.NE": "Artificial Intelligence",
    "cs.RO": "Robotics",
    "cs.CR": "Cybersecurity",
    "cs.IR": "Information Retrieval",
    "eess.AS": "Speech & Audio",
    "cs.SD": "Speech & Audio",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.HC": "Human-Computer Interaction",
    "cs.GR": "Graphics & Multimedia",
    "cs.MM": "Graphics & Multimedia",
    "eess.SP": "Signal Processing",
    "cs.DC": "Distributed Systems",
    "cs.CY": "Computers & Society",
}

ALL_DOMAINS = "All domains"
AUTO_DOMAIN = "Auto"
OTHER_DOMAIN = "Other / Interdisciplinary"


def normalize_arxiv_id(value: Any) -> str:
    """Normalize user input or dataset ID to canonical arXiv format (e.g. 2006.07909)."""
    text = str(value).strip().lower()
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", text)
    text = re.sub(r"^arxiv:\s*", "", text)
    text = re.sub(r"\.pdf$", "", text)
    return re.sub(r"v\d+$", "", text).strip()


def extract_domains(row: Any) -> List[str]:
    """Map primary_category and categories into human-friendly research domains."""
    if isinstance(row, dict):
        text = f"{row.get('primary_category', '')} {row.get('categories', '')}"
    else:
        p_cat = getattr(row, "primary_category", "") if hasattr(row, "primary_category") else row.get("primary_category", "")
        cats = getattr(row, "categories", "") if hasattr(row, "categories") else row.get("categories", "")
        text = f"{p_cat} {cats}"

    category_ids = set(re.findall(r"[a-z]+(?:-[a-z]+)?\.[A-Z]{2}", str(text)))
    domains = sorted({CATEGORY_TO_DOMAIN[c] for c in category_ids if c in CATEGORY_TO_DOMAIN})
    return domains or [OTHER_DOMAIN]


def fetch_arxiv_paper_live(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Fetch live paper metadata from official arXiv Atom API as resilient fallback."""
    norm_id = normalize_arxiv_id(arxiv_id)
    url = f"https://export.arxiv.org/api/query?id_list={norm_id}"
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "JournalLab/1.0 (academic search fallback)"})
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.content)
        atom_ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        entry = root.find("atom:entry", atom_ns)
        if entry is None:
            return None

        title_el = entry.find("atom:title", atom_ns)
        summary_el = entry.find("atom:summary", atom_ns)
        if title_el is None or summary_el is None:
            return None

        title = " ".join(title_el.text.split()) if title_el.text else ""
        if not title or "Error" in title:
            return None

        summary = " ".join(summary_el.text.split()) if summary_el.text else ""

        authors = [
            a.find("atom:name", atom_ns).text
            for a in entry.findall("atom:author", atom_ns)
            if a.find("atom:name", atom_ns) is not None and a.find("atom:name", atom_ns).text
        ]
        authors_str = ", ".join(authors) if authors else "Unknown Authors"

        published_el = entry.find("atom:published", atom_ns)
        pub_year = int(published_el.text[:4]) if published_el is not None and published_el.text else None

        primary_cat_el = entry.find("arxiv:primary_category", atom_ns)
        primary_cat = primary_cat_el.attrib.get("term", "") if primary_cat_el is not None else ""

        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", atom_ns)]
        cat_str = " ".join(categories)

        domain_tags = extract_domains({"primary_category": primary_cat, "categories": cat_str})

        doi_el = entry.find("arxiv:doi", atom_ns)
        doi = doi_el.text if doi_el is not None else None

        return {
            "arxiv_id": norm_id,
            "title": title,
            "abstract": summary,
            "authors": authors_str,
            "categories": cat_str,
            "primary_category": primary_cat,
            "publication_year": pub_year,
            "doi": doi,
            "url": f"https://arxiv.org/abs/{norm_id}",
            "pdf_url": f"https://arxiv.org/pdf/{norm_id}.pdf",
            "s2_citation_count": 0,
            "s2_venue": "arXiv",
            "domain_tags": domain_tags,
        }
    except Exception as exc:
        logger.warning("Failed to fetch live arXiv metadata for %s: %s", norm_id, exc)
        return None


def generate_bibtex(paper: Any) -> str:
    """Generate clean BibTeX citation string for a paper."""
    if hasattr(paper, "model_dump"):
        d = paper.model_dump()
    elif isinstance(paper, dict):
        d = paper
    else:
        d = dict(paper)

    arxiv_id = str(d.get("arxiv_id", "")).strip()
    clean_id = re.sub(r"[^a-zA-Z0-9]", "_", arxiv_id)
    title = str(d.get("title", "")).strip()
    authors = str(d.get("authors", "")).strip() or "Unknown Authors"
    if "," in authors and " and " not in authors:
        authors_bib = " and ".join([a.strip() for a in authors.split(",") if a.strip()])
    else:
        authors_bib = authors
    year = d.get("publication_year") or "Unknown"
    url = d.get("url") or f"https://arxiv.org/abs/{arxiv_id}"

    return (
        f"@article{{arxiv_{clean_id},\n"
        f"  title={{{{{title}}}}},\n"
        f"  author={{{authors_bib}}},\n"
        f"  journal={{arXiv preprint arXiv:{arxiv_id}}},\n"
        f"  year={{{year}}},\n"
        f"  url={{{url}}}\n"
        f"}}"
    )


class JournalLabRetriever:
    """Production hybrid retrieval engine (TF-IDF + MiniLM-L6-v2 + Reciprocal Rank Fusion)."""

    def __init__(self) -> None:
        self.papers: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.semantic_embeddings: Optional[np.ndarray] = None
        self.paper_ids: Optional[np.ndarray] = None
        self.id_to_index: Dict[str, int] = {}
        self.domain_to_indices: Dict[str, List[int]] = {}
        self.encoder: Optional[SentenceTransformer] = None
        self.is_ready: bool = False

    def load_artifacts(self) -> None:
        """Load precomputed offline indices into memory."""
        vec_path = settings.TFIDF_VECTORIZER_PATH
        mat_path = settings.TFIDF_MATRIX_PATH
        emb_path = settings.SEMANTIC_EMBEDDINGS_PATH
        cat_path = settings.CATALOG_PATH

        for path, name in [
            (vec_path, "TF-IDF vectorizer"),
            (mat_path, "TF-IDF matrix"),
            (emb_path, "Semantic embeddings"),
            (cat_path, "Paper catalog"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact '{name}' not found at {path}. "
                    "Run 'python backend/build_index.py' to generate artifacts."
                )

        logger.info("Loading paper catalog from %s...", cat_path)
        self.papers = joblib.load(cat_path)
        if not isinstance(self.papers, pd.DataFrame):
            self.papers = pd.DataFrame(self.papers)

        self.papers["arxiv_id"] = self.papers["arxiv_id"].astype(str)
        self.paper_ids = self.papers["arxiv_id"].to_numpy()

        logger.info("Loading TF-IDF vectorizer and matrix...")
        self.tfidf_vectorizer = joblib.load(vec_path)
        self.tfidf_matrix = load_npz(mat_path)

        logger.info("Loading dense semantic embeddings...")
        with np.load(emb_path, allow_pickle=True) as data:
            self.semantic_embeddings = data["embeddings"].astype(np.float32)

        # Integrity validations
        n_papers = len(self.papers)
        if self.tfidf_matrix.shape[0] != n_papers:
            raise ValueError(f"TF-IDF matrix rows ({self.tfidf_matrix.shape[0]}) != papers count ({n_papers})")
        if self.semantic_embeddings.shape[0] != n_papers:
            raise ValueError(f"Semantic embeddings ({self.semantic_embeddings.shape[0]}) != papers count ({n_papers})")

        # Build fast lookup tables
        self.id_to_index = {
            normalize_arxiv_id(pid): idx for idx, pid in enumerate(self.paper_ids)
        }

        # Build domain indices
        self.domain_to_indices = {}
        for idx, tags in enumerate(self.papers["domain_tags"]):
            for tag in tags:
                self.domain_to_indices.setdefault(tag, []).append(idx)

        logger.info("Initializing SentenceTransformer model: %s...", settings.EMBEDDING_MODEL_NAME)
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

        self.is_ready = True
        logger.info("JournalLab Retriever initialized successfully with %d papers.", n_papers)

    def get_supported_domains(self) -> List[str]:
        """Return list of distinct supported domains."""
        if not self.is_ready or self.papers is None:
            return [AUTO_DOMAIN, ALL_DOMAINS] + sorted(set(CATEGORY_TO_DOMAIN.values()))
        unique_domains = sorted({d for tags in self.papers["domain_tags"] for d in tags if d != OTHER_DOMAIN})
        return [AUTO_DOMAIN, ALL_DOMAINS] + unique_domains + [OTHER_DOMAIN]

    def get_paper(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Fetch a single paper by its arXiv ID, with live fallback to arXiv API."""
        if not self.is_ready or self.papers is None:
            return None
        norm_id = normalize_arxiv_id(arxiv_id)
        idx = self.id_to_index.get(norm_id)
        if idx is not None:
            return self._row_to_metadata(self.papers.iloc[idx])
        # Resilient fallback: fetch live paper metadata from arXiv Atom API
        live_data = fetch_arxiv_paper_live(norm_id)
        if live_data is not None:
            return self._row_to_metadata(live_data)
        return None

    def search(
        self,
        query: str,
        domain: str = AUTO_DOMAIN,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """
        Execute hybrid search using the existing JournalLab algorithm:
        1. Parse input (natural language topic vs. arXiv ID / URL)
        2. Resolve domain constraints
        3. Compute lexical (TF-IDF) & dense semantic cosine similarities
        4. Fuse ranks with Reciprocal Rank Fusion (k=60)
        5. Exclude seed paper if searching by ID
        """
        if not self.is_ready or self.papers is None:
            raise RuntimeError("Retriever artifacts are not loaded. Server is not ready.")

        user_input = str(query).strip()
        if not user_input:
            raise ValueError("Query cannot be empty.")

        n = len(self.papers)
        source_paper_row = None
        is_arxiv_id = False

        # Check if query matches an arXiv ID pattern
        id_match = re.search(r"\d{4}\.\d{4,5}(?:v\d+)?", user_input)
        if id_match:
            candidate_id = normalize_arxiv_id(id_match.group(0))
            idx = self.id_to_index.get(candidate_id)
            if idx is not None:
                source_paper_row = self.papers.iloc[idx]
                is_arxiv_id = True
                search_query = f"{source_paper_row['title']}. {source_paper_row['abstract']}"
            else:
                # Attempt on-demand live fetch from arXiv API
                live_meta = fetch_arxiv_paper_live(candidate_id)
                if live_meta is not None:
                    source_paper_row = live_meta
                    is_arxiv_id = True
                    search_query = f"{live_meta['title']}. {live_meta['abstract']}"
                elif user_input.lower().startswith(("arxiv:", "http")) or user_input == id_match.group(0):
                    raise ValueError(f"Paper arXiv ID '{candidate_id}' could not be found locally or via arXiv API.")
                else:
                    search_query = user_input
        else:
            search_query = user_input

        # Domain candidate filtering
        if domain == AUTO_DOMAIN and source_paper_row is not None:
            source_domains = set(source_paper_row.get("domain_tags", []))
            eligible = np.array([
                i for i, tags in enumerate(self.papers["domain_tags"])
                if source_domains.intersection(tags)
            ], dtype=int)
            if len(eligible) == 0:
                eligible = np.arange(n)
        elif domain in (AUTO_DOMAIN, ALL_DOMAINS):
            eligible = np.arange(n)
        else:
            eligible = np.asarray(self.domain_to_indices.get(domain, []), dtype=int)

        if len(eligible) == 0:
            raise ValueError(f"No papers available in selected domain: '{domain}'.")

        # 1. Lexical retrieval (TF-IDF cosine similarity)
        q_tfidf = self.tfidf_vectorizer.transform([search_query])
        lexical_scores = (q_tfidf @ self.tfidf_matrix.T).toarray().ravel()

        # 2. Semantic retrieval (SentenceTransformer dot product)
        q_embedding = self.encoder.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        semantic_scores = self.semantic_embeddings @ q_embedding

        # 3. Reciprocal Rank Fusion within eligible pool
        lex_order = eligible[np.argsort(-lexical_scores[eligible], kind="stable")]
        sem_order = eligible[np.argsort(-semantic_scores[eligible], kind="stable")]

        lex_rank = np.zeros(n, dtype=np.int32)
        sem_rank = np.zeros(n, dtype=np.int32)
        lex_rank[lex_order] = np.arange(1, len(eligible) + 1)
        sem_rank[sem_order] = np.arange(1, len(eligible) + 1)

        rrf_scores = np.full(n, -np.inf, dtype=np.float64)
        rrf_scores[eligible] = (
            1.0 / (rrf_k + lex_rank[eligible])
            + 1.0 / (rrf_k + sem_rank[eligible])
        )

        # Exclude source paper if querying by arXiv ID
        if source_paper_row is not None:
            source_idx = self.id_to_index.get(normalize_arxiv_id(source_paper_row["arxiv_id"]))
            if source_idx is not None:
                rrf_scores[source_idx] = -np.inf

        # Rank candidates
        ranked = eligible[np.argsort(-rrf_scores[eligible], kind="stable")]
        ranked = [i for i in ranked if np.isfinite(rrf_scores[i])][:top_k]

        results: List[PaperMetadata] = []
        for rank_num, idx in enumerate(ranked, start=1):
            row = self.papers.iloc[idx]
            paper_meta = self._row_to_metadata(
                row,
                rank=rank_num,
                rrf_score=round(float(rrf_scores[idx]), 6),
                tfidf_score=round(float(lexical_scores[idx]), 4),
                semantic_score=round(float(semantic_scores[idx]), 4),
            )
            results.append(paper_meta)

        seed_paper_meta = self._row_to_metadata(source_paper_row) if source_paper_row is not None else None

        return {
            "query": user_input,
            "domain": domain,
            "is_arxiv_id_query": is_arxiv_id,
            "seed_paper": seed_paper_meta,
            "total_results": len(results),
            "results": results,
        }

    def _row_to_metadata(
        self,
        row: Any,
        rank: Optional[int] = None,
        rrf_score: Optional[float] = None,
        tfidf_score: Optional[float] = None,
        semantic_score: Optional[float] = None,
    ) -> PaperMetadata:
        """Convert a catalog row to a PaperMetadata model."""
        def safe_str(val: Any) -> Optional[str]:
            if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "None"):
                return None
            return str(val).strip()

        def safe_int(val: Any) -> Optional[int]:
            try:
                if pd.notna(val) and val is not None and str(val).strip() not in ("", "nan"):
                    return int(float(val))
            except (ValueError, TypeError):
                pass
            return None

        arxiv_id = str(row.get("arxiv_id", "")).strip()
        tags = row.get("domain_tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(";") if t.strip()]
        elif not isinstance(tags, list):
            tags = []

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=str(row.get("title", "")).strip(),
            abstract=str(row.get("abstract", "")).strip(),
            authors=safe_str(row.get("authors")),
            categories=safe_str(row.get("categories")),
            primary_category=safe_str(row.get("primary_category")),
            publication_year=safe_int(row.get("publication_year")),
            doi=safe_str(row.get("doi")),
            url=safe_str(row.get("url")) or f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=safe_str(row.get("pdf_url")) or f"https://arxiv.org/pdf/{arxiv_id}",
            citation_count=safe_int(row.get("s2_citation_count")),
            venue=safe_str(row.get("s2_venue")),
            domain_tags=tags,
            rank=rank,
            rrf_score=rrf_score,
            tfidf_score=tfidf_score,
            semantic_score=semantic_score,
        )


# Global singleton instance
retriever = JournalLabRetriever()
