import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.retrieval import extract_domains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("journallab.build_index")


def build_artifacts(force: bool = False) -> None:
    """Build and serialize TF-IDF indices, semantic embeddings, and paper catalog."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    required_artifacts = [
        settings.TFIDF_VECTORIZER_PATH,
        settings.TFIDF_MATRIX_PATH,
        settings.SEMANTIC_EMBEDDINGS_PATH,
        settings.CATALOG_PATH,
        settings.MANIFEST_PATH,
    ]

    all_exist = all(p.exists() and p.stat().st_size > 0 for p in required_artifacts)
    if all_exist and not force:
        logger.info("All required artifacts already exist in %s. Use --force to rebuild.", settings.MODELS_DIR)
        return

    csv_path = settings.DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Master dataset CSV not found at: {csv_path}")

    logger.info("Loading master dataset from %s...", csv_path)
    t0 = time.time()
    df = pd.read_csv(
        csv_path,
        dtype={"arxiv_id": str},
        low_memory=False
    )
    logger.info("Loaded %d rows in %.2f seconds.", len(df), time.time() - t0)

    # Validate essential columns
    for col in ["arxiv_id", "title", "abstract"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from dataset.")

    papers = df.reset_index(drop=True).copy()
    papers["arxiv_id"] = papers["arxiv_id"].astype(str).str.strip()
    papers["title"] = papers["title"].fillna("").astype(str).str.strip()
    papers["abstract"] = papers["abstract"].fillna("").astype(str).str.strip()

    # Assert uniqueness of arXiv IDs
    assert papers["arxiv_id"].is_unique, "Duplicate arXiv IDs found in dataset!"

    # Compute domain tags
    logger.info("Extracting research domain taxonomy tags...")
    papers["domain_tags"] = papers.apply(extract_domains, axis=1)

    # 1. Build Lexical (TF-IDF) Representation
    # Give title terms extra weight (title + " " + title + " " + abstract) as in original pipeline
    papers["search_text"] = papers["title"] + " " + papers["title"] + " " + papers["abstract"]

    logger.info("Fitting TF-IDF Vectorizer (ngram_range=(1,2), max_features=%d)...", settings.TFIDF_MAX_FEATURES)
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=settings.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    t_tfidf = time.time()
    tfidf_matrix = tfidf_vectorizer.fit_transform(papers["search_text"])
    logger.info("TF-IDF matrix built: shape %s in %.2f seconds.", tfidf_matrix.shape, time.time() - t_tfidf)

    # 2. Build Dense Semantic Embeddings
    logger.info("Loading SentenceTransformer model '%s'...", settings.EMBEDDING_MODEL_NAME)
    encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    texts_to_embed = (papers["title"] + ". " + papers["abstract"]).tolist()
    logger.info("Encoding %d papers using dense transformer...", len(texts_to_embed))
    t_enc = time.time()
    semantic_embeddings = encoder.encode(
        texts_to_embed,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    logger.info("Dense embeddings computed: shape %s in %.2f seconds.", semantic_embeddings.shape, time.time() - t_enc)

    # 3. Save Artifacts
    logger.info("Saving artifacts to %s...", settings.MODELS_DIR)
    joblib.dump(tfidf_vectorizer, settings.TFIDF_VECTORIZER_PATH)
    save_npz(settings.TFIDF_MATRIX_PATH, tfidf_matrix)

    paper_ids = papers["arxiv_id"].to_numpy().astype("U")
    np.savez_compressed(
        settings.SEMANTIC_EMBEDDINGS_PATH,
        embeddings=semantic_embeddings,
        arxiv_ids=paper_ids,
    )

    catalog_columns = [
        "arxiv_id", "title", "abstract", "authors", "categories", "primary_category",
        "publication_year", "doi", "url", "pdf_url", "s2_citation_count", "s2_venue",
        "domain_tags",
    ]
    catalog_cols_present = [c for c in catalog_columns if c in papers.columns]
    joblib.dump(papers[catalog_cols_present].reset_index(drop=True), settings.CATALOG_PATH)

    manifest = {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "paper_count": len(papers),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": int(semantic_embeddings.shape[1]),
        "tfidf_vocabulary_size": int(tfidf_matrix.shape[1]),
        "rrf_k": settings.RRF_K,
        "dataset_file": str(settings.DATA_PATH.name),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    settings.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Artifact generation completed successfully in %.2f total seconds.", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build JournalLab retrieval artifacts.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if artifacts exist.")
    args = parser.parse_args()
    build_artifacts(force=args.force)
