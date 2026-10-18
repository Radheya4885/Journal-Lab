import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the repository (parent of backend)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "JournalLab"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    BASE_DIR: Path = BASE_DIR

    # Data and artifact paths
    DATA_PATH: Path = BASE_DIR / "data" / "arxiv_papers_enriched_semantic_scholar_v1.csv"
    MODELS_DIR: Path = BASE_DIR / "models"
    FIGURES_DIR: Path = BASE_DIR / "data" / "paper_figures"

    # Artifact filenames in models/
    TFIDF_VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.joblib"
    TFIDF_MATRIX_PATH: Path = MODELS_DIR / "tfidf_matrix.npz"
    SEMANTIC_EMBEDDINGS_PATH: Path = MODELS_DIR / "semantic_embeddings.npz"
    CATALOG_PATH: Path = MODELS_DIR / "paper_catalog.joblib"
    MANIFEST_PATH: Path = MODELS_DIR / "manifest.json"

    # ML & Retrieval hyperparameters
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RRF_K: int = 60
    TFIDF_MAX_FEATURES: int = 150_000

    # Optional Groq LLM integration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_RAG_MODEL: str = os.getenv("GROQ_RAG_MODEL", "qwen/qwen3.8-27b")
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
