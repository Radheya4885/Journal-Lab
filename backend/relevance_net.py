import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("journallab.relevance_net")

FEATURE_NAMES = [
    "semantic_cosine_sim",
    "semantic_dot_product",
    "embedding_l2_dist",
    "title_jaccard_overlap",
    "abstract_jaccard_overlap",
    "exact_title_match",
    "log_citation_count",
    "recency_score",
    "query_title_len_ratio",
    "domain_alignment",
]


class RelevanceFFNN(nn.Module):
    """
    Feedforward Neural Network for Research Paper Relevance Prediction.
    
    Architecture:
        Input(10) -> Linear(64) -> BatchNorm1d(64) -> GELU -> Dropout(0.2)
                  -> Linear(32) -> BatchNorm1d(32) -> GELU -> Dropout(0.1)
                  -> Linear(16) -> GELU
                  -> Linear(1) [Logit]
    """

    def __init__(self, in_features: int = 10, hidden_dims: Tuple[int, ...] = (64, 32, 16), dropout_rate: float = 0.2):
        super().__init__()
        self.in_features = in_features

        # Hidden layer 1
        self.fc1 = nn.Linear(in_features, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Hidden layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate / 2.0)

        # Hidden layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.act3 = nn.GELU()

        # Output layer (unnormalized logit for BCEWithLogitsLoss numerical stability)
        self.out = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw relevance logit."""
        h = self.drop1(self.act1(self.bn1(self.fc1(x))))
        h = self.drop2(self.act2(self.bn2(self.fc2(h))))
        h = self.act3(self.fc3(h))
        return self.out(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated relevance probability in [0, 1] using sigmoid activation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def extract_pair_features(
    query_text: str,
    paper_title: str,
    paper_abstract: str,
    publication_year: Optional[int] = None,
    citation_count: Optional[float] = None,
    query_domain: Optional[str] = None,
    paper_domain: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    paper_embedding: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract a 10-dimensional interaction feature vector for a query-paper pair.
    """
    q_words = set(query_text.lower().split())
    t_words = set(paper_title.lower().split())
    a_words = set(paper_abstract.lower().split())

    # 1. Semantic embedding interactions (if embeddings are supplied)
    if query_embedding is not None and paper_embedding is not None:
        q_norm = np.linalg.norm(query_embedding)
        p_norm = np.linalg.norm(paper_embedding)
        if q_norm > 1e-8 and p_norm > 1e-8:
            q_unit = query_embedding / q_norm
            p_unit = paper_embedding / p_norm
            cos_sim = float(np.dot(q_unit, p_unit))
            dot_prod = float(np.dot(query_embedding, paper_embedding))
            l2_dist = float(np.linalg.norm(q_unit - p_unit))
        else:
            cos_sim, dot_prod, l2_dist = 0.0, 0.0, 1.414
    else:
        # Fallback lexical proxy if pre-computed embeddings unavailable
        overlap = len(q_words.intersection(t_words | a_words))
        cos_sim = min(1.0, overlap / max(1, len(q_words)))
        dot_prod = cos_sim
        l2_dist = math.sqrt(max(0.0, 2.0 - 2.0 * cos_sim))

    # 2. Lexical Jaccard overlaps
    t_overlap = len(q_words.intersection(t_words)) / max(1, len(q_words.union(t_words)))
    a_overlap = len(q_words.intersection(a_words)) / max(1, len(q_words.union(a_words)))

    # 3. Exact phrase match
    clean_q = " ".join(query_text.lower().split())
    clean_t = " ".join(paper_title.lower().split())
    exact_title_match = 1.0 if clean_q in clean_t else 0.0

    # 4. Citations & Recency (handling NULLs & scaling gracefully)
    cites = float(citation_count) if citation_count is not None and not math.isnan(citation_count) else 0.0
    log_cites = math.log1p(max(0.0, cites))

    yr = publication_year if publication_year is not None and not math.isnan(publication_year) else 2022
    recency = (float(yr) - 2000.0) / 25.0

    # 5. Length ratio
    len_ratio = len(query_text.split()) / max(1, len(paper_title.split()))

    # 6. Domain alignment
    if query_domain and paper_domain:
        domain_match = 1.0 if query_domain.lower() in paper_domain.lower() or paper_domain.lower() in query_domain.lower() else 0.0
    else:
        domain_match = 0.5  # Neutral prior

    return np.array([
        cos_sim,
        dot_prod,
        l2_dist,
        t_overlap,
        a_overlap,
        exact_title_match,
        log_cites,
        recency,
        len_ratio,
        domain_match,
    ], dtype=np.float32)


class RelevancePredictor:
    """Production inference wrapper for the trained Relevance Feedforward Neural Network."""

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        import joblib
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = model_path or (base_dir / "models" / "relevance_net.pt")
        self.scaler_path = scaler_path or (base_dir / "models" / "relevance_scaler.joblib")
        
        self.model = RelevanceFFNN(in_features=len(FEATURE_NAMES))
        self.scaler = None
        self.is_ready = False
        self._load()

    def _load(self) -> None:
        import joblib
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                state_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.scaler = joblib.load(self.scaler_path)
                self.is_ready = True
                logger.info("Loaded trained RelevanceFFNN and scaler successfully.")
            except Exception as e:
                logger.warning("Failed loading RelevanceFFNN weights: %s", e)

    def predict(
        self,
        query_text: str,
        paper_title: str,
        paper_abstract: str,
        publication_year: Optional[int] = None,
        citation_count: Optional[float] = None,
        query_domain: Optional[str] = None,
        paper_domain: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        paper_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Predict continuous relevance probability in [0.0, 1.0]."""
        raw_feat = extract_pair_features(
            query_text=query_text,
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            publication_year=publication_year,
            citation_count=citation_count,
            query_domain=query_domain,
            paper_domain=paper_domain,
            query_embedding=query_embedding,
            paper_embedding=paper_embedding,
        ).reshape(1, -1)

        if self.scaler is not None:
            norm_feat = self.scaler.transform(raw_feat)
        else:
            norm_feat = raw_feat

        x_tensor = torch.tensor(norm_feat, dtype=torch.float32)
        with torch.no_grad():
            prob = float(self.model.predict_proba(x_tensor).item())
        return round(prob, 4)
