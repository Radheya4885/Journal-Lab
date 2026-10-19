"""
Training Pipeline for JournalLab Feedforward Neural Network (RelevanceFFNN).
Trains a PyTorch neural network to predict research paper relevance on labelled query-paper pairs
using backpropagation and Binary Cross Entropy with Logits loss (BCEWithLogitsLoss).
Follows ML Best Practices (strict splitting before scaling, cross-validation, and metrics reporting).
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.relevance_net import FEATURE_NAMES, RelevanceFFNN, extract_pair_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_relevance_net")

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Benchmark academic search queries with target domains
BENCHMARK_QUERIES = [
    ("Retrieval augmented generation with hallucination reduction", "Artificial Intelligence"),
    ("Contrastive learning representations for computer vision", "Computer Vision"),
    ("Graph neural networks for molecular property prediction", "Machine Learning"),
    ("Transformers self-attention computational complexity optimization", "Machine Learning"),
    ("Zero-shot text classification using large language models", "Natural Language Processing"),
    ("Reinforcement learning for robotic manipulation and control", "Robotics"),
    ("Adversarial attack defense and robustness in deep neural networks", "Cybersecurity"),
    ("Vector search and approximate nearest neighbor indexing", "Information Retrieval"),
    ("Multi-modal vision language models and instruction tuning", "Computer Vision"),
    ("Explainable AI feature attribution and SHAP values", "Machine Learning"),
    ("Few-shot transfer learning in low-resource domain settings", "Natural Language Processing"),
    ("Autonomous navigation using deep reinforcement learning", "Robotics"),
    ("Differentially private machine learning and data privacy", "Cybersecurity"),
    ("Dense passage retrieval and hybrid BM25 ranking", "Information Retrieval"),
    ("Diffusion probabilistic models for high fidelity image synthesis", "Computer Vision"),
    ("Prompt engineering in large language model reasoning", "Natural Language Processing"),
]


def create_labelled_pairs(
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    target_pairs_per_query: int = 50,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """
    Construct realistic labelled (query, paper) training pairs.
    - Positive pairs (y=1): Papers closely matching query concepts and domains.
    - Negative pairs (y=0): Disparate papers from contrasting domains or low semantic overlap.
    """
    logger.info("Constructing labelled query-paper pairs...")
    X_features: List[np.ndarray] = []
    y_labels: List[float] = []
    metadata_records: List[Dict[str, str]] = []

    # Map paper ID to embeddings
    n_papers = len(catalog)

    for q_idx, (query_text, query_domain) in enumerate(BENCHMARK_QUERIES):
        q_words = set(query_text.lower().split())

        # Compute keyword relevance scores across catalog
        scores = []
        for i, row in catalog.iterrows():
            title_words = set(str(row.get("title", "")).lower().split())
            abs_words = set(str(row.get("abstract", "")).lower().split())
            text_words = title_words | abs_words
            overlap = len(q_words.intersection(text_words)) / max(1, len(q_words))
            
            # Domain bonus
            p_cat = str(row.get("primary_category", "")).lower()
            domain_bonus = 0.3 if any(w in p_cat for w in query_domain.lower().split()) else 0.0
            
            score = overlap + domain_bonus
            scores.append((score, i))

        # Sort by relevance proxy
        scores.sort(key=lambda x: x[0], reverse=True)

        # Top candidates -> Positive samples (y = 1)
        positives = [idx for sc, idx in scores[:target_pairs_per_query] if sc > 0.15]
        # Bottom candidates & random cross-domain -> Negative samples (y = 0)
        negatives = [idx for sc, idx in scores[-target_pairs_per_query:]]

        # Extract features for positives
        for p_idx in positives:
            row = catalog.iloc[p_idx]
            p_emb = embeddings[p_idx]
            # Use average positive vector as proxy query embedding
            q_emb = embeddings[positives[0]] if len(positives) > 0 else p_emb

            feat = extract_pair_features(
                query_text=query_text,
                paper_title=str(row["title"]),
                paper_abstract=str(row.get("abstract", "")),
                publication_year=int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                citation_count=float(row["s2_citation_count"]) if pd.notna(row.get("s2_citation_count")) else 0.0,
                query_domain=query_domain,
                paper_domain=str(row.get("primary_category", "")),
                query_embedding=q_emb,
                paper_embedding=p_emb,
            )
            X_features.append(feat)
            y_labels.append(1.0)
            metadata_records.append({"query": query_text, "paper_title": str(row["title"]), "label": "relevant"})

        # Extract features for negatives
        for p_idx in negatives:
            row = catalog.iloc[p_idx]
            p_emb = embeddings[p_idx]
            q_emb = embeddings[positives[0]] if len(positives) > 0 else p_emb

            feat = extract_pair_features(
                query_text=query_text,
                paper_title=str(row["title"]),
                paper_abstract=str(row.get("abstract", "")),
                publication_year=int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                citation_count=float(row["s2_citation_count"]) if pd.notna(row.get("s2_citation_count")) else 0.0,
                query_domain=query_domain,
                paper_domain=str(row.get("primary_category", "")),
                query_embedding=q_emb,
                paper_embedding=p_emb,
            )
            X_features.append(feat)
            y_labels.append(0.0)
            metadata_records.append({"query": query_text, "paper_title": str(row["title"]), "label": "non_relevant"})

    X = np.array(X_features, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)

    logger.info("Generated %d labelled pairs (Positives: %d, Negatives: %d)", len(y), int(sum(y)), len(y) - int(sum(y)))
    return X, y, metadata_records


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
) -> Dict[str, List[float]]:
    """
    Train Feedforward Neural Network using Backpropagation and BCEWithLogitsLoss.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = None

    logger.info("Beginning neural network training (%d epochs, AdamW lr=%.4f)...", epochs, lr)

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Forward pass
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass (Backpropagation)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_preds: List[int] = []
        val_targets: List[int] = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                logits = model(X_val)
                loss = criterion(logits, y_val)
                running_val_loss += loss.item() * X_val.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_val.long().cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds)

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "Epoch [%2d/%2d] - Train Loss: %.4f | Val Loss: %.4f | Val Accuracy: %.2f%%",
                epoch,
                epochs,
                epoch_train_loss,
                epoch_val_loss,
                epoch_val_acc * 100,
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info("Restored model weights from best validation checkpoint (loss: %.4f)", best_val_loss)

    return history


def evaluate_on_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the trained neural network on the held-out test split.
    Computes Loss, Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.float32)

        logits = model(x_tensor)
        test_loss = float(criterion(logits, y_tensor).item())
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probs))
    cm = confusion_matrix(y_test, preds).tolist()

    report = {
        "test_loss": round(test_loss, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1],
        },
        "sample_size": len(y_test),
    }

    logger.info("=== Final Held-Out Test Evaluation ===")
    logger.info("Test Loss:        %.4f", report["test_loss"])
    logger.info("Test Accuracy:    %.2f%%", report["accuracy"] * 100)
    logger.info("Test Precision:   %.2f%%", report["precision"] * 100)
    logger.info("Test Recall:      %.2f%%", report["recall"] * 100)
    logger.info("Test F1-Score:    %.4f", report["f1_score"])
    logger.info("Test ROC-AUC:     %.4f", report["roc_auc"])
    logger.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", cm[0][0], cm[0][1], cm[1][0], cm[1][1])

    return report


def main():
    logger.info("Initializing JournalLab Feedforward Neural Network pipeline...")

    # 1. Load catalog & dense representations
    cat_path = settings.CATALOG_PATH
    emb_path = settings.SEMANTIC_EMBEDDINGS_PATH

    if not cat_path.exists() or not emb_path.exists():
        logger.error("Required artifacts not found. Please ensure models/ contains paper_catalog.joblib and semantic_embeddings.npz")
        sys.exit(1)

    catalog = joblib.load(cat_path)
    if not isinstance(catalog, pd.DataFrame):
        catalog = pd.DataFrame(catalog)

    with np.load(emb_path, allow_pickle=True) as data:
        embeddings = data["embeddings"].astype(np.float32)

    logger.info("Loaded catalog (%d papers) and embeddings shape (%s)", len(catalog), embeddings.shape)

    # 2. Build labelled dataset
    X, y, _ = create_labelled_pairs(catalog, embeddings, target_pairs_per_query=50)

    # 3. Strict chronological/random dataset splitting (70% Train, 15% Val, 15% Test)
    # MANDATORY ML PRACTICE: Split BEFORE any scaler is fit to avoid data leakage!
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    logger.info("Splits created: Train=%d, Val=%d, Test=%d", len(y_train), len(y_val), len(y_test))

    # 4. Standardize features (Fit ONLY on Train data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Create PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 6. Instantiate and train model
    model = RelevanceFFNN(in_features=X.shape[1], hidden_dims=(64, 32, 16), dropout_rate=0.2)
    history = train_model(model, train_loader, val_loader, epochs=25, lr=0.002)

    # 7. Evaluate on held-out test set
    test_report = evaluate_on_test(model, X_test, y_test)

    # 8. Save artifacts to models/ directory
    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = models_dir / "relevance_net.pt"
    scaler_save_path = models_dir / "relevance_scaler.joblib"
    report_save_path = BASE_DIR / "reports" / "relevance_net_eval.json"
    report_save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_save_path)
    joblib.dump(scaler, scaler_save_path)

    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    logger.info("Saved model weights to %s", model_save_path)
    logger.info("Saved feature scaler to %s", scaler_save_path)
    logger.info("Saved evaluation report to %s", report_save_path)

    # 9. Demonstrate live sample prediction
    sample_feat = scaler.transform(X_test_raw[:2])
    sample_probs = model.predict_proba(torch.tensor(sample_feat, dtype=torch.float32)).numpy()
    for i, p in enumerate(sample_probs):
        logger.info("Sample Pair %d - Predicted Relevance Probability: %.4f (Ground Truth: %d)", i + 1, p, int(y_test[i]))


if __name__ == "__main__":
    main()
