# JournalLab — AI-Based Research Paper Recommendation Platform

JournalLab is a scientific literature discovery and hypothesis exploration platform designed to recommend relevant academic papers and assist researchers with evidence-grounded AI analysis.

---

## Architecture Overview

1. **Hybrid Retrieval Core**:
   - **Lexical Matching**: Sublinear TF-IDF vectorizer over paper titles and abstracts.
   - **Dense Semantic Matching**: 384-dimensional dense sentence embeddings powered by `sentence-transformers/all-MiniLM-L6-v2`.
   - **Reciprocal Rank Fusion (RRF)**: Combines rankings using the formula $\frac{1}{60 + \text{rank}_{\text{lexical}}} + \frac{1}{60 + \text{rank}_{\text{semantic}}}$ to bridge the vocabulary gap without sacrificing keyword precision.
2. **Domain-Aware Filtering & Dual Input**:
   - Queries can be natural language research questions (e.g., *"Reducing hallucinations in RAG systems"*) or canonical arXiv IDs/URLs (e.g., `2006.07909` or `https://arxiv.org/abs/2006.07909`).
   - When searching via arXiv ID, the seed paper's title and abstract are used as query context, and the seed paper is automatically excluded from recommendations.
   - Supports domain filtering across Machine Learning, Natural Language Processing, Computer Vision, Artificial Intelligence, Robotics, Cybersecurity, Information Retrieval, and Databases.
3. **Evidence-Grounded RAG & Citation Guardrails**:
   - Synthesizes candidate research gaps, counter-evidence stress tests, and 10-section experiment plans using Groq LLM inference.
   - Programmatically verifies all generated `[arXiv:ID]` citations against the retrieved paper corpus to prevent hallucinations.
4. **Multimodal Architecture Extraction**:
   - Scrapes arXiv HTML5 and PDF versions for framework and pipeline diagrams, prioritizing architecture figures via keyword scoring and providing optional multimodal vision explanations.

---

## Prerequisites

- **Python**: 3.10+ (tested and verified on Python 3.14)
- **Node/Browser**: Any modern web browser (Chrome, Edge, Firefox, Safari). No frontend build step or npm required.
- **Optional**: Free Groq API Key (from [console.groq.com](https://console.groq.com/keys)) for LLM research gap reports and vision diagram explanations.

---

## Installation & Setup

### 1. Clone & Install Dependencies

```bash
git clone <repository_url>
cd "Research paper"
pip install -r requirements.txt
```

### 2. Dataset Verification

Ensure the enriched master dataset is placed at:
```
data/arxiv_papers_enriched_semantic_scholar_v1.csv
```
The dataset contains 4,900 papers with arXiv metadata and Semantic Scholar citation metrics.

### 3. Build Retrieval Artifacts (One-Time Setup)

Generate the TF-IDF index, dense sentence embeddings, and paper catalog:
```bash
python backend/build_index.py
```
This builds and saves the following artifacts into the `models/` directory:
- `models/tfidf_vectorizer.joblib`
- `models/tfidf_matrix.npz`
- `models/semantic_embeddings.npz`
- `models/paper_catalog.joblib`
- `models/manifest.json`

Once built, these artifacts are loaded directly into memory in < 1 second upon server startup.

### 4. Configure Environment Variables (Optional)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your Groq API key to enable AI Research Gap Analysis and Vision diagram explanations:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_RAG_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
```

---

## Running the Application

### 1. Start the FastAPI Backend

Run the server with Uvicorn:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
The API documentation is available at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 2. Open the Frontend

Simply open `index.html` in your web browser or serve it via a local static server:
```bash
# Option A: Double click index.html or open in browser
start index.html

# Option B: Use Python HTTP server
python -m http.server 3000
```
Navigate to `http://localhost:3000` (or your local file URL). The frontend automatically connects to `http://127.0.0.1:8000`.

---

## Frontend Features

- **Live Hybrid Search**: Type a research question, topic, or arXiv ID to get real-time recommendations.
- **Dynamic arXiv Fallback**: Query any arXiv ID (even outside the local dataset) — metadata is resolved on the fly via the official arXiv API.
- **BibTeX Citation Export**: Copy standard BibTeX citations with one click on any paper card, or export all saved papers as a `.bib` file.
- **Saved Papers Collection**: Bookmark papers with the "☆ Save" button; saved papers persist across page reloads via `localStorage`.
- **Abstract Accordions**: Expand and collapse paper abstracts on demand.
- **Direct Links**: Quick access to original arXiv abstracts and direct PDF downloads.
- **AI Research Gap Copilot (Streaming)**: Token-by-token live streaming of structured research opportunities and limitations from top retrieved papers.
- **Markdown Export**: Copy or download generated research gap reports and experimental plans directly as `.md` files.
- **Architecture Figures**: Click "Framework 🖼" to inspect system architecture diagrams extracted from arXiv.

---

## Running with Docker

Deploy with a single command using Docker Compose:

```bash
docker-compose up --build
```
Access the application at [http://localhost:8000](http://localhost:8000).

---

## Running Tests

Execute the automated test suite with `pytest`:

```bash
# Run all unit and integration tests
python -m pytest tests/

# Run individual test modules
python -m pytest tests/test_api.py
python -m pytest tests/test_retrieval.py
python -m pytest tests/test_hybrid_retrieval.py
```

---

## API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Check server and retrieval artifact readiness |
| `GET` | `/api/domains` | List all supported research domains |
| `POST` | `/api/recommend` | Hybrid search & recommendation (text or arXiv ID) |
| `GET` | `/api/papers/{arxiv_id}` | Fetch paper metadata (local catalog + live arXiv fallback) |
| `GET` | `/api/papers/{arxiv_id}/bibtex` | Generate formatted BibTeX citation for any paper |
| `POST` | `/api/research-gaps` | Synthesize research gaps from retrieved papers *(requires Groq)* |
| `POST` | `/api/research-gaps/stream` | Token-by-token live streaming of research gap analysis |
| `POST` | `/api/counter-evidence` | Stress-test gap hypothesis against prior art *(requires Groq)* |
| `POST` | `/api/plan-experiment` | Generate 10-section reproducible research plan *(requires Groq)* |
| `POST` | `/api/plan-experiment/stream` | Token-by-token live streaming of experiment plan |
| `GET` | `/api/papers/{arxiv_id}/figures` | Extract architecture and pipeline figures from arXiv |
| `POST` | `/api/papers/{arxiv_id}/figures/explain`| Vision AI explanation of paper diagram *(requires Groq)* |

---

## Limitations & Boundaries

1. **Local Corpus Scope**: Recommendations are ranked over the 4,900 papers present in `data/arxiv_papers_enriched_semantic_scholar_v1.csv`.
2. **LLM Credentials**: Core hybrid search runs entirely locally without any external API keys. AI research gap reports and vision explanations require `GROQ_API_KEY`.
3. **arXiv Rate Limits**: arXiv figure extraction scrapes arXiv HTML/PDF. Figure results are cached locally in `data/paper_figures/` to minimize requests.
