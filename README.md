# Agentic-RAG System

Agentic-RAG is a full-stack Retrieval-Augmented Generation (RAG) system designed to go beyond simple search retrieval. It provides document ingestion, text chunking, vector similarity search using ChromaDB and Google Gemini embeddings (`text-embedding-004`), and grounded Q&A answer generation (`gemini-2.5-flash`) with verifiable source citations.

---

## Implemented vs. Planned Features

### Implemented Features
- **Document Ingestion & Parsing**: Extract text page-by-page from `.pdf` and `.txt` documents using PyMuPDF and UTF-8 / Latin-1 decoding.
- **Configurable Text Chunking**: Slide character windows with user-defined chunk size and chunk overlap parameters.
- **Vector Database Storage**: Store document embeddings and metadata in local persistent ChromaDB collections with deduplication support.
- **Semantic Vector Search**: Perform cosine distance vector similarity searches over ingested context chunks.
- **Grounded Answer Generation & Citation Validation**: Generate Q&A responses grounded strictly in retrieved document context, with automated verification of `[S1]`, `[S2]` source labels and context sufficiency flags.
- **Interactive Dashboard**: Modern dark-mode React dashboard with real-time backend connection status, collection statistics, file drop zone, vector search inspector, and Q&A console.
- **Robust Error & Input Handling**: Client-side parameter validation (`chunk_overlap < chunk_size`), 10 MB file size limit, empty file detection, accessible notification alerts, and safe non-JSON HTTP response handling.
- **Automated Backend Testing**: 67 automated unit and integration tests covering parser, chunker, embeddings, vector store, retrieval, answer generation, schemas, and API endpoints.

### Planned Features (Future Roadmap)
- **Multi-Hop Agentic Planning & Query Reformulation**: Query planning and multi-hop reasoning loops via LangGraph.
- **n8n Workflow Automation**: Automated document pipelines and trigger integrations via n8n workflows (`n8n/workflows/`).
- **Live Gemini API Key Verification**: End-to-end cloud verification of Gemini API models in production.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              React Dashboard (Vite + Tailwind CSS v4)   │
│   • Document Upload  • Vector Search  • Grounded Q&A    │
└────────────────────────────┬────────────────────────────┘
                             │ REST API (Fetch / Proxy)
┌────────────────────────────▼────────────────────────────┐
│                  FastAPI Backend Server                 │
│   • /health             • /api/v1/stats                 │
│   • /documents/ingest   • /api/v1/search  • /v1/query   │
└──────┬─────────────────────┬────────────────────┬───────┘
       │                     │                    │
┌──────▼──────┐       ┌──────▼──────┐      ┌──────▼──────┐
│  PyMuPDF    │       │  ChromaDB   │      │ Google GenAI│
│ Text Parser │       │ VectorStore │      │ Gemini API  │
└─────────────┘       └─────────────┘      └─────────────┘
```

- **Frontend**: Built with React 19, Vite 8, and Tailwind CSS v4. Communicates with backend endpoints via Vite dev proxy (`http://127.0.0.1:8000`).
- **Backend**: Built with FastAPI, Pydantic v2 validation models, and Uvicorn server.
- **RAG Pipeline**:
  1. **Document Ingestion**: File validation $\rightarrow$ PyMuPDF parsing $\rightarrow$ Sliding window chunking.
  2. **Embedding & Storage**: `google-genai` SDK (`text-embedding-004`, 768-dim) $\rightarrow$ ChromaDB storage.
  3. **Retrieval**: Cosine similarity vector search over stored document embeddings.
  4. **Grounded Generation**: Structured prompt grounding $\rightarrow$ `gemini-2.5-flash` text generation $\rightarrow$ Citation cleaning and context sufficiency evaluation.

---

## Repository Structure

```
Agentic-RAG/
├── backend/                        # FastAPI Backend Application
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py        # REST API router (/health, /ingest, /search, /query, /stats)
│   │   ├── services/
│   │   │   ├── answer_generation_service.py # Grounded Q&A generation & citation validation
│   │   │   ├── document_parser.py           # PDF (PyMuPDF) and TXT extraction
│   │   │   ├── embedding_service.py         # Google Gemini embedding service
│   │   │   ├── retrieval_service.py         # Vector search retrieval service
│   │   │   ├── text_chunker.py              # Character-based sliding window chunker
│   │   │   └── vector_store.py              # Local persistent ChromaDB vector store
│   │   ├── main.py                 # FastAPI application entrypoint
│   │   └── schemas.py              # Pydantic data request/response models
│   ├── tests/                      # Automated pytest test suite (67 passing tests)
│   ├── .env.example                # Environment variables template
│   └── requirements.txt            # Python dependencies
├── frontend/                       # React Single Page Dashboard
│   ├── public/                     # Static assets (favicons, icons)
│   ├── src/
│   │   ├── services/
│   │   │   └── api.js              # Fetch client API service
│   │   ├── App.jsx                 # Dashboard UI component
│   │   ├── index.css               # Tailwind CSS imports & base styles
│   │   └── main.jsx                # React root renderer
│   ├── index.html                  # Dashboard HTML document
│   ├── package.json                # Node dependencies & npm scripts
│   ├── vite.config.js              # Vite server & proxy configuration
│   └── .oxlintrc.json              # OxLint configuration
├── chroma_db/                      # Local persistent ChromaDB data store directory
├── n8n/                            # Planned n8n integration workflows
│   └── workflows/
├── .gitignore                      # Git ignore file (excludes secrets, venv, node_modules)
└── README.md                       # Project documentation
```

---

## Prerequisites

Before running the project on Windows PowerShell, ensure you have installed:
- **Python**: Version `3.10` to `3.13`
- **Node.js**: Version `18.x` or higher (includes `npm`)
- **Git**

---

## Setup & Installation

### 1. Environment Configuration
Copy the template environment file inside `backend/`:

```powershell
Copy-Item backend\.env.example backend\.env
```

Open `backend\.env` in a text editor and add your Google Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
APP_ENV=development
PORT=8000
HOST=127.0.0.1
CHROMA_DB_DIR=./chroma_db
```
> **Security Notice**: Never commit `.env` to Git repository. `.env` is listed in `.gitignore`.

### 2. Backend Setup
Navigate to the `backend` directory, create a virtual environment, and install Python dependencies:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Frontend Setup
In a new terminal window, navigate to `frontend` and install dependencies:

```powershell
cd frontend
npm install
```

---

## Running the Application

### 1. Start the Backend Server
From the `backend` directory (with virtual environment activated):

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
The FastAPI backend will run at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Start the Frontend Dashboard
From the `frontend` directory:

```powershell
npm run dev
```
Open your browser at `http://localhost:5173` to access the Agentic-RAG Dashboard.

---

## Frontend Commands

Inside `frontend`:
- **Development Server**: `npm run dev`
- **Lint Check**: `npm run lint` (runs `oxlint`)
- **Production Build**: `npm run build` (runs `vite build`)
- **Preview Production Build**: `npm run preview`

---

## Backend Automated Tests

To execute the 67 automated unit and integration tests, run from the `backend` directory:

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest -q
```
*(Or `python -m pytest` if the virtual environment is activated).*

---

## REST API Endpoint Reference

| HTTP Method | Endpoint Path | Purpose / Description |
|---|---|---|
| `GET` | `/health` | Backend connection health check (`{"status": "healthy"}`). |
| `GET` | `/api/v1/stats` | Retrieves collection name, persistence directory, and total record count. |
| `POST` | `/api/v1/documents/ingest` | Ingests `.pdf`/`.txt` files with `chunk_size`, `chunk_overlap`, and `overwrite_duplicates`. |
| `POST` | `/api/v1/search` | Executes semantic vector similarity search with `{ "query": string, "top_k": int }`. |
| `POST` | `/api/v1/query` | Generates grounded Q&A response with `{ "question": string, "top_k": int }`. |

---

## Current Limitations

1. **Unverified Live Gemini Integration**: Live network calls to Gemini endpoints require a valid `GEMINI_API_KEY` in `backend/.env`. Live cloud calls have intentionally not been executed in this local environment.
2. **LangGraph Agentic Loops**: Agentic multi-hop query planning and tool-calling loops are planned for future iterations.
3. **n8n Automation**: Workflow JSON definitions in `n8n/workflows/` are placeholders for planned automation pipelines.
