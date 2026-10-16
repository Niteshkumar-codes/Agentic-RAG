# Agentic-RAG Backend API Guide

Welcome to the **Agentic-RAG Backend API Guide**. This guide provides a beginner-friendly, complete reference for interacting with the existing FastAPI REST backend.

---

## 1. Overview & Architecture

The Agentic-RAG backend is built using **FastAPI** and served with **Uvicorn**. It powers a Retrieval-Augmented Generation (RAG) pipeline composed of:

- **Document Parsing**: Extracts text from `.pdf` (via PyMuPDF) and `.txt` files.
- **Text Chunking**: Splits parsed document text into sliding character windows with configurable size and overlap.
- **Embedding Generation**: Generates 768-dimensional text embeddings using Google Gemini (`text-embedding-004`).
- **Vector Storage**: Stores embeddings and metadata in a local persistent **ChromaDB** collection (`agentic_rag_collection`).
- **Semantic Retrieval**: Performs cosine-distance vector similarity searches over stored document chunks.
- **Grounded Answer Generation**: Uses Google Gemini (`gemini-2.5-flash`) to generate answers strictly grounded in retrieved context chunks, automatically validating citations (`[S1]`, `[S2]`).

```
Client (Dashboard / cURL / Script)
       │
       ▼
 FastAPI Backend (http://127.0.0.1:8000)
 ├── GET  /health
 ├── GET  /api/v1/stats
 ├── POST /api/v1/documents/ingest ──► PyMuPDF ──► Chunker ──► Gemini Embeddings ──► ChromaDB
 ├── POST /api/v1/search           ──► Gemini Embeddings ──► ChromaDB Vector Search
 └── POST /api/v1/query            ──► Semantic Search ──► Gemini 2.5 Flash (Grounded Q&A)
```

---

## 2. Important Notes & Prerequisites

> [!WARNING]
> **Gemini Credentials Requirement**
> - The server can start and serve `GET /health` and `GET /api/v1/stats` completely offline without any API keys.
> - However, live execution of document ingestion (`/api/v1/documents/ingest`), vector search (`/api/v1/search`), and Q&A answer generation (`/api/v1/query`) requires a valid Google Gemini API key configured in `backend/.env` as `GEMINI_API_KEY=your_key_here`.
> - Without a valid key, calling ingestion, search, or query endpoints will return `500 Internal Server Error` with a sanitized error message (`"Embedding generation failed. Please try again later."` or `"Context retrieval failed. Please try again later."`).

> [!NOTE]
> **Implementation Scope & Roadmap**
> - This guide documents the **current, active implementation** in `backend/app/main.py` and `backend/app/api/endpoints.py`.
> - Planned features such as **LangGraph multi-hop agentic reasoning loops** and **n8n workflow pipelines** (`n8n/workflows/`) are roadmapped for future releases and are **not yet implemented** in the active API.

---

## 3. Starting the Local Backend

Open Windows PowerShell and navigate to the `backend/` directory:

```powershell
cd C:\Users\NITESH\Desktop\Agentic-RAG\backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

*Alternative direct execution (if PowerShell script execution policy prevents running `Activate.ps1`):*

```powershell
cd C:\Users\NITESH\Desktop\Agentic-RAG\backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once started, the backend listens at `http://127.0.0.1:8000`.

---

## 4. Interactive Documentation (Swagger UI & ReDoc)

FastAPI automatically generates interactive API documentation. While the server is running, open these URLs in your browser:

- **Swagger UI (Interactive Playground)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  - Allows you to inspect request/response schemas and test every endpoint directly in your browser.
- **ReDoc (API Reference)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
  - Provides a clean, readable documentation layout.

---

## 5. API Endpoint Reference

### 5.1. Health Check

Verifies that the backend server is running and responsive. Does not touch the database or external APIs.

- **HTTP Method**: `GET`
- **Path**: `/health`
- **Authentication**: None
- **Query Parameters**: None
- **Request Body**: None

#### Example Request

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

*cURL equivalent:*

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

#### Successful Response (`200 OK`)

```json
{
  "status": "healthy",
  "message": "Agentic-RAG backend is running"
}
```

---

### 5.2. Vector Store Collection Statistics

Retrieves metadata about the ChromaDB vector store collection, including the collection name, database storage directory, and total number of stored chunks.

- **HTTP Method**: `GET`
- **Path**: `/api/v1/stats`
- **Authentication**: None
- **Query Parameters**: None
- **Request Body**: None

#### Example Request

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/stats" -Method Get
```

*cURL equivalent:*

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/stats"
```

#### Successful Response (`200 OK`)

Response model: `CollectionStatsResponse`

```json
{
  "collection_name": "agentic_rag_collection",
  "persist_directory": "chroma_db",
  "total_records": 12
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `collection_name` | `string` | The ChromaDB collection name (defaults to `agentic_rag_collection`). |
| `persist_directory` | `string` | The directory on disk where vectors are persisted (defaults to `chroma_db`). |
| `total_records` | `integer` | Total number of chunk embeddings currently stored in the collection. |

#### Error Responses

- **`500 Internal Server Error`**:
  Raised if ChromaDB fails to connect or read collection stats.
  ```json
  {
    "detail": "Failed to retrieve vector store collection statistics."
  }
  ```

---

### 5.3. Document Ingestion

Uploads a `.pdf` or `.txt` document, extracts its text, splits it into sliding window chunks, generates Gemini embeddings, and stores them in ChromaDB.

- **HTTP Method**: `POST`
- **Path**: `/api/v1/documents/ingest`
- **Content-Type**: `multipart/form-data`

#### Parameters

| Parameter | Location | Type | Default | Required | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `file` | Form body | File | — | **Yes** | `.pdf` or `.txt`, size $> 0$ bytes | The document file to ingest. |
| `chunk_size` | Query / Form | `integer` | `500` | No | Must be $> 0$ | Maximum characters per chunk window. |
| `chunk_overlap` | Query / Form | `integer` | `100` | No | Must be $\ge 0$ and $< \text{chunk\_size}$ | Number of overlapping characters between chunks. |
| `overwrite_duplicates` | Query / Form | `boolean` | `false` | No | `true` or `false` | If `true`, overwrites existing chunks with matching IDs; if `false`, raises `409 Conflict`. |

#### Example Request

Using PowerShell:

```powershell
$form = @{
    file = Get-Item "C:\path\to\document.pdf"
    chunk_size = 500
    chunk_overlap = 100
    overwrite_duplicates = $true
}
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/documents/ingest" -Method Post -Form $form
```

Using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/ingest?chunk_size=500&chunk_overlap=100&overwrite_duplicates=true" \
  -F "file=@sample_paper.pdf;type=application/pdf"
```

#### Successful Response (`200 OK`)

Response model: `DocumentIngestionResponse`

```json
{
  "filename": "sample_paper.pdf",
  "total_pages": 4,
  "total_chunks": 18,
  "added_count": 18,
  "collection_name": "agentic_rag_collection",
  "vector_dimension": 768,
  "status": "success"
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `filename` | `string` | The original name of the uploaded document. |
| `total_pages` | `integer` | Number of pages extracted from the document ($1$ for `.txt` files). |
| `total_chunks` | `integer` | Total number of chunks generated by the chunker. |
| `added_count` | `integer` | Number of chunk vectors stored into ChromaDB. |
| `collection_name` | `string` | Target ChromaDB collection name. |
| `vector_dimension` | `integer` | Dimensionality of embeddings generated ($768$ for `text-embedding-004`). |
| `status` | `string` | Operation status (`"success"` or `"no_op"` if document was blank). |

#### Error Responses

- **`400 Bad Request`**:
  - Missing file:
    ```json
    { "detail": "File must be provided." }
    ```
  - Unsupported file type:
    ```json
    { "detail": "Unsupported file format '.docx'. Only .pdf and .txt files are supported." }
    ```
  - Empty file:
    ```json
    { "detail": "Uploaded file 'notes.txt' is empty (0 bytes)." }
    ```
  - Invalid chunk parameters:
    ```json
    { "detail": "chunk_size must be greater than 0, got 0." }
    ```
    ```json
    { "detail": "chunk_overlap cannot be negative, got -10." }
    ```
    ```json
    { "detail": "chunk_overlap (500) must be strictly less than chunk_size (500)." }
    ```
- **`409 Conflict`**:
  - Duplicate chunks already exist in ChromaDB and `overwrite_duplicates` is `false`:
    ```json
    { "detail": "Duplicate document chunks already exist. Set overwrite_duplicates=true to replace them." }
    ```
- **`500 Internal Server Error`**:
  - Embedding failure (missing or invalid Gemini API key):
    ```json
    { "detail": "Embedding generation failed. Please try again later." }
    ```
  - Vector database write failure:
    ```json
    { "detail": "Vector store operation failed. Please try again later." }
    ```

---

### 5.4. Semantic Vector Search

Executes a semantic vector similarity search against the document chunks stored in ChromaDB. The query string is converted to an embedding vector and compared against stored chunks using cosine distance.

- **HTTP Method**: `POST`
- **Path**: `/api/v1/search`
- **Content-Type**: `application/json`

#### Request Schema (`SearchRequest`)

```json
{
  "query": "How does attention mechanism improve machine translation?",
  "top_k": 3
}
```

| Field | Type | Required | Default | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | — | Non-empty, non-whitespace | Natural language search query. Whitespace is trimmed automatically. |
| `top_k` | `integer` | No | `4` | Must be $> 0$ (`gt=0`) | Maximum number of relevant chunks to retrieve. |

#### Example Request

Using PowerShell:

```powershell
$body = @{
    query = "How does attention mechanism improve machine translation?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/search" -Method Post -ContentType "application/json" -Body $body
```

Using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does attention mechanism improve machine translation?", "top_k": 3}'
```

#### Successful Response (`200 OK`)

Response model: `SearchResponse`

```json
{
  "query": "How does attention mechanism improve machine translation?",
  "results_count": 2,
  "results": [
    {
      "chunk_id": "attention_paper.pdf_p1_c1_3fa85f",
      "text": "The attention mechanism allows the decoder to focus on specific parts of the source sentence.",
      "source_filename": "attention_paper.pdf",
      "page_number": 1,
      "chunk_index": 1,
      "model_name": "text-embedding-004",
      "distance": 0.1245
    },
    {
      "chunk_id": "attention_paper.pdf_p3_c5_6be91a",
      "text": "By dynamically computing context vectors at each decoding step, performance on long sentences increases significantly.",
      "source_filename": "attention_paper.pdf",
      "page_number": 3,
      "chunk_index": 5,
      "model_name": "text-embedding-004",
      "distance": 0.2319
    }
  ]
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `query` | `string` | The submitted search query. |
| `results_count` | `integer` | Number of matched chunks returned. |
| `results` | `array` | List of `SearchResultItem` objects ordered by similarity. |
| `results[].chunk_id` | `string` | Unique deterministic chunk identifier. |
| `results[].text` | `string` | The extracted text content of the chunk. |
| `results[].source_filename` | `string` | Name of the source document file. |
| `results[].page_number` | `integer` | Source page number ($1$-indexed). |
| `results[].chunk_index` | `integer` | Index of the chunk in the ingestion sequence. |
| `results[].model_name` | `string` | Embedding model used (`text-embedding-004`). |
| `results[].distance` | `float` | Cosine distance score (lower value indicates higher similarity). |

*(If the collection contains no records, `results_count` is `0` and `results` is `[]`.)*

#### Error Responses

- **`422 Unprocessable Entity`**:
  Raised by FastAPI/Pydantic validation if `query` is blank or `top_k <= 0`.
  ```json
  {
    "detail": [
      {
        "type": "value_error",
        "loc": ["body", "query"],
        "msg": "Value error, query must not be empty or whitespace-only"
      }
    ]
  }
  ```
- **`400 Bad Request`**:
  Raised if there is an embedding model mismatch (e.g. stored chunks were embedded with a different model).
  ```json
  {
    "detail": "Invalid search parameters or embedding model mismatch."
  }
  ```
- **`500 Internal Server Error`**:
  Raised if embedding generation fails (missing API key) or database query fails.
  ```json
  {
    "detail": "Semantic vector search failed. Please try again later."
  }
  ```

---

### 5.5. Grounded Question Answering (RAG Query)

Executes the complete RAG pipeline:
1. Performs semantic vector retrieval to find relevant document chunks.
2. Injects chunks into a grounded prompt referencing `[S1]`, `[S2]`, etc.
3. Requests answer generation from Gemini (`gemini-2.5-flash`).
4. Cleans and validates citation tags against actually supplied source context chunks.
5. Evaluates whether sufficient context existed to answer the question.

- **HTTP Method**: `POST`
- **Path**: `/api/v1/query`
- **Content-Type**: `application/json`

#### Request Schema (`QueryRequest`)

```json
{
  "question": "What is the primary benefit of multi-head attention?",
  "top_k": 4
}
```

| Field | Type | Required | Default | Validation Rules | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `question` | `string` | **Yes** | — | Non-empty, non-whitespace | User question to be answered from ingested context. |
| `top_k` | `integer` | No | `4` | Must be $> 0$ (`gt=0`) | Number of context chunks to retrieve for grounding. |

#### Example Request

Using PowerShell:

```powershell
$body = @{
    question = "What is the primary benefit of multi-head attention?"
    top_k = 4
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/query" -Method Post -ContentType "application/json" -Body $body
```

Using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the primary benefit of multi-head attention?", "top_k": 4}'
```

#### Successful Response (`200 OK`)

Response model: `QueryResponse`

```json
{
  "question": "What is the primary benefit of multi-head attention?",
  "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions [S1]. This prevents single attention mechanisms from averaging out important features [S2].",
  "sources": [
    {
      "source_label": "[S1]",
      "source_filename": "attention_paper.pdf",
      "page_number": 4,
      "chunk_id": "attention_paper.pdf_p4_c8_4c1e2f"
    },
    {
      "source_label": "[S2]",
      "source_filename": "attention_paper.pdf",
      "page_number": 5,
      "chunk_id": "attention_paper.pdf_p5_c9_9b8a7c"
    }
  ],
  "sufficient_context": true,
  "model_name": "gemini-2.5-flash"
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `question` | `string` | The submitted question. |
| `answer` | `string` | The generated answer, strictly grounded in retrieved chunks with citations. |
| `sources` | `array` | List of `SourceReferenceItem` objects matching the citations in the answer. |
| `sources[].source_label` | `string` | Reference label used in text (e.g., `"[S1]"`). |
| `sources[].source_filename` | `string` | Filename of the source document. |
| `sources[].page_number` | `integer` | Page number in the source file where the chunk originated. |
| `sources[].chunk_id` | `string` | Identifier of the chunk. |
| `sufficient_context` | `boolean` | `true` if document context contained enough info to answer; `false` otherwise. |
| `model_name` | `string` | Generative model used (`gemini-2.5-flash`). |

#### When Document Context is Insufficient

If no documents match the query or the database is empty:
- The backend does **not** call the Gemini generative API.
- It returns `200 OK` with:
  ```json
  {
    "question": "What is the chemical composition of dark matter?",
    "answer": "I do not have enough information in the provided document context to answer your question.",
    "sources": [],
    "sufficient_context": false,
    "model_name": "gemini-2.5-flash"
  }
  ```

#### Error Responses

- **`422 Unprocessable Entity`**:
  Raised if `question` is missing, empty, or whitespace-only, or if `top_k <= 0`.
- **`400 Bad Request`**:
  Raised if embedding model mismatch occurs during retrieval.
  ```json
  {
    "detail": "Invalid search parameters or embedding model mismatch."
  }
  ```
- **`500 Internal Server Error`**:
  - Context retrieval failure:
    ```json
    { "detail": "Context retrieval failed. Please try again later." }
    ```
  - Answer generation failure (e.g. Gemini API error):
    ```json
    { "detail": "Answer generation failed. Please try again later." }
    ```

---

## 6. Summary of Endpoints and HTTP Status Codes

| Endpoint | Method | Success | Client Errors | Server Errors | Offline Usable? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/health` | `GET` | `200` | — | — | **Yes** |
| `/api/v1/stats` | `GET` | `200` | — | `500` | **Yes** |
| `/api/v1/documents/ingest` | `POST` | `200` | `400`, `409` | `500` | No (requires Gemini embedding) |
| `/api/v1/search` | `POST` | `200` | `400`, `422` | `500` | No (requires Gemini embedding) |
| `/api/v1/query` | `POST` | `200` | `400`, `422` | `500` | No (requires Gemini generation) |

---

## 7. Troubleshooting Tips

1. **Backend fails to start (`ModuleNotFoundError`)**:
   Make sure you activate the virtual environment (`.\venv\Scripts\Activate.ps1`) or invoke Python directly through `.\venv\Scripts\python.exe`.
2. **`422 Unprocessable Entity` on Search or Query**:
   Check that your request body is valid JSON with `Content-Type: application/json` and that `query` or `question` is not empty or pure spaces.
3. **`409 Conflict` during Document Ingestion**:
   The uploaded document contains chunks that were already ingested into ChromaDB. Pass `overwrite_duplicates=true` to replace existing chunks.
4. **`500 Internal Server Error` during Ingestion, Search, or Query**:
   Confirm that your `GEMINI_API_KEY` is set in `backend/.env` or in your system environment variables.
