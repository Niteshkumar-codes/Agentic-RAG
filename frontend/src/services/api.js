/**
 * Agentic-RAG Native Fetch API Client Service
 */

const BASE_URL = '' // Uses Vite dev proxy (or relative URL in production)

async function handleResponse(response) {
  if (!response.ok) {
    let detailMessage = `HTTP Error ${response.status}`
    try {
      const errorJson = await response.json()
      if (errorJson && errorJson.detail) {
        if (typeof errorJson.detail === 'string') {
          detailMessage = errorJson.detail
        } else if (Array.isArray(errorJson.detail)) {
          // FastAPI Pydantic 422 validation errors
          detailMessage = errorJson.detail
            .map((err) => `${err.loc ? err.loc.join('.') : 'field'}: ${err.msg}`)
            .join(' | ')
        }
      }
    } catch {
      // Non-JSON response body
    }
    throw new Error(detailMessage)
  }
  return await response.json()
}

/**
 * Checks backend API health status.
 * GET /health
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: 'GET' })
    return await handleResponse(res)
  } catch (err) {
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error('Backend server is unreachable (Failed to connect to http://127.0.0.1:8000).')
    }
    throw err
  }
}

/**
 * Fetches collection statistics.
 * GET /api/v1/stats
 */
export async function getStats() {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/stats`, { method: 'GET' })
    return await handleResponse(res)
  } catch (err) {
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error('Backend server is unreachable.')
    }
    throw err
  }
}

/**
 * Ingests a document file (.pdf or .txt) with chunking parameters.
 * POST /api/v1/documents/ingest
 */
export async function ingestDocument(file, options = {}) {
  const { chunkSize = 500, chunkOverlap = 100, overwriteDuplicates = false } = options

  const formData = new FormData()
  formData.append('file', file)

  const queryParams = new URLSearchParams({
    chunk_size: String(chunkSize),
    chunk_overlap: String(chunkOverlap),
    overwrite_duplicates: String(overwriteDuplicates),
  })

  try {
    const res = await fetch(`${BASE_URL}/api/v1/documents/ingest?${queryParams.toString()}`, {
      method: 'POST',
      body: formData,
    })
    return await handleResponse(res)
  } catch (err) {
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error('Upload failed. Backend server is unreachable.')
    }
    throw err
  }
}

/**
 * Executes semantic vector similarity search.
 * POST /api/v1/search
 */
export async function searchDocuments(query, topK = 4) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query.trim(),
        top_k: Number(topK),
      }),
    })
    return await handleResponse(res)
  } catch (err) {
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error('Search failed. Backend server is unreachable.')
    }
    throw err
  }
}

/**
 * Generates grounded answer for a user question using RAG pipeline.
 * POST /api/v1/query
 */
export async function queryRag(question, topK = 4) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question.trim(),
        top_k: Number(topK),
      }),
    })
    return await handleResponse(res)
  } catch (err) {
    if (err.message && err.message.includes('Failed to fetch')) {
      throw new Error('Question answering failed. Backend server is unreachable.')
    }
    throw err
  }
}
