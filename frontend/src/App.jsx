import { useState, useEffect } from 'react'
import {
  checkHealth,
  getStats,
  ingestDocument,
  searchDocuments,
  queryRag,
} from './services/api'

export default function App() {
  // Connection & System Stats State
  const [isBackendConnected, setIsBackendConnected] = useState(false)
  const [stats, setStats] = useState({
    collection_name: 'agentic_rag_collection',
    persist_directory: 'chroma_db',
    total_records: 0,
  })

  // Document Ingestion State
  const [selectedFile, setSelectedFile] = useState(null)
  const [chunkSize, setChunkSize] = useState(500)
  const [chunkOverlap, setChunkOverlap] = useState(100)
  const [overwriteDuplicates, setOverwriteDuplicates] = useState(false)
  const [isIngesting, setIsIngesting] = useState(false)
  const [ingestResponse, setIngestResponse] = useState(null)

  // Semantic Search State
  const [searchQuery, setSearchQuery] = useState('')
  const [searchTopK, setSearchTopK] = useState(4)
  const [isSearching, setIsSearching] = useState(false)
  const [searchResults, setSearchResults] = useState(null)

  // RAG Query Console State
  const [questionQuery, setQuestionQuery] = useState('')
  const [queryTopK, setQueryTopK] = useState(4)
  const [isQuerying, setIsQuerying] = useState(false)
  const [queryResponse, setQueryResponse] = useState(null)

  // UI Notification & Drag State
  const [uiNotice, setUiNotice] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 // 10 MB limit

  // Fetch initial health status and statistics on load
  const loadSystemStatus = async () => {
    try {
      await checkHealth()
      setIsBackendConnected(true)
    } catch {
      setIsBackendConnected(false)
    }

    try {
      const statsData = await getStats()
      setStats(statsData)
    } catch {
      // Keep default stats
    }
  }

  useEffect(() => {
    loadSystemStatus()
  }, [])

  const validateFile = (file) => {
    if (!file) return false
    const ext = file.name.split('.').pop().toLowerCase()
    if (!['pdf', 'txt'].includes(ext)) {
      setUiNotice({
        type: 'error',
        message: `Unsupported file format .${ext}. Only .pdf and .txt files are supported.`,
      })
      return false
    }
    if (file.size === 0) {
      setUiNotice({
        type: 'error',
        message: `Selected file "${file.name}" is empty (0 bytes). Please select a valid document with content.`,
      })
      return false
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setUiNotice({
        type: 'error',
        message: `Selected file "${file.name}" (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 10 MB size limit.`,
      })
      return false
    }
    return true
  }

  // File Selection Handlers
  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (validateFile(file)) {
        setSelectedFile(file)
        setIngestResponse(null)
        setUiNotice(null)
      } else {
        setSelectedFile(null)
        e.target.value = ''
      }
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) {
      if (validateFile(file)) {
        setSelectedFile(file)
        setIngestResponse(null)
        setUiNotice(null)
      } else {
        setSelectedFile(null)
      }
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    setIngestResponse(null)
  }

  // 1. Ingest Document Action
  const handleIngestSubmit = async (e) => {
    e.preventDefault()
    if (!selectedFile || isIngesting) return

    if (!chunkSize || chunkSize <= 0) {
      setUiNotice({
        type: 'error',
        message: 'Chunk size must be greater than 0.',
      })
      return
    }

    if (chunkOverlap < 0) {
      setUiNotice({
        type: 'error',
        message: 'Chunk overlap cannot be negative.',
      })
      return
    }

    if (chunkOverlap >= chunkSize) {
      setUiNotice({
        type: 'error',
        message: `Chunk overlap (${chunkOverlap}) must be strictly less than chunk size (${chunkSize}).`,
      })
      return
    }

    setIsIngesting(true)
    setUiNotice(null)
    setIngestResponse(null)

    try {
      const res = await ingestDocument(selectedFile, {
        chunkSize,
        chunkOverlap,
        overwriteDuplicates,
      })

      setIngestResponse(res)
      setUiNotice({
        type: 'success',
        message: `Successfully ingested "${res.filename}"! Added ${res.added_count} chunks to collection "${res.collection_name}".`,
      })

      // Refresh vector store collection stats
      await loadSystemStatus()
    } catch (err) {
      setUiNotice({
        type: 'error',
        message: err.message || 'Document ingestion failed.',
      })
    } finally {
      setIsIngesting(false)
    }
  }

  // 2. Semantic Search Action
  const handleSearchSubmit = async (e) => {
    e.preventDefault()
    if (!searchQuery.trim() || isSearching) return

    setIsSearching(true)
    setSearchResults(null)
    setUiNotice(null)

    try {
      const res = await searchDocuments(searchQuery, searchTopK)
      setSearchResults(res)
    } catch (err) {
      setUiNotice({
        type: 'error',
        message: err.message || 'Semantic search failed.',
      })
    } finally {
      setIsSearching(false)
    }
  }

  // 3. Grounded RAG Query Action
  const handleQuerySubmit = async (e) => {
    e.preventDefault()
    if (!questionQuery.trim() || isQuerying) return

    setIsQuerying(true)
    setQueryResponse(null)
    setUiNotice(null)

    try {
      const res = await queryRag(questionQuery, queryTopK)
      setQueryResponse(res)
    } catch (err) {
      setUiNotice({
        type: 'error',
        message: err.message || 'Question answering failed.',
      })
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header Bar */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <span className="font-bold text-lg bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
                Agentic-RAG
              </span>
              <span className="ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                v0.1.0
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Live Backend Health Badge */}
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border text-xs transition-colors ${
              isBackendConnected
                ? 'bg-emerald-950/60 border-emerald-800/80 text-emerald-300'
                : 'bg-amber-950/60 border-amber-800/80 text-amber-300'
            }`}>
              <span className={`h-2 w-2 rounded-full ${isBackendConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
              <span>
                Backend:{' '}
                <strong className="font-semibold">
                  {isBackendConnected ? 'Connected' : 'Not connected'}
                </strong>
              </span>
            </div>

            <button
              onClick={loadSystemStatus}
              className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white transition-colors text-xs"
              title="Refresh connection status & stats"
              aria-label="Refresh connection status and statistics"
            >
              <span aria-hidden="true">🔄</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Banner Notice / Error Alert */}
        {uiNotice && (
          <div
            role="alert"
            aria-live="polite"
            className={`p-4 rounded-xl border flex items-center justify-between text-sm transition-all shadow-md ${
              uiNotice.type === 'success'
                ? 'bg-emerald-950/80 border-emerald-800/80 text-emerald-200'
                : uiNotice.type === 'error'
                ? 'bg-rose-950/80 border-rose-800/80 text-rose-200'
                : 'bg-indigo-950/80 border-indigo-800/80 text-indigo-200'
            }`}
          >
            <div className="flex items-start space-x-3">
              <span className="text-base mt-0.5">
                {uiNotice.type === 'success' ? '✓' : uiNotice.type === 'error' ? '⚠️' : 'ℹ️'}
              </span>
              <div className="space-y-0.5">
                <p className="font-medium">{uiNotice.message}</p>
                {!isBackendConnected && uiNotice.type === 'error' && (
                  <p className="text-xs text-slate-400">
                    Tip: Start the backend server with command: <code className="text-amber-300 bg-slate-900 px-1.5 py-0.5 rounded">python -m uvicorn app.main:app --reload</code>
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={() => setUiNotice(null)}
              className="text-slate-400 hover:text-white transition-colors focus:outline-none rounded p-1"
              aria-label="Dismiss notice"
            >
              ✕
            </button>
          </div>
        )}

        {/* Live Vector Collection Statistics Bar */}
        <section aria-label="Vector Store Statistics">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between shadow-sm">
              <span className="text-xs font-medium text-slate-400">Collection Name</span>
              <div className="mt-2">
                <span className="text-sm font-semibold text-slate-100 font-mono block truncate">
                  {stats.collection_name}
                </span>
                <span className="text-[11px] text-slate-500">ChromaDB Store</span>
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between shadow-sm">
              <span className="text-xs font-medium text-slate-400">Stored Chunks Count</span>
              <div className="flex items-baseline space-x-2 mt-2">
                <span className="text-2xl font-bold text-indigo-400">
                  {stats.total_records}
                </span>
                <span className="text-xs text-slate-500">records</span>
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between shadow-sm">
              <span className="text-xs font-medium text-slate-400">Embedding Model</span>
              <div className="mt-2">
                <span className="text-sm font-semibold text-indigo-300 font-mono block truncate">text-embedding-004</span>
                <span className="text-[11px] text-slate-500">768-dim vectors</span>
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between shadow-sm">
              <span className="text-xs font-medium text-slate-400">LLM Generator</span>
              <div className="mt-2">
                <span className="text-sm font-semibold text-violet-300 font-mono block truncate">gemini-2.5-flash</span>
                <span className="text-[11px] text-slate-500">Grounded Citations</span>
              </div>
            </div>
          </div>
        </section>

        {/* Dashboard Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Ingestion & Search Inspector (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* Card 1: Document Upload */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
                  <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 0115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <span>Document Ingestion</span>
                </h2>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">.pdf / .txt</span>
              </div>

              <form onSubmit={handleIngestSubmit} className="space-y-4">
                {/* Drag & Drop Upload Zone */}
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors ${
                    isDragOver
                      ? 'border-indigo-500 bg-indigo-950/20'
                      : selectedFile
                      ? 'border-emerald-600/60 bg-emerald-950/10'
                      : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'
                  }`}
                >
                  <input
                    type="file"
                    id="file-upload"
                    accept=".pdf,.txt"
                    disabled={isIngesting}
                    onChange={handleFileChange}
                    className="sr-only"
                  />

                  {!selectedFile ? (
                    <label htmlFor="file-upload" className="cursor-pointer space-y-2 block focus-within:outline-none">
                      <div className="h-10 w-10 mx-auto rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400">
                        📄
                      </div>
                      <div className="text-sm font-medium text-slate-300">
                        <span className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2">Click to select</span> or drag & drop file
                      </div>
                      <p className="text-xs text-slate-500">PDF or TXT document (up to 10 MB)</p>
                    </label>
                  ) : (
                    <div className="flex items-center justify-between bg-slate-900 border border-emerald-800/50 rounded-lg p-3">
                      <div className="flex items-center space-x-3 truncate">
                        <span className="text-xl">📄</span>
                        <div className="text-left truncate">
                          <p className="text-sm font-medium text-emerald-300 truncate">{selectedFile.name}</p>
                          <p className="text-xs text-slate-400">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                        </div>
                      </div>
                      {!isIngesting && (
                        <button
                          type="button"
                          onClick={handleClearFile}
                          className="text-slate-400 hover:text-rose-400 p-1 transition-colors focus:outline-none rounded"
                          title="Remove file"
                          aria-label="Remove selected file"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* Chunk Parameters */}
                <div className="grid grid-cols-2 gap-3 pt-1">
                  <div>
                    <label htmlFor="chunk-size" className="block text-xs font-medium text-slate-400 mb-1">
                      Chunk Size (chars)
                    </label>
                    <input
                      type="number"
                      id="chunk-size"
                      disabled={isIngesting}
                      value={chunkSize}
                      onChange={(e) => setChunkSize(Number(e.target.value))}
                      min={100}
                      max={2000}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 disabled:opacity-50"
                    />
                  </div>

                  <div>
                    <label htmlFor="chunk-overlap" className="block text-xs font-medium text-slate-400 mb-1">
                      Chunk Overlap (chars)
                    </label>
                    <input
                      type="number"
                      id="chunk-overlap"
                      disabled={isIngesting}
                      value={chunkOverlap}
                      onChange={(e) => setChunkOverlap(Number(e.target.value))}
                      min={0}
                      max={500}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 disabled:opacity-50"
                    />
                  </div>
                </div>

                {/* Overwrite Toggle */}
                <div className="flex items-center space-x-2 pt-1">
                  <input
                    type="checkbox"
                    id="overwrite-toggle"
                    disabled={isIngesting}
                    checked={overwriteDuplicates}
                    onChange={(e) => setOverwriteDuplicates(e.target.checked)}
                    className="h-4 w-4 rounded border-slate-800 bg-slate-950 text-indigo-600 focus:ring-indigo-500/50"
                  />
                  <label htmlFor="overwrite-toggle" className="text-xs text-slate-400 cursor-pointer">
                    Overwrite duplicate document chunks
                  </label>
                </div>

                {/* Ingest Button */}
                <button
                  type="submit"
                  disabled={!selectedFile || isIngesting}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 px-4 rounded-xl transition-all shadow-md shadow-indigo-600/20 focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-40 disabled:cursor-not-allowed text-sm flex items-center justify-center space-x-2"
                >
                  {isIngesting ? (
                    <>
                      <span className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                      <span>Processing & Embedding...</span>
                    </>
                  ) : (
                    <span>Ingest & Process Document</span>
                  )}
                </button>

                {/* Ingest Result Output */}
                {ingestResponse && (
                  <div className="mt-3 p-3 rounded-lg border border-emerald-800/60 bg-emerald-950/40 text-xs space-y-1">
                    <div className="flex items-center justify-between text-emerald-300 font-medium">
                      <span>Status: {ingestResponse.status}</span>
                      <span>{ingestResponse.total_chunks} chunks</span>
                    </div>
                    <p className="text-slate-300">
                      Total Pages: {ingestResponse.total_pages} | Chunks Added: {ingestResponse.added_count}
                    </p>
                  </div>
                )}
              </form>
            </div>

            {/* Card 2: Semantic Vector Search Panel */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
                  <svg className="w-5 h-5 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <span>Semantic Vector Search</span>
                </h2>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">POST /search</span>
              </div>

              <form onSubmit={handleSearchSubmit} className="space-y-4">
                <div>
                  <label htmlFor="search-query" className="block text-xs font-medium text-slate-400 mb-1">
                    Query String
                  </label>
                  <input
                    type="text"
                    id="search-query"
                    disabled={isSearching}
                    placeholder="Enter search query (e.g. 'vector retrieval')"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500 disabled:opacity-50"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <label htmlFor="search-top-k" className="text-xs text-slate-400">
                      Top K Results:
                    </label>
                    <select
                      id="search-top-k"
                      disabled={isSearching}
                      value={searchTopK}
                      onChange={(e) => setSearchTopK(Number(e.target.value))}
                      className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-violet-500 disabled:opacity-50"
                    >
                      <option value={2}>2</option>
                      <option value={4}>4</option>
                      <option value={8}>8</option>
                    </select>
                  </div>

                  <button
                    type="submit"
                    disabled={!searchQuery.trim() || isSearching}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium py-1.5 px-4 rounded-lg text-xs transition-colors border border-slate-700 focus:outline-none focus:ring-2 focus:ring-violet-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-1.5"
                  >
                    {isSearching ? (
                      <>
                        <span className="h-3 w-3 border-2 border-slate-300 border-t-transparent rounded-full animate-spin"></span>
                        <span>Searching...</span>
                      </>
                    ) : (
                      <span>Execute Search</span>
                    )}
                  </button>
                </div>
              </form>

              {/* Vector Search Results Display */}
              {searchResults && (
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Found {searchResults.results_count} results</span>
                    <span>Query: "{searchResults.query}"</span>
                  </div>

                  {searchResults.results.length === 0 ? (
                    <p className="text-xs text-slate-500 italic p-3 text-center bg-slate-950/60 rounded-lg border border-slate-800">
                      No matching vector chunks found in collection.
                    </p>
                  ) : (
                    <div className="space-y-2.5 max-h-60 overflow-y-auto pr-1">
                      {searchResults.results.map((item) => (
                        <div
                          key={item.chunk_id}
                          className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-3 text-xs space-y-1.5 hover:border-slate-700 transition-colors"
                        >
                          <div className="flex items-center justify-between font-mono text-[11px] text-violet-300">
                            <span className="truncate">{item.source_filename} (p.{item.page_number})</span>
                            <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                              dist: {item.distance.toFixed(4)}
                            </span>
                          </div>
                          <p className="text-slate-300 line-clamp-3 leading-relaxed">{item.text}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Q&A Console (7 Cols) */}
          <div className="lg:col-span-7">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-5 h-full flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                  <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
                    <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                    <span>Grounded Q&A Console</span>
                  </h2>
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-violet-950/80 text-violet-300 border border-violet-800/60 font-mono">
                    POST /query
                  </span>
                </div>

                {/* Question Input Form */}
                <form onSubmit={handleQuerySubmit} className="space-y-3">
                  <div>
                    <label htmlFor="question-input" className="block text-xs font-medium text-slate-400 mb-1">
                      User Question
                    </label>
                    <textarea
                      id="question-input"
                      rows={3}
                      disabled={isQuerying}
                      placeholder="Ask a question about your ingested documents (e.g. 'What is Agentic RAG and how does retrieval work?')..."
                      value={questionQuery}
                      onChange={(e) => setQuestionQuery(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 resize-none disabled:opacity-50"
                    />
                  </div>

                  <div className="flex items-center justify-between pt-1">
                    <div className="flex items-center space-x-2">
                      <label htmlFor="query-top-k" className="text-xs text-slate-400">
                        Context Chunks (Top K):
                      </label>
                      <select
                        id="query-top-k"
                        disabled={isQuerying}
                        value={queryTopK}
                        onChange={(e) => setQueryTopK(Number(e.target.value))}
                        className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
                      >
                        <option value={2}>2</option>
                        <option value={4}>4</option>
                        <option value={6}>6</option>
                        <option value={8}>8</option>
                      </select>
                    </div>

                    <button
                      type="submit"
                      disabled={!questionQuery.trim() || isQuerying}
                      className="bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-medium py-2 px-5 rounded-xl text-sm transition-all shadow-md shadow-indigo-600/20 focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-2"
                    >
                      {isQuerying ? (
                        <>
                          <span className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                          <span>Retrieving & Generating...</span>
                        </>
                      ) : (
                        <>
                          <span>Ask Question</span>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                          </svg>
                        </>
                      )}
                    </button>
                  </div>
                </form>

                {/* Real Answer Display Area */}
                <div className="mt-6 border border-slate-800 rounded-xl bg-slate-950/60 p-4 space-y-4">
                  <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800/80 pb-2">
                    <span className="font-semibold text-slate-300">Generated Grounded Answer</span>
                    {queryResponse && (
                      <span className="font-mono text-indigo-300">{queryResponse.model_name}</span>
                    )}
                  </div>

                  {!queryResponse ? (
                    <p className="text-sm text-slate-500 leading-relaxed italic text-center py-6">
                      No question answered yet. Submit a question above to retrieve context and generate grounded answers.
                    </p>
                  ) : (
                    <div className="space-y-4">
                      {/* Context Sufficiency Badge */}
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400">
                          Question: <strong className="text-slate-200">"{queryResponse.question}"</strong>
                        </span>
                        <span
                          className={`text-[11px] px-2.5 py-0.5 rounded-full border font-medium ${
                            queryResponse.sufficient_context
                              ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                              : 'bg-amber-950 text-amber-300 border-amber-800'
                          }`}
                        >
                          {queryResponse.sufficient_context ? '✓ Sufficient Context' : '⚠️ Context Insufficient'}
                        </span>
                      </div>

                      {/* Generated Answer Body */}
                      <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-4 text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                        {queryResponse.answer}
                      </div>

                      {/* Source References List */}
                      {queryResponse.sources && queryResponse.sources.length > 0 && (
                        <div className="space-y-2 pt-2 border-t border-slate-800/80">
                          <span className="text-xs font-semibold text-slate-300 block">
                            Source References ({queryResponse.sources.length}):
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {queryResponse.sources.map((src) => (
                              <div
                                key={src.chunk_id}
                                className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono text-indigo-300 flex items-center space-x-1.5"
                              >
                                <span className="font-bold text-violet-400">{src.source_label}</span>
                                <span>{src.source_filename} (p.{src.page_number})</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
                <span>FastAPI Endpoint: <code className="text-slate-400">POST /api/v1/query</code></span>
                <span>Grounding Verification: Active</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div>Agentic-RAG Pipeline • Google Gemini & ChromaDB</div>
          <div>Phase 3B: Native API Integration Complete</div>
        </div>
      </footer>
    </div>
  )
}
