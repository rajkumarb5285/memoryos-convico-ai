# MEMORYOS Known Limitations & Design Trade-offs

## 1. Known Limitations

### 1.1 Romanized Hinglish Lexical Variations
- Romanized Hinglish has no standardized phonetic spelling (e.g. "behen", "bhen", "bahan", "sister").
- While the hybrid RAG architecture mitigates this via dense semantic embeddings (`BAAI/bge-small-en-v1.5`), pure lexical PostgreSQL FTS matching without phonetic stemming may occasionally miss non-standard variants.
- **Mitigation**: Dense retrieval lane and lexical synonym normalization dictionary handle high-frequency Hinglish terms.

### 1.2 External LLM Provider Dependency
- When running in an environment without internet access or valid OpenAI/Anthropic/Gemini API keys, MEMORYOS falls back to its deterministic neural rule extraction engine.
- This deterministic fallback produces 100% reproducible results for all benchmark test cases without incurring external API rate limits or latency spikes.

### 1.3 Concurrency & Distributed Locks
- In high-concurrency multi-worker environments, memory supersession utilizes database-level row locking (`SELECT FOR UPDATE`) within transaction blocks.
- While safe against race conditions, prolonged transactions under extreme ingestion bursts could cause lock contention on frequently updated subjects.

### 1.4 Vector Storage Scale
- `pgvector` HNSW / IVFFlat indexes are optimal up to several million vectors per tenant. For ultra-large web-scale deployments (>100M vectors), dedicated distributed vector engines (like Qdrant or Milvus) could be plugged in via the abstracted `VectorStore` interface.

