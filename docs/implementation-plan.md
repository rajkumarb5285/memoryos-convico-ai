# MEMORYOS Implementation Plan

## 1. Objective
Build a production-grade, leak-proof, multi-lingual conversational memory engine adhering to the 100-point Convico AI specification.

## 2. Phased Architecture & Execution Roadmap

### Phase A: Environment & Foundation
- Setup Docker Compose services for PostgreSQL 16 (with `pgvector` 0.8+) and Redis 7.
- Configure asynchronous SQLAlchemy 2.x engine and connection pooling.
- Configure Alembic migrations for relational entities (`users`, `characters`, `conversations`, `messages`, `memories`).
- Setup Pydantic v2 schemas and centralized settings with `.env.example`.

### Phase B: Security & Authentication Engine
- Implement cryptographic token authentication and `Principal` context dependency.
- Enforce strict tenancy: `owner_user_id` and `character_id` must be bound to every database query and cache namespace.
- Zero-trust IDOR prevention: path user parameter must match authenticated token; foreign character or conversation returns 404/403 with no information leakage.

### Phase C: Neural Memory Extraction (Module A)
- Abstract `MemoryExtractor` interface.
- Implement production-grade deterministic extractor supporting:
  - English and Romanized Hinglish (e.g. "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai", "Main Indore mein rehta hoon").
  - Multi-fact extraction within single utterances.
  - Polarity detection (positive, negative, neutral).
  - Domain and entity classification (identity, location, occupation, preference, activity, future plan).
  - Assertion filtering (rejecting questions like "Do you know someone at Google?").
- Pluggable LLM extractor adapter with fallbacks and strict structured output validation.

### Phase D: Memory Lifecycle & Temporal Contradiction Engine (Module B)
- Entity and predicate normalization.
- Semantic duplicate detection and deduplication.
- Contradiction resolution preserving historical truth:
  - Temporal states: `historical` vs `current`.
  - `valid_from` and `valid_until` timestamping.
  - Provenance tracking with `source_message_id` and `supersedes_memory_id`.
- Atomic multi-fact batch updates using database transactions.
- Soft deletion and tombstone tracking.

### Phase E: Hybrid RAG Retrieval Engine (Module C)
- **Lane A**: Structured metadata and domain filtering.
- **Lane B**: Lexical search with PostgreSQL Full-Text Search and Hinglish transliteration handling.
- **Lane C**: Dense vector retrieval using `BAAI/bge-small-en-v1.5` / `all-MiniLM-L6-v2` embeddings (dim 384) with `pgvector` cosine similarity.
- **Reciprocal Rank Fusion (RRF)**: Deterministic ranking combiner across lanes.
- **Multi-factor Reranker**: Scoring relevance, confidence, domain match, and temporal suitability (historical queries favor historical facts).
- **Conversational Follow-up Rewriter**: Resolves follow-ups like "And where before that?" from short-term context.
- **Hallucination Prevention**: Explicit `insufficient_evidence` response when confidence falls below threshold.

### Phase F: Redis Caching & Short-Term Context
- Scoped cache key strategy: `memory_context:{owner_id}:{character_id}:{conversation_id}`.
- Sliding window TTL expiration.
- Instant invalidation on memory updates or deletions.
- Resilience: System gracefully degrades to PostgreSQL direct query if Redis is unreachable.

### Phase G: API & Integration Endpoints (Module E)
- `POST /memory/ingest`
- `POST /memory/retrieve`
- `GET /memory/{user_id}`
- `DELETE /memory/{memory_id}`
- `GET /health`

### Phase H: Automated Testing & Evaluation (Module F)
- 100+ conversation evaluation dataset (`evaluation/dataset.json`).
- Automated evaluation runner computing Precision, Recall, F1, P@5, R@5, leakage rate.
- Latency benchmarking (mean, P95).
- Comprehensive test suites: Unit, Integration, API, Security/IDOR (10 attack vectors), Failure injection.
- Generation of submission artifacts: PDF documentation, architecture diagram, CSV/JSON metrics.

