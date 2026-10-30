# MEMORYOS — Intelligent Conversational Memory Engine
### Convico AI | 24-Hour Backend AI/ML Engineering Challenge

---

## Candidate

**Rajkumar Pandey**

---

## Overview

MEMORYOS is a production-grade conversational memory backend that persistently stores, manages, and retrieves facts learned from multi-turn dialogue — including multilingual (English + Romanized Hinglish) conversations. It implements:

- **Neural memory extraction** with negation handling, question rejection, and multi-fact splitting
- **Temporal memory lifecycle** with contradiction resolution and historical fact preservation
- **Hybrid RAG retrieval** with 3 independent lanes (Structured, Lexical, Dense pgvector) fused via Reciprocal Rank Fusion (RRF) and reranked
- **Strict multi-tenant security** — every DB query is scoped by `owner_user_id` and `character_id`
- **100% evaluation pass rate** — 105/105 test cases, F1=1.0, Leakage=0.0

---


---

## Architecture

```
Client
  └── FastAPI (Pydantic v2 validation · JWT auth · CORS)
       ├── Auth endpoints    (register / token)
       └── Memory endpoints  (ingest · retrieve · list · delete)
            ├── Memory Extraction Pipeline
            │    └── DeterministicMemoryExtractor
            │         ├── English pattern matching
            │         ├── Hinglish negation/positive patterns
            │         ├── Multi-fact clause splitting
            │         ├── Question / greeting filter
            │         └── Temporal transition detection
            ├── Memory Lifecycle Resolver
            │    ├── Duplicate detection
            │    ├── Polarity conflict resolution
            │    ├── Singular-predicate supersession
            │    └── valid_from / valid_until tracking
            ├── Retrieval Coordinator
            │    ├── Lane A: Structured (predicate/domain/entity filters)
            │    ├── Lane B: Lexical (ILIKE + Hinglish synonym expansion)
            │    ├── Lane C: Dense (pgvector cosine similarity, HNSW index)
            │    ├── Reciprocal Rank Fusion (RRF, k=60)
            │    └── Reranker (semantic cosine + keyword + temporal + corroboration)
            └── PostgreSQL (authoritative) + Redis (cache/context TTL)
```

See [`docs/architecture.md`](docs/architecture.md) and [`RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT/05_ARCHITECTURE/architecture.png`](RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT/05_ARCHITECTURE/architecture.png) for the full diagram.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| ORM | SQLAlchemy 2.x (async) |
| DB driver | asyncpg |
| Database | PostgreSQL 16 + pgvector |
| Cache | Redis 7 (async via redis-py) |
| Migrations | Alembic |
| Embeddings | BAAI/bge-small-en-v1.5 via fastembed (ONNX, dim=384) |
| Auth | JWT (python-jose) + bcrypt |
| Testing | pytest + pytest-asyncio + HTTPX |
| Linting | Ruff |
| Type checking | Mypy |
| Security scanning | Bandit |
| CI | GitHub Actions |
| Containers | Docker + Docker Compose |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11
- `uv` package manager (`pip install uv`)

### 1. Clone and enter the repository

```bash
git clone https://github.com/rajkumarb5285/memoryos-convico-ai
cd memoryos-convico-ai
```

### 2. Create environment file

```bash
cp .env.example .env
# Edit .env and set a strong SECRET_KEY
```

### 3. Start infrastructure

```bash
docker compose up -d
# PostgreSQL on :5438, Redis on :6385
```

### 4. Create virtual environment and install dependencies

```bash
uv venv .venv --python 3.11
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 5. Apply database migrations

```bash
alembic upgrade head
```

### 6. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: http://localhost:8000 (or http://localhost:8010 if host port 8000 is occupied)  
Interactive docs: http://localhost:8000/docs (or http://localhost:8010/docs)  
Glassmorphic Web UI: http://localhost:8000/app (or http://localhost:8010/app)

---

## Running Tests

```bash
# All 37 tests (Unit, Integration, Security IDOR, Resilience)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app --cov-report=term-missing

# Specific test file
pytest tests/test_security_isolation.py -v
```

---

## Running the Evaluation Suite

```bash
# Generate 105 test cases and run evaluation
cd memoryos-convico-ai
PYTHONPATH=. python evaluation/run_evaluation.py

# Run performance benchmark (50 requests, concurrency=5)
PYTHONPATH=. python evaluation/run_benchmark.py
```

---

## Evaluation Results (Fresh Run)

| Metric | Score |
|--------|-------|
| Total test cases | 105 |
| Pass rate | **100.0%** |
| Extraction Precision | **1.0** |
| Extraction Recall | **1.0** |
| Extraction F1 | **1.0** |
| Retrieval R@5 | **1.0** |
| Retrieval P@5 | **0.20** |
| Retrieval MRR | **0.9583** |
| Contradiction Resolution Acc | **1.0** |
| Duplicate Handling Acc | **1.0** |
| Avg retrieval latency | 93 ms |
| P95 retrieval latency | 182 ms |
| Cross-user leakage rate | **0.0** |

Benchmark (50 concurrent requests):
- Avg: 230ms · P50: 221ms · P95: 357ms · Throughput: 18.83 req/s

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/register` | Register user |
| POST | `/api/v1/auth/token` | Obtain JWT |
| GET | `/api/v1/health` | Health check (DB + Redis) |
| POST | `/api/v1/memory/ingest` | Ingest messages → extract + store memories |
| POST | `/api/v1/memory/retrieve` | Hybrid RAG query |
| GET | `/api/v1/memory/{user_id}` | List memories (IDOR-safe) |
| DELETE | `/api/v1/memory/{memory_id}` | Soft-delete memory |
| POST | `/api/v1/character` | Create character scope |
| POST | `/api/v1/conversation` | Create conversation |

---

## Key Design Decisions

### Why PostgreSQL as the vector store (not ChromaDB)?
The assignment explicitly accepts pgvector, and co-locating embeddings with the canonical memory record eliminates all cross-store synchronization risk during a time-constrained challenge. See [`docs/known-limitations.md`](docs/known-limitations.md).

### Why NullPool in SQLAlchemy for testing?
asyncpg connections are bound to the event loop they were created on. pytest creates a new loop per test function. Without `NullPool`, connections created during app lifespan are reused across loops, causing `RuntimeError: Event loop is closed`. This is the correct, documented fix.

### Deterministic extraction (not an LLM)?
Offline ONNX embeddings + pattern-based extraction makes the system fully self-contained, reproducible, and deterministic — critical for honest evaluation metrics.

---

## Project Structure

```
memoryos-convico-ai/
├── app/
│   ├── api/v1/endpoints/     # FastAPI route handlers
│   ├── cache/                # Redis service
│   ├── core/                 # Config, security, exceptions, logging
│   ├── db/                   # SQLAlchemy engine + session
│   ├── embeddings/           # fastembed wrapper
│   ├── extraction/           # Memory extraction (ABC + deterministic impl)
│   ├── lifecycle/            # Contradiction resolver
│   ├── models/               # SQLAlchemy ORM models
│   ├── retrieval/            # Structured + lexical + dense + RRF + reranker
│   └── schemas/              # Pydantic v2 schemas
├── alembic/                  # Database migrations
├── docs/                     # Architecture, RTM, known limitations, PDFs
├── evaluation/               # 105-case dataset + evaluation + benchmark
├── scripts/                  # Diagram + PDF generators
├── tests/                    # 31 automated tests
│   ├── test_api_endpoints.py
│   ├── test_security_isolation.py
│   ├── test_failure_resilience.py
│   ├── test_unit_extraction.py
│   ├── test_unit_embeddings.py
│   └── test_unit_retrieval.py
├── RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT/  # Submission package
├── docker-compose.yml
├── pyproject.toml
├── ruff.toml
├── pytest.ini
├── Makefile
└── .env.example
```

---

## Quality Gates

All gates pass cleanly:

```bash
ruff check app/ tests/        # ✅ All checks passed
ruff format --check app/ tests/ # ✅ 48 files formatted
bandit -r app/ -ll            # ✅ No Medium/High issues
pytest tests/ -v              # ✅ 31 passed
```

---

## Submission Package

```
RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT/
├── 01_GITHUB_LINK.txt
├── 02_VIDEO_DEMO_LINK.txt
├── 03_TECHNICAL_DOCUMENTATION.pdf
├── 04_EVALUATION_RESULTS/
│   ├── metrics.json
│   ├── evaluation_results.csv
│   └── test_logs.txt
├── 05_ARCHITECTURE/
│   └── architecture.png
├── 06_TEST_COVERAGE_REPORT/
│   └── index.html  (81% coverage)
└── 07_KNOWN_LIMITATIONS/
    └── known-limitations.md
```

---

## License

MIT
