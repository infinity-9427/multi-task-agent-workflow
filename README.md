# Automated Task Review Agent


##  Quick Start

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 2. Start services  
docker compose up --build -d

# 3. Ingest documents (run once)
docker exec task-agent-api python -m rag.ingest

# 4. Test API
curl -X POST "http://localhost:8080/review" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "TEST-001", "details": "implement secure user authentication system"}'
```

**API Documentation**: http://localhost:8080/docs  
**Health Check**: http://localhost:8080/health

## 🏗️ Architecture

**pgvector-Only RAG System**:
-  **Retriever Agent**: pgvector cosine similarity search with Top-K=4
-  **Decision Agent**: Gemini 1.5 Flash 2.0 with JSON-only output  
-  **Coverage Gates**: 0.35 threshold blocks insufficient context
-  **Policy Gates**: Approval requires ≥2 citations + coverage ≥0.45
-  **Deterministic Flow**: retrieve → coverage gate → decide → policy gate → finalize

**pgvector Storage**:
-  **PostgreSQL 16**: Single source of truth with pgvector extension
-  **Gemini Embeddings**: gemini-embedding-001 model (768 dimensions)
-  **Chunked Content**: ~900 chars with 150-200 overlap
-  **ANN Index**: ivfflat cosine similarity for fast search

##  Features

- pgvector-only architecture (no FAISS complexity)
- Gemini 2.0Flash  for reliable decisions
- Security validation (HTML injection protection)
- PostgreSQL 16 with pgvector extension
- Fast similarity search
- Deterministic retrieval and coverage calculation
- Envelope response format with detailed metadata
- Multi-citation requirements for approval (≥2)
- Docker containerization with health checks

## 📦 Installation

### Prerequisites
- Docker & Docker Compose
- Google AI API key ([Get one here](https://aistudio.google.com/app/apikey))

### Setup Steps
1. **Clone & Configure**:
```bash
git clone <multi-task-agent-workflow>
cd backend
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

2. **Start Services**:
```bash
docker compose up --build -d
```

3. **Ingest Documents** (run once):
```bash
docker exec task-agent-api python -m rag.ingest
```

4. **Verify Installation**:
- 📖 API Docs: http://localhost:8080/docs  
- ✅ Health Check: http://localhost:8080/health

## 🔌 API Usage

### Core Endpoint
```bash
POST /review
```

### Example Requests

**Simple Task** (Low Coverage → Insufficient Context):
```bash
curl -X POST "http://localhost:8080/review" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "details": "create login form"
  }'
```

**Security Task** (Higher Coverage → Potential Approval):
```bash
curl -X POST "http://localhost:8080/review" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001", 
    "details": "implement secure user authentication with MFA, following security guidelines for session management and password policies"
  }'
```

**Response Format** (Envelope):
```json
{
  "message": "review completed",
  "data": {
    "task_id": "task-001",
    "decision": "approve",
    "rationale": "Task meets security requirements based on doc:security#chunk:12",
    "citations": ["doc:security#chunk:12", "doc:auth#chunk:5"],
    "retrieved_doc_ids": [1, 3],
    "coverage": 0.72,
    "latency_ms": 287,
    "required_actions": [],
    "confidence": 0.85
  }
}
```

### Decision Types
- `approve` - Task approved (coverage ≥0.45, ≥2 citations)
- `reject` - Task needs improvements (actionable feedback provided)  

## 🧪 Testing & Validation

### Run Tests
```bash
# Full test suite
pytest tests/ -v

# Quick minimal test
pytest tests/test_orchestration.py::TestOrchestration::test_coverage_gate_blocks_low_coverage -v
```

**Test Coverage**:
- ✅ Orchestration flow (coverage gates, policy gates)
- ✅ Retriever agent (pgvector queries, coverage calculation)  
- ✅ Decision agent (JSON parsing, citation filtering)
- ✅ API endpoints (health check, review endpoint)
- ✅ Security validation (HTML injection protection)

## 🔧 Development

### Local Development
```bash
# Install dependencies
pip install uv && uv sync

# Start database
docker compose up db -d

# Ingest documents
python -m rag.ingest

# Run locally  
source .venv/bin/activate
uvicorn main:app --reload --port 8080
```

### Project Structure
```
backend/
├── rag/                 # RAG pipeline
│   ├── ingest.py           # Offline document ingestion  
│   └── orchestrator.py     # Main orchestration flow
├── agents/              # Specialist agents
│   ├── retriever_agent.py  # pgvector retrieval + coverage
│   └── decision_agent.py   # Gemini 1.5 decision making
├── db/sql/              # Database schema
├── routes/              # API endpoints  
├── schemas/             # Pydantic models
├── data/               # Document corpus (PDFs)
├── tests/              # Test suite
└── docker-compose.yml  # Services orchestration
```



