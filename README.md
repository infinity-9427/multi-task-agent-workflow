# Automated Task Review Agent

AI-powered system that reviews task requests using RAG (Retrieval-Augmented Generation) with pgvector similarity search and LLM decision making.

## Quick Start

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

API Documentation: http://localhost:8080/docs  
Health Check: http://localhost:8080/health

## Architecture

The system uses a two-stage RAG pipeline with deterministic gates:

**Retrieval Stage:**
- PostgreSQL 16 with pgvector extension
- Gemini embeddings (768 dimensions) 
- Cosine similarity search with Top-K=4
- Coverage calculation based on similarity scores

**Decision Stage:**
- Gemini 2.0 Flash for decision making
- JSON-only output format
- Citation filtering and validation
- Confidence scoring

**Gate System:**
- Coverage Gate: Blocks requests with coverage < 0.35
- Policy Gate: Requires ≥2 citations + coverage ≥0.45 for approval

## Features

- pgvector-based document retrieval
- Gemini 2.0 Flash LLM integration
- Security validation (HTML injection protection)
- Deterministic coverage calculation
- Multi-citation approval requirements
- Docker containerization
- Comprehensive test suite

## Installation

### Prerequisites
- Docker & Docker Compose
- Google AI API key (get from https://aistudio.google.com/app/apikey)

### Setup Steps
1. Clone and configure:
```bash
git clone <repository>
cd backend
cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

2. Start services:
```bash
docker compose up --build -d
```

3. Ingest documents (run once):
```bash
docker exec task-agent-api python -m rag.ingest
```

4. Verify installation:
- API Docs: http://localhost:8080/docs  
- Health Check: http://localhost:8080/health

## API Usage

### Core Endpoint
```bash
POST /review
```

### Example Requests

Simple task (typically low coverage):
```bash
curl -X POST "http://localhost:8080/review" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001",
    "details": "create login form"
  }'
```

Security task (typically higher coverage):
```bash
curl -X POST "http://localhost:8080/review" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-001", 
    "details": "implement secure user authentication with MFA, following security guidelines for session management and password policies"
  }'
```

### Response Format
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

## Testing

### Run Tests
```bash
# All tests (recommended)
make test-docker

# Quick minimal tests
make test-docker-minimal

# Simple functionality tests
make test-docker-simple

# Show all commands
make help
```

### Test Coverage
- API endpoints and validation
- RAG pipeline (retrieval, coverage, decision)
- Security validation (HTML injection protection)
- Database integration
- Error handling and edge cases

## Development

### Local Development
```bash
# Install dependencies
pip install uv && uv sync

# Start database only
docker compose up db -d

# Ingest documents
python -m rag.ingest

# Run API locally  
source .venv/bin/activate
uvicorn main:app --reload --port 8080
```

### Project Structure
```
backend/
├── rag/                 # RAG pipeline
│   ├── ingest.py           # Document ingestion  
│   └── orchestrator.py     # Main orchestration flow
├── agents/              # Specialist agents
│   ├── retriever_agent.py  # pgvector retrieval + coverage
│   └── decision_agent.py   # Gemini decision making
├── db/sql/              # Database schema
├── routes/              # API endpoints  
├── schemas/             # Pydantic models
├── data/               # Document corpus (PDFs)
├── tests/              # Test suite
└── docker-compose.yml  # Services orchestration
```



