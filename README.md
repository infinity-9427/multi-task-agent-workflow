# Automated Task Review Agent

Microservice for automated task review and approval using RAG pipeline and multi-agent architecture.

## Stack
- **FastAPI** - REST API
- **PostgreSQL + pgvector** - Vector database
- **LangGraph** - Multi-agent orchestration
- **Docker** - Containerization

## Features
- RAG-based document retrieval
- Multi-agent decision pipeline
- Vector similarity search
- RESTful API endpoints

## Setup

### Prerequisites
- Docker and Docker Compose

### Environment Configuration

1. **Create .env file** in project root:
```bash
# PostgreSQL Database Configuration
POSTGRES_DB=task_agent_db
POSTGRES_USER=task_agent_user
POSTGRES_PASSWORD=SecurePassword123!

# API Configuration
PORT=8080

# Database Connection (for application use)
DATABASE_URL=postgresql://task_agent_user:SecurePassword123!@postgres:5432/task_agent_db
```

2. **Start services**:
```bash
cd backend
docker compose up -d
```

### Verify Installation
- API: http://localhost:8080/docs
- Health: http://localhost:8080/api/v1/

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/` | Welcome message |
| POST | `/api/v1/review` | Review endpoint |
| GET | `/docs` | Interactive API documentation |

## Development

### Local Development
```bash
# Install dependencies
pip install uv && uv sync

# Run locally
source .venv/bin/activate
uvicorn main:app --reload --port 8080
```

### Project Structure
```
backend/
├── main.py                 # FastAPI app
├── routes/                 # API endpoints
├── documents/              # Document corpus
├── Dockerfile             # Container config
└── docker-compose.yml     # Services orchestration
```

### Testing
```bash
pytest                     # Run tests
pytest --cov=.            # With coverage
```

## Architecture

The system implements a multi-agent RAG pipeline:

1. **Document Ingestion** → Vector embeddings in PostgreSQL
2. **Retrieval Agent** → Finds relevant documents via similarity search  
3. **Decision Agent** → Makes approval/rejection decision
4. **API Response** → Returns structured decision with reasoning

