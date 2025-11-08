# AP Policy Reasoning System

Multi-engine RAG (Retrieval-Augmented Generation) system for Andhra Pradesh government policy queries using LangGraph, Vertex AI, and FastAPI.

## Architecture

This system orchestrates retrieval from multiple Vertex AI RAG-managed engines, merges results, and synthesizes grounded answers with citations.

### Components

- **FastAPI** - REST API entrypoint
- **LangGraph** - Orchestration graph with checkpointing
- **Vertex AI RAG** - Managed vector stores for 5 verticals
- **Vertex AI Ranking API** - Cross-engine semantic reranking
- **Gemini 1.5 Pro** - Answer synthesis with citations

### RAG Engines

1. **Data Report** - ASER, budget, financial, NAS, SES, teacher data, UDISE
2. **GOs** - Government Orders and circulars
3. **Judicial** - Court judgments and case law
4. **Legal** - Acts, rules, constitution, RTE, service rules, transfer policy
5. **Schemes** - Educational schemes and welfare programs

## Project Structure

```
├─ app.py                     # FastAPI entrypoint
├─ config/
│  ├─ __init__.py            # Configuration management
│  └─ settings.yaml          # Engine configs, models, routing rules
├─ router/
│  ├─ query_analyzer.py      # NER, entity extraction, query expansion
│  ├─ engine_scorer.py       # Engine scoring and selection
│  └─ planner.py             # Execution plan creation
├─ orchestrator/
│  ├─ graph.py               # LangGraph orchestration
│  └─ state.py               # State TypedDict
├─ agents/                   # Domain-specific agents
│  ├─ legal.py
│  ├─ judicial.py
│  ├─ schemes.py
│  ├─ education.py
│  └─ data_report.py
├─ rag_clients/
│  ├─ vertex_rag.py          # Vertex RAG client
│  └─ ranking_api.py         # Ranking API wrapper
├─ fusion/
│  ├─ dedupe.py              # Deduplication
│  ├─ rerank.py              # Cross-engine reranking
│  └─ merge.py               # Result merging
├─ llm/
│  └─ synth.py               # Answer synthesis
└─ utils/
   ├─ logging.py             # Structured logging
   └─ tracing.py             # Request tracing
```

## Setup

### Prerequisites

- Python 3.11+
- Google Cloud Project with Vertex AI enabled
- Vertex AI RAG corpora configured
- GCP credentials configured

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nithin8919/AI_policy_assistant_gcp.git
cd AI_policy_assistant_gcp
```

2. Install dependencies using UV:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Configure `config/settings.yaml`:
   - Update `project.gcp_project_id` with your GCP project ID
   - Update engine RAG corpus IDs
   - Adjust routing and ranking parameters

4. Run the API:
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `POST /answer`
Main query endpoint. Returns grounded answer with citations.

**Request:**
```json
{
  "query": "What are the teacher transfer rules?",
  "jurisdiction": "Andhra Pradesh",
  "max_engines": 3
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "query": "...",
  "answer": "...",
  "citations": [...],
  "used_engines": ["legal", "gos"],
  "confidence": 0.85,
  "plan_id": "uuid",
  "timestamp": "...",
  "processing_time_ms": 1234
}
```

### `GET /plan/{plan_id}`
Retrieve execution plan details for audit/debugging.

### `POST /feedback`
Submit user feedback for RLHF.

### `GET /health`
Health check endpoint.

### `GET /engines`
List available RAG engines.

## Pipeline Flow

1. **Analyze** - Extract entities, facets, constraints, temporal info
2. **Plan** - Score engines, select top N, apply forced pairs
3. **Retrieve** - Parallel retrieval from selected engines
4. **Fuse** - Deduplicate, rerank using Vertex Ranking API
5. **Synthesize** - Generate answer with citations using Gemini

## Configuration

Key settings in `config/settings.yaml`:

- **Models**: LLM, embedding model, temperature
- **Engines**: RAG corpus IDs, facets, weights
- **Routing**: Max engines, min scores, forced pairs
- **Ranking**: Top-k per engine, final k, thresholds
- **Fusion**: Deduplication strategy, conflict resolution

## Development

### Testing

```bash
pytest
```

### Code Quality

```bash
black .
ruff check .
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

