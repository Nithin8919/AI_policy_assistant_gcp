"""
AP Policy Reasoning System - FastAPI Entry Point
Orchestrates multi-engine RAG with LangGraph for Andhra Pradesh government data
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import uuid

from orchestrator.graph import build_policy_graph
from utils.logging import setup_logger, log_request
from utils.tracing import create_trace_context
from config import load_config

# Initialize
app = FastAPI(
    title="AP Policy Reasoning API",
    description="Multi-engine RAG system for Andhra Pradesh government policy queries",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = load_config()
logger = setup_logger()
policy_graph = None


@app.on_event("startup")
async def startup_event():
    """Initialize LangGraph and connections on startup"""
    global policy_graph
    logger.info("🚀 Starting AP Policy Reasoning System")
    try:
        policy_graph = build_policy_graph(config)
        logger.info("✅ LangGraph orchestrator initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Policy question")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    jurisdiction: Optional[str] = Field("Andhra Pradesh", description="Geographic scope")
    max_engines: Optional[int] = Field(3, ge=1, le=5, description="Max engines to query")


class Citation(BaseModel):
    vertical: str
    doc_id: str
    locator: str  # section, page, para
    snippet: str
    score: float
    source_date: Optional[str] = None
    source_url: Optional[str] = None


class PolicyAnswer(BaseModel):
    request_id: str
    query: str
    answer: str
    citations: List[Citation]
    used_engines: List[str]
    confidence: float
    plan_id: str
    timestamp: str
    processing_time_ms: int


class PlanDetails(BaseModel):
    plan_id: str
    query: str
    entities: Dict[str, Any]
    engine_scores: Dict[str, float]
    selected_engines: List[str]
    routing_rationale: str
    created_at: str


# API Endpoints
@app.post("/answer", response_model=PolicyAnswer)
async def answer_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint: route query through multi-engine RAG pipeline
    Returns grounded answer with citations
    """
    request_id = str(uuid.uuid4())
    trace_ctx = create_trace_context(request_id)
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"📥 Request {request_id}: {request.query[:100]}")
        
        # Build initial state
        initial_state = {
            "request_id": request_id,
            "query": request.query,
            "user_context": request.user_context or {},
            "jurisdiction": request.jurisdiction,
            "max_engines": request.max_engines,
            "trace": trace_ctx
        }
        
        # Execute LangGraph
        result = await policy_graph.graph.ainvoke(initial_state)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Build response
        response = PolicyAnswer(
            request_id=request_id,
            query=request.query,
            answer=result["answer"],
            citations=result["citations"],
            used_engines=result["used_engines"],
            confidence=result["confidence"],
            plan_id=result["plan_id"],
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_ms
        )
        
        # Log async
        background_tasks.add_task(log_request, request_id, request.query, response, trace_ctx)
        
        logger.info(f"✅ Request {request_id} completed in {processing_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Request {request_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/plan/{plan_id}", response_model=PlanDetails)
async def get_plan(plan_id: str):
    """
    Retrieve execution plan details for audit/debugging
    Shows which engines were scored and selected for a query
    """
    try:
        # Fetch from checkpointed state (implementation depends on your store)
        # For now, return from in-memory cache or database
        plan = policy_graph.get_plan(plan_id)
        
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        
        return PlanDetails(**plan)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve plan {plan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(
    request_id: str,
    rating: int = Field(..., ge=1, le=5),
    comments: Optional[str] = None
):
    """
    Collect user feedback for RLHF and system improvement
    """
    try:
        feedback = {
            "request_id": request_id,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store feedback (implement based on your preference: BigQuery, Firestore, etc.)
        logger.info(f"📝 Feedback received for {request_id}: {rating}/5")
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """System health and readiness check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "graph_ready": policy_graph is not None
    }


@app.get("/engines")
async def list_engines():
    """List available RAG engines and their status"""
    return {
        "engines": list(config["engines"].keys()),
        "total": len(config["engines"]),
        "config": {k: {"facets": v.get("facets", []), "weight": v["weight"]} 
                   for k, v in config["engines"].items()}
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )