"""
LangGraph State - TypedDict for checkpointed state management
Defines the shape of data flowing through the orchestration graph
"""
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from operator import add


class PolicyGraphState(TypedDict):
    """
    State object for the policy reasoning graph
    Flows through: analyze → plan → retrieve → fuse → synthesize
    """
    
    # Input
    request_id: str
    query: str
    user_context: Dict[str, Any]
    jurisdiction: str
    max_engines: int
    trace: Dict[str, Any]
    
    # Analysis phase
    features: Optional[Dict[str, Any]]
    entities: Optional[Dict[str, List[str]]]
    facets: Optional[List[str]]
    query_type: Optional[str]
    
    # Planning phase
    plan_id: Optional[str]
    engine_scores: Optional[Dict[str, float]]
    selected_engines: Optional[List[str]]
    engine_configs: Optional[Dict[str, Dict[str, Any]]]
    routing_rationale: Optional[str]
    
    # Retrieval phase (accumulates results from parallel branches)
    retrieved_docs: Annotated[List[Dict[str, Any]], add]  # Accumulates across engines
    retrieval_metadata: Optional[Dict[str, Any]]
    
    # Fusion phase
    deduplicated_docs: Optional[List[Dict[str, Any]]]
    reranked_docs: Optional[List[Dict[str, Any]]]
    final_docs: Optional[List[Dict[str, Any]]]
    
    # Synthesis phase
    answer: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    confidence: Optional[float]
    used_engines: Optional[List[str]]
    
    # Error handling
    error: Optional[str]
    retry_count: Optional[int]


def create_initial_state(
    request_id: str,
    query: str,
    user_context: Dict[str, Any],
    jurisdiction: str,
    max_engines: int,
    trace: Dict[str, Any]
) -> PolicyGraphState:
    """Create initial state for graph execution"""
    return {
        "request_id": request_id,
        "query": query,
        "user_context": user_context,
        "jurisdiction": jurisdiction,
        "max_engines": max_engines,
        "trace": trace,
        
        # Initialize optional fields
        "features": None,
        "entities": None,
        "facets": None,
        "query_type": None,
        
        "plan_id": None,
        "engine_scores": None,
        "selected_engines": None,
        "engine_configs": None,
        "routing_rationale": None,
        
        "retrieved_docs": [],
        "retrieval_metadata": None,
        
        "deduplicated_docs": None,
        "reranked_docs": None,
        "final_docs": None,
        
        "answer": None,
        "citations": None,
        "confidence": None,
        "used_engines": None,
        
        "error": None,
        "retry_count": 0
    }


def is_error_state(state: PolicyGraphState) -> bool:
    """Check if state contains an error"""
    return state.get("error") is not None


def get_checkpoint_fields() -> List[str]:
    """Return fields to include in checkpoints for recovery"""
    return [
        "request_id",
        "query",
        "plan_id",
        "selected_engines",
        "retrieved_docs",
        "final_docs",
        "answer",
        "error"
    ]