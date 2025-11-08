"""
LangGraph Orchestrator - Main graph for multi-engine RAG pipeline
Defines nodes, edges, parallel branches, and checkpointing
"""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import asyncio

# Try to import checkpoint saver, fallback to None if not available
try:
    from langgraph.checkpoint.memory import MemorySaver
    CHECKPOINT_AVAILABLE = True
except ImportError:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        CHECKPOINT_AVAILABLE = True
    except ImportError:
        CHECKPOINT_AVAILABLE = False
        MemorySaver = None

from orchestrator.state import PolicyGraphState, is_error_state
from router.planner import QueryPlanner
from rag_clients.vertex_rag import VertexRAGClient
from fusion.dedupe import deduplicate_docs
from fusion.rerank import rerank_docs
from fusion.merge import merge_and_trim
from llm.synth import synthesize_answer
from utils.logging import get_logger

logger = get_logger()


class PolicyGraphOrchestrator:
    """Orchestrates the multi-engine RAG pipeline using LangGraph"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.planner = QueryPlanner()
        self.rag_client = VertexRAGClient(config)
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph with nodes and edges
        
        Flow: analyze → plan → [retrieve_*] → fuse → synthesize
        """
        # Initialize graph with state schema
        workflow = StateGraph(PolicyGraphState)
        
        # Add nodes
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("fuse", self._fuse_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Dynamic retrieval nodes will be added per query
        # We'll handle parallel retrieval within the plan node
        
        # Define edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "plan")
        workflow.add_conditional_edges(
            "plan",
            self._should_retrieve,
            {
                "retrieve": "retrieve_all",  # Parallel retrieval
                "error": END
            }
        )
        workflow.add_node("retrieve_all", self._retrieve_all_node)
        workflow.add_edge("retrieve_all", "fuse")
        workflow.add_edge("fuse", "synthesize")
        workflow.add_edge("synthesize", END)
        
        # Compile with checkpointing if available
        if CHECKPOINT_AVAILABLE and MemorySaver:
            try:
                memory = MemorySaver()
                app = workflow.compile(checkpointer=memory)
            except Exception:
                # Fallback to no checkpointing
                app = workflow.compile()
        else:
            # No checkpointing available
            app = workflow.compile()
        
        return app
    
    async def _analyze_node(self, state: PolicyGraphState) -> PolicyGraphState:
        """Node 1: Analyze query"""
        logger.info(f"[{state['request_id']}] Analyzing query")
        
        try:
            features = self.planner.analyzer.analyze(state["query"])
            
            return {
                **state,
                "features": features,
                "entities": features["entities"],
                "facets": features["facets"],
                "query_type": features["query_type"]
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {**state, "error": f"Analysis failed: {str(e)}"}
    
    async def _plan_node(self, state: PolicyGraphState) -> PolicyGraphState:
        """Node 2: Create execution plan"""
        logger.info(f"[{state['request_id']}] Planning retrieval")
        
        try:
            plan = self.planner.create_plan(
                query=state["query"],
                max_engines=state["max_engines"],
                user_context=state["user_context"]
            )
            
            logger.info(
                f"[{state['request_id']}] Plan: "
                f"engines={plan['selected_engines']}"
            )
            
            return {
                **state,
                "plan_id": plan["plan_id"],
                "engine_scores": plan["all_engine_scores"],
                "selected_engines": plan["selected_engines"],
                "engine_configs": plan["engine_configs"],
                "routing_rationale": plan["routing_rationale"]
            }
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {**state, "error": f"Planning failed: {str(e)}"}
    
    def _should_retrieve(self, state: PolicyGraphState) -> str:
        """Conditional edge: check if retrieval should proceed"""
        if is_error_state(state):
            return "error"
        if not state.get("selected_engines"):
            return "error"
        return "retrieve"
    
    async def _retrieve_all_node(self, state: PolicyGraphState) -> PolicyGraphState:
        """Node 3: Parallel retrieval from all selected engines"""
        logger.info(
            f"[{state['request_id']}] Retrieving from "
            f"{len(state['selected_engines'])} engines"
        )
        
        try:
            # Parallel retrieval using asyncio.gather
            tasks = []
            for engine_name in state["selected_engines"]:
                engine_config = state["engine_configs"][engine_name]
                task = self.rag_client.search(
                    engine_name=engine_name,
                    query=state["query"],
                    config=engine_config
                )
                tasks.append(task)
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect documents
            all_docs = []
            metadata = {}
            
            for i, (engine_name, result) in enumerate(zip(state["selected_engines"], results)):
                if isinstance(result, Exception):
                    logger.error(f"Engine {engine_name} failed: {result}")
                    metadata[engine_name] = {"status": "error", "error": str(result)}
                else:
                    all_docs.extend(result["documents"])
                    metadata[engine_name] = {
                        "status": "success",
                        "count": len(result["documents"]),
                        "latency_ms": result.get("latency_ms", 0)
                    }
            
            logger.info(
                f"[{state['request_id']}] Retrieved {len(all_docs)} total documents"
            )
            
            return {
                **state,
                "retrieved_docs": all_docs,
                "retrieval_metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {**state, "error": f"Retrieval failed: {str(e)}"}
    
    async def _fuse_node(self, state: PolicyGraphState) -> PolicyGraphState:
        """Node 4: Deduplicate and rerank documents"""
        logger.info(f"[{state['request_id']}] Fusing results")
        
        try:
            docs = state["retrieved_docs"]
            
            # Step 1: Deduplicate
            deduped = deduplicate_docs(docs)
            logger.info(f"After dedup: {len(deduped)} docs")
            
            # Step 2: Rerank using Vertex AI Ranking API
            reranked = await rerank_docs(
                query=state["query"],
                documents=deduped,
                config=self.config
            )
            logger.info(f"After rerank: {len(reranked)} docs")
            
            # Step 3: Merge and trim to final_k
            final_k = self.config["ranking"]["final_k"]
            final = merge_and_trim(reranked, top_k=final_k)
            logger.info(f"Final set: {len(final)} docs")
            
            return {
                **state,
                "deduplicated_docs": deduped,
                "reranked_docs": reranked,
                "final_docs": final
            }
            
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            return {**state, "error": f"Fusion failed: {str(e)}"}
    
    async def _synthesize_node(self, state: PolicyGraphState) -> PolicyGraphState:
        """Node 5: Generate grounded answer with citations"""
        logger.info(f"[{state['request_id']}] Synthesizing answer")
        
        try:
            result = await synthesize_answer(
                query=state["query"],
                documents=state["final_docs"],
                features=state["features"],
                config=self.config
            )
            
            logger.info(
                f"[{state['request_id']}] Answer generated with "
                f"{len(result['citations'])} citations"
            )
            
            return {
                **state,
                "answer": result["answer"],
                "citations": result["citations"],
                "confidence": result["confidence"],
                "used_engines": state["selected_engines"]
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {**state, "error": f"Synthesis failed: {str(e)}"}
    
    async def ainvoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the graph asynchronously"""
        try:
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            raise
    
    def get_plan(self, plan_id: str) -> Dict[str, Any]:
        """Retrieve a saved plan from checkpointed state"""
        # Implementation depends on checkpoint storage
        # For now, return None (implement with your checkpoint store)
        return None


def build_policy_graph(config: Dict[str, Any]) -> PolicyGraphOrchestrator:
    """Factory function to build and return the orchestrator"""
    return PolicyGraphOrchestrator(config)


if __name__ == "__main__":
    # Test graph construction
    from config import load_config
    
    config = load_config()
    graph = build_policy_graph(config)
    print("✅ LangGraph orchestrator built successfully")
    print(f"Graph nodes: {graph.graph.nodes}")