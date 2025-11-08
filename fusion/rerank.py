"""
Rerank - Reorder documents using Vertex AI Ranking API
"""
from typing import List, Dict, Any
try:
    from rag_clients.ranking_api import VertexRankingAPI
    RANKING_API_AVAILABLE = True
except ImportError:
    RANKING_API_AVAILABLE = False
    VertexRankingAPI = None
from utils.logging import get_logger

logger = get_logger()


async def rerank_docs(
    query: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Vertex AI Ranking API
    
    This is a thin wrapper around the ranking API client
    with additional logic for handling edge cases
    """
    if not documents:
        logger.warning("No documents to rerank")
        return []
    
    if len(documents) == 1:
        logger.info("Only one document, skipping rerank")
        return documents
    
    # Check if ranking API is available
    if not RANKING_API_AVAILABLE or VertexRankingAPI is None:
        logger.warning("Ranking API not available, using score-based sorting")
        return sorted(documents, key=lambda d: d.get("score", 0.0), reverse=True)
    
    try:
        project_id = config["project"]["gcp_project_id"]
        location = config["project"]["location"]
        model = config["ranking"].get("model", "semantic-ranker-512@latest")
        top_k = min(len(documents), config["ranking"].get("final_k", 20))
        
        ranker = VertexRankingAPI(project_id, location, model)
        reranked = await ranker.rank(query, documents, top_k)
        
        return reranked
        
    except Exception as e:
        logger.error(f"Reranking failed, returning original order: {e}")
        # Fallback: return documents sorted by original score
        return sorted(documents, key=lambda d: d.get("score", 0.0), reverse=True)


def score_with_diversity(
    documents: List[Dict[str, Any]],
    diversity_weight: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Optionally adjust scores to promote diversity across engines/sources
    
    This helps ensure the final set includes evidence from multiple verticals
    rather than being dominated by a single high-scoring engine
    """
    if not documents:
        return []
    
    # Count documents per vertical
    vertical_counts = {}
    for doc in documents:
        vertical = doc.get("vertical", "unknown")
        vertical_counts[vertical] = vertical_counts.get(vertical, 0) + 1
    
    # Adjust scores with diversity penalty
    adjusted_docs = []
    vertical_seen = {}
    
    for doc in documents:
        vertical = doc.get("vertical", "unknown")
        times_seen = vertical_seen.get(vertical, 0)
        vertical_seen[vertical] = times_seen + 1
        
        # Apply diminishing returns for repeated verticals
        diversity_penalty = 1.0 - (diversity_weight * times_seen / 10.0)
        diversity_penalty = max(0.5, diversity_penalty)  # Floor at 50%
        
        adjusted_doc = doc.copy()
        original_score = doc.get("rerank_score", doc.get("score", 0.0))
        adjusted_doc["diversity_adjusted_score"] = original_score * diversity_penalty
        adjusted_docs.append(adjusted_doc)
    
    # Re-sort by adjusted score
    adjusted_docs.sort(key=lambda d: d.get("diversity_adjusted_score", 0.0), reverse=True)
    
    return adjusted_docs


if __name__ == "__main__":
    # Test reranking
    import asyncio
    from config import load_config
    
    config = load_config()
    
    async def test():
        test_docs = [
            {
                "id": "1",
                "text": "Transfer rules for teachers",
                "score": 0.85,
                "vertical": "legal"
            },
            {
                "id": "2",
                "text": "GO implementing transfer policy",
                "score": 0.80,
                "vertical": "gos"
            },
            {
                "id": "3",
                "text": "Budget for education",
                "score": 0.70,
                "vertical": "data_report"
            }
        ]
        
        reranked = await rerank_docs(
            query="What are teacher transfer rules?",
            documents=test_docs,
            config=config
        )
        
        print("Reranked documents:")
        for i, doc in enumerate(reranked):
            print(f"{i+1}. {doc['text']}")
    
    asyncio.run(test())