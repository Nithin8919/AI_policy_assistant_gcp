"""
Vertex AI Ranking API - Wrapper for semantic reranking
Uses Vertex AI's ranking service to reorder documents by relevance
"""
from typing import List, Dict, Any
try:
    from google.cloud.discoveryengine import RankServiceClient, RankRequest, RankingRecord
    discoveryengine = None  # Use direct imports
except ImportError:
    try:
        from google.cloud import discoveryengine_v1 as discoveryengine
        RankServiceClient = discoveryengine.RankServiceClient
        RankRequest = discoveryengine.RankRequest
        RankingRecord = discoveryengine.RankingRecord
    except ImportError:
        discoveryengine = None
        RankServiceClient = None
        RankRequest = None
        RankingRecord = None
from utils.logging import get_logger

logger = get_logger()


class VertexRankingAPI:
    """Client for Vertex AI Ranking API"""
    
    def __init__(self, project_id: str, location: str, model: str = "semantic-ranker-512@latest"):
        self.project_id = project_id
        self.location = location
        self.model = model
        
        # Initialize the ranking service client
        if RankServiceClient is None:
            raise ImportError("RankServiceClient not available. Please install google-cloud-discoveryengine correctly.")
        self.client = RankServiceClient()
        
        # Construct the ranking config path
        self.ranking_config = f"projects/{project_id}/locations/{location}/rankingConfigs/default_ranking_config"
        
        logger.info(
            f"Initialized Vertex Ranking API: "
            f"project={project_id}, location={location}, model={model}"
        )
    
    async def rank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Vertex AI Ranking API
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []
        
        try:
            logger.info(f"Reranking {len(documents)} documents with query: '{query[:50]}...'")
            
            # Prepare records for ranking
            records = [
                {
                    "id": doc.get("id", str(i)),
                    "title": doc.get("title", ""),
                    "content": doc.get("text", "")
                }
                for i, doc in enumerate(documents)
            ]
            
            # Create the ranking request
            request = RankRequest(
                ranking_config=self.ranking_config,
                model=self.model,
                top_n=top_k,
                query=query,
                records=[
                    RankingRecord(
                        id=record["id"],
                        title=record["title"],
                        content=record["content"]
                    )
                    for record in records
                ]
            )
            
            # Execute ranking
            response = self.client.rank(request=request)
            
            # Map ranked results back to original documents
            reranked = self._map_ranked_results(
                response.records,
                documents
            )
            
            logger.info(f"Reranking complete: {len(reranked)} documents")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Ranking API call failed: {e}", exc_info=True)
            # Fallback: return original documents with their scores
            return documents[:top_k]
    
    def _map_ranked_results(
        self,
        ranked_records: List[Any],
        original_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map ranked records back to original document structure"""
        
        # Create lookup dictionary
        doc_lookup = {doc.get("id", str(i)): doc for i, doc in enumerate(original_docs)}
        
        reranked_docs = []
        for rank, record in enumerate(ranked_records):
            doc_id = record.id
            if doc_id in doc_lookup:
                doc = doc_lookup[doc_id].copy()
                # Update with reranking score and rank
                doc["rerank_score"] = record.score if hasattr(record, 'score') else 0.0
                doc["rerank"] = rank
                reranked_docs.append(doc)
        
        return reranked_docs


async def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convenience function for reranking documents
    
    Args:
        query: Search query
        documents: Documents to rerank
        config: System configuration
    
    Returns:
        Reranked documents
    """
    if not documents:
        return []
    
    project_id = config["project"]["gcp_project_id"]
    location = config["project"]["location"]
    model = config["ranking"].get("model", "semantic-ranker-512@latest")
    top_k = config["ranking"].get("final_k", 20)
    
    ranker = VertexRankingAPI(project_id, location, model)
    reranked = await ranker.rank(query, documents, top_k)
    
    return reranked


if __name__ == "__main__":
    # Test ranking API
    import asyncio
    from config import load_config
    
    config = load_config()
    
    async def test():
        # Mock documents
        docs = [
            {
                "id": "doc1",
                "text": "Teacher transfer rules are governed by GO Ms No 45",
                "vertical": "legal",
                "score": 0.85
            },
            {
                "id": "doc2",
                "text": "Budget allocation for education department",
                "vertical": "data_report",
                "score": 0.60
            },
            {
                "id": "doc3",
                "text": "Transfer policy implementation guidelines",
                "vertical": "gos",
                "score": 0.75
            }
        ]
        
        ranker = VertexRankingAPI(
            project_id=config["vertex_ai"]["project_id"],
            location=config["vertex_ai"]["location"]
        )
        
        reranked = await ranker.rank(
            query="What are the transfer rules?",
            documents=docs,
            top_k=3
        )
        
        print("Reranked results:")
        for i, doc in enumerate(reranked):
            print(f"{i+1}. {doc['text'][:50]}... (score: {doc.get('rerank_score', 0):.3f})")
    
    asyncio.run(test())