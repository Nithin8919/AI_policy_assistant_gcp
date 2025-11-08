"""
Fixed RAG client that works with your Vertex AI setup
"""
import vertexai
from vertexai.preview import rag
from vertexai.preview.rag import RagResource
import asyncio

# Your configuration
PROJECT_ID = "tech-bharath"
LOCATION = "asia-south1"
CORPUS_ID = "projects/tech-bharath/locations/asia-south1/ragCorpora/7638104968020361216"

class FixedVertexRAGClient:
    """Fixed Vertex RAG Client that works with your setup"""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(f"Initialized RAG client for project {PROJECT_ID} in {LOCATION}")
    
    async def search(self, query: str, top_k: int = 10) -> dict:
        """Search the RAG corpus"""
        print(f"Searching for: '{query}' (top_k={top_k})")
        
        try:
            # Execute query
            loop = asyncio.get_event_loop()
            
            def sync_search():
                return rag.retrieval_query(
                    text=query,
                    rag_resources=[RagResource(rag_corpus=CORPUS_ID)],
                    similarity_top_k=top_k
                )
            
            response = await loop.run_in_executor(None, sync_search)
            
            # Parse response
            documents = []
            if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
                contexts = list(response.contexts.contexts)
                print(f"Found {len(contexts)} contexts")
                
                for i, ctx in enumerate(contexts):
                    if hasattr(ctx, 'text'):
                        doc = {
                            "id": f"doc_{i}",
                            "text": ctx.text,
                            "source_uri": getattr(ctx, 'source_uri', ''),
                            "distance": getattr(ctx, 'distance', 1.0),
                            "sparse_distance": getattr(ctx, 'sparse_distance', 1.0),
                            "score": 1.0 - float(getattr(ctx, 'distance', 1.0)),  # Convert distance to score
                            "rank": i
                        }
                        documents.append(doc)
            
            return {
                "documents": documents,
                "count": len(documents),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error in search: {e}")
            return {
                "documents": [],
                "count": 0, 
                "status": "error",
                "error": str(e)
            }

async def test_fixed_client():
    """Test the fixed client with various queries"""
    client = FixedVertexRAGClient()
    
    # Test queries - some that work, some that don't
    test_queries = [
        "education policy",           # Known to work
        "teacher transfer rules",     # Your original query
        "government order",
        "school management",
        "student scholarship",
        "NEP implementation",
        "budget allocation",
        "learning outcomes"
    ]
    
    print("Testing Fixed RAG Client")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await client.search(query, top_k=5)
        
        if result["status"] == "success" and result["count"] > 0:
            print(f"✅ Found {result['count']} documents")
            first_doc = result["documents"][0]
            print(f"  Score: {first_doc['score']:.3f}")
            print(f"  Source: {first_doc['source_uri'].split('/')[-1] if first_doc['source_uri'] else 'No source'}")
            print(f"  Text preview: {first_doc['text'][:100]}...")
        elif result["status"] == "success":
            print(f"⚠️  No documents found")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

async def test_education_agent_integration():
    """Test integration with education agent patterns"""
    from agents.education import EducationAgent
    from config import load_config
    
    print("\n" + "=" * 50)
    print("TESTING EDUCATION AGENT INTEGRATION")
    print("=" * 50)
    
    # Load config
    config = load_config()
    
    # Create education agent
    education_agent = EducationAgent(config)
    
    # Create fixed RAG client
    rag_client = FixedVertexRAGClient()
    
    # Test query
    query = "What are the guidelines for teacher recruitment in primary schools?"
    
    # Analyze with education agent
    features = {
        "entities": {"education_levels": ["primary"]},
        "query_type": "policy",
        "temporal": {"years": ["2023"]}
    }
    
    parsed_query = education_agent.analyze_query(query, features)
    print(f"Parsed query: {parsed_query}")
    
    enhanced_query = education_agent.enhance_query(query, parsed_query)
    print(f"Enhanced query: {enhanced_query}")
    
    # Search with enhanced query
    result = await rag_client.search(enhanced_query, top_k=10)
    
    if result["count"] > 0:
        print(f"✅ Found {result['count']} documents with enhanced query")
        
        # Post-process results
        processed_docs = education_agent.postprocess_results(result["documents"], parsed_query)
        
        print("Top 3 processed results:")
        for i, doc in enumerate(processed_docs[:3]):
            print(f"  {i+1}. Score: {doc.get('education_score', 0):.3f}")
            print(f"     Text: {doc['text'][:80]}...")
    else:
        print("❌ No results even with enhanced query")

if __name__ == "__main__":
    asyncio.run(test_fixed_client())
    asyncio.run(test_education_agent_integration())