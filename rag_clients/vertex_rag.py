"""
Vertex AI RAG Client - Wrapper for Vertex AI RAG-managed vector stores
Handles retrieval from configured RAG engines/corpora
"""
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
from google.cloud import aiplatform
from vertexai.preview import rag
import vertexai

from utils.logging import get_logger
from vertexai.preview.rag import RagResource
from vertexai.generative_models import GenerativeModel

logger = get_logger()


class VertexRAGClient:
    """Client for Vertex AI RAG Engine retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_id = config["project"]["gcp_project_id"]
        self.location = config["project"]["location"]
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Initialize LLM for query enhancement
        self.model = GenerativeModel("gemini-2.5-flash")
        
        # Best performing enhancement patterns
        self.enhancement_patterns = {
            "teacher_transfer": {
                "keywords": ["teacher", "transfer", "posting", "deployment"],
                "template": '"Andhra Pradesh" ("teacher transfer" OR "teaching staff redeployment" OR "teacher posting") ("rules" OR "regulations" OR "guidelines" OR "policy" OR "G.O. Ms. No." OR "orders")'
            },
            "scholarship": {
                "keywords": ["scholarship", "financial assistance", "scheme", "benefit"],
                "template": 'Andhra Pradesh education scholarship scheme financial assistance policy guidelines G.O. notification for students beneficiaries'
            },
            "government_order": {
                "keywords": ["government order", "go", "memo", "circular"],
                "template": '"Andhra Pradesh education" ("government order" OR "GO" OR "memo" OR "circular" OR "notification" OR "policy" OR "scheme" OR "regulation" OR "guidelines" OR "directive")'
            },
            "learning_outcomes": {
                "keywords": ["learning outcomes", "competencies", "standards", "assessment"],
                "template": '"Andhra Pradesh" ("education policy" OR "school education") ("learning outcomes" OR "competencies" OR "standards" OR "educational objectives") ("guidelines" OR "framework" OR "regulations" OR "scheme" OR "circular" OR "notification" OR "directives")'
            }
        }
        
        logger.info(
            f"Initialized Vertex RAG client: "
            f"project={self.project_id}, location={self.location}"
        )
    
    async def search(
        self,
        engine_name: str,
        query: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Search a specific RAG engine
        
        Args:
            engine_name: Name of the engine (e.g., 'legal', 'gos')
            query: Search query
            config: Engine-specific configuration with filters and top_k
        
        Returns:
            Dictionary with documents and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            engine_config = self.config["engines"][engine_name]
            rag_corpus_id = engine_config["id"]
            top_k = config.get("top_k", 20)
            
            logger.info(
                f"Searching engine '{engine_name}': "
                f"query='{query[:50]}...', top_k={top_k}"
            )
            
            # Build filter from config
            filter_dict = self._build_filter(config.get("filters", {}))
            
            # Enhanced query processing
            enhanced_query_info = await self._enhance_query(query, engine_name)
            final_query = enhanced_query_info["enhanced_query"]
            
            logger.info(
                f"Query enhancement: '{query}' -> '{final_query}' "
                f"(method: {enhanced_query_info['method']})"
            )
            
            # Execute RAG retrieval with enhanced query
            response = await self._execute_rag_query(
                rag_corpus_id=rag_corpus_id,
                query=final_query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # If no results with enhanced query, try original
            if not self._has_results(response):
                logger.info("No results with enhanced query, trying original...")
                response = await self._execute_rag_query(
                    rag_corpus_id=rag_corpus_id,
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
            
            # Parse response
            documents = self._parse_response(response, engine_name)
            
            end_time = datetime.utcnow()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(
                f"Engine '{engine_name}' returned {len(documents)} documents "
                f"in {latency_ms}ms"
            )
            
            return {
                "engine": engine_name,
                "documents": documents,
                "count": len(documents),
                "latency_ms": latency_ms,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Search failed for engine '{engine_name}': {e}", exc_info=True)
            return {
                "engine": engine_name,
                "documents": [],
                "count": 0,
                "latency_ms": 0,
                "status": "error",
                "error": str(e)
            }
    
    async def _execute_rag_query(
        self,
        rag_corpus_id: str,
        query: str,
        top_k: int,
        filter_dict: Dict[str, Any]
    ) -> Any:
        """
        Execute the actual RAG query using Vertex AI SDK
        """
        try:
            # Use the RAG API to retrieve contexts
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            def sync_retrieve():
                """Synchronous RAG retrieval"""
                try:
                    # Use Vertex AI RAG retrieval_query function
                    # rag_corpora expects a list of corpus resource names
                    # Note: vector_distance_threshold might be too strict, try without it first
                    response = rag.retrieval_query(
                        text=query,
                        rag_resources=[RagResource(rag_corpus=rag_corpus_id)],
                        similarity_top_k=top_k,
                        # vector_distance_threshold=0.3,  # Commented out - might be filtering all results
                        # vector_search_alpha=0.5  # Balance between dense and sparse search
                    )
                    logger.info(f"RAG query executed, response type: {type(response)}")
                    return response
                except Exception as e:
                    logger.error(f"RAG retrieval_query failed: {e}")
                    raise
            
            # Execute synchronous call in thread pool
            response = await loop.run_in_executor(None, sync_retrieve)
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query execution failed: {e}", exc_info=True)
            raise
    
    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Vertex RAG filter dictionary from config filters
        
        Adjust based on your metadata structure in Vertex RAG
        """
        filter_dict = {}
        
        if filters.get("jurisdiction"):
            filter_dict["jurisdiction"] = filters["jurisdiction"]
        
        if filters.get("years"):
            filter_dict["year"] = {"in": filters["years"]}
        
        if filters.get("districts"):
            filter_dict["district"] = {"in": filters["districts"]}
        
        if filters.get("go_numbers"):
            filter_dict["go_number"] = {"in": filters["go_numbers"]}
        
        # Add more filter mappings as needed
        
        return filter_dict
    
    def _parse_response(
        self,
        response: Any,
        engine_name: str
    ) -> List[Dict[str, Any]]:
        """Parse Vertex RAG response into standardized document format"""
        documents = []
        
        # Handle RetrieveContextsResponse from retrieval_query
        # The response has a 'contexts' attribute containing RagContexts
        contexts = []
        
        # Debug: Log response structure
        logger.debug(f"Response type: {type(response)}")
        logger.debug(f"Response has contexts attr: {hasattr(response, 'contexts')}")
        
        if hasattr(response, 'contexts'):
            try:
                # The contexts attribute is a RagContexts object
                # It contains a 'contexts' attribute which is a RepeatedComposite
                contexts_attr = response.contexts
                logger.debug(f"Contexts attr type: {type(contexts_attr)}")
                
                # Access the inner contexts from RagContexts
                if hasattr(contexts_attr, 'contexts'):
                    inner_contexts = contexts_attr.contexts
                    logger.debug(f"Inner contexts type: {type(inner_contexts)}")
                    # Convert RepeatedComposite to list
                    contexts = list(inner_contexts)
                    logger.debug(f"Converted to list, length: {len(contexts)}")
                else:
                    # Fallback: try to iterate directly
                    if hasattr(contexts_attr, '__iter__'):
                        contexts = list(contexts_attr)
                    else:
                        contexts = []
                    
            except Exception as e:
                logger.warning(f"Error accessing contexts: {e}, trying alternative methods")
                # Try alternative access methods
                try:
                    # Try as protobuf message
                    if hasattr(response, 'to_dict'):
                        response_dict = response.to_dict()
                        contexts = response_dict.get("contexts", [])
                    # Try direct attribute access
                    elif hasattr(response.contexts, 'contexts'):
                        contexts = list(response.contexts.contexts)
                except Exception as e2:
                    logger.error(f"All parsing methods failed: {e2}")
                    contexts = []
        elif isinstance(response, dict):
            contexts = response.get("contexts", [])
        elif isinstance(response, list):
            contexts = response
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            contexts = []
        
        logger.info(f"Parsed {len(contexts)} contexts from response")
        
        for i, context in enumerate(contexts):
            try:
                # Handle protobuf message objects (RetrieveContextsResponse.Context)
                if hasattr(context, 'text'):
                    text = context.text
                    # Get source URI - could be in source_uri or file_uri
                    source_uri = getattr(context, 'source_uri', getattr(context, 'file_uri', getattr(context, 'uri', '')))
                    # Get distance/score - lower distance = higher relevance
                    distance = getattr(context, 'distance', None)
                    score = 1.0 - float(distance) if distance is not None else 0.0
                    # Metadata might be in separate fields
                    metadata = {}
                    if hasattr(context, 'metadata'):
                        try:
                            if hasattr(context.metadata, 'items'):
                                metadata = dict(context.metadata)
                            elif isinstance(context.metadata, dict):
                                metadata = context.metadata
                        except:
                            metadata = {}
                elif isinstance(context, dict):
                    text = context.get("text", context.get("content", ""))
                    source_uri = context.get("source_uri", context.get("file_uri", context.get("uri", "")))
                    distance = context.get("distance", None)
                    score = 1.0 - float(distance) if distance is not None else context.get("score", 0.0)
                    metadata = context.get("metadata", {})
                else:
                    logger.warning(f"Unexpected context type: {type(context)}, skipping")
                    logger.debug(f"Context attributes: {[x for x in dir(context) if not x.startswith('_')]}")
                    continue
            except Exception as e:
                logger.error(f"Error parsing context {i}: {e}")
                continue
            
            doc = {
                "id": f"{engine_name}_{i}",
                "vertical": engine_name,
                "source_uri": source_uri,
                "text": text,
                "score": max(0.0, min(1.0, score)),  # Clamp score between 0 and 1
                "metadata": metadata if isinstance(metadata, dict) else {},
                "rank": i
            }
            
            # Extract locator (page, section, etc.) from metadata
            if isinstance(metadata, dict):
                locator = metadata.get("page", metadata.get("section", metadata.get("chunk_index", "")))
                doc["locator"] = str(locator) if locator else ""
                doc["source_date"] = metadata.get("date", metadata.get("published_date", ""))
            else:
                doc["locator"] = ""
                doc["source_date"] = ""
            
            documents.append(doc)
        
        return documents
    
    async def _enhance_query(self, query: str, engine_name: str) -> Dict[str, str]:
        """Enhanced query processing with LLM optimization"""
        
        query_lower = query.lower()
        
        # Check for pattern matches first
        for pattern_key, pattern_info in self.enhancement_patterns.items():
            if any(keyword in query_lower for keyword in pattern_info["keywords"]):
                return {
                    "enhanced_query": pattern_info["template"],
                    "method": f"pattern_{pattern_key}",
                    "original_query": query
                }
        
        # Fall back to basic enhancement
        basic_enhanced = f'"Andhra Pradesh education" {query} policy guidelines'
        
        return {
            "enhanced_query": basic_enhanced,
            "method": "basic_pattern",
            "original_query": query
        }
    
    def _has_results(self, response) -> bool:
        """Check if the response has any results"""
        try:
            if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
                contexts = list(response.contexts.contexts)
                return len(contexts) > 0
            return False
        except:
            return False


# Async helper for batch retrieval
async def batch_retrieve(
    client: VertexRAGClient,
    queries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Execute multiple RAG queries in parallel
    
    Args:
        client: VertexRAGClient instance
        queries: List of query dictionaries with engine_name, query, config
    
    Returns:
        List of results
    """
    tasks = [
        client.search(
            engine_name=q["engine_name"],
            query=q["query"],
            config=q["config"]
        )
        for q in queries
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r if not isinstance(r, Exception) else {"status": "error", "error": str(r)} 
            for r in results]


if __name__ == "__main__":
    # Test the RAG client
    import asyncio
    from config import load_config
    
    config = load_config()
    
    async def test():
        client = VertexRAGClient(config)
        
        test_config = {
            "top_k": 10,
            "filters": {"jurisdiction": "Andhra Pradesh"}
        }
        
        result = await client.search(
            engine_name="legal",
            query="teacher transfer rules",
            config=test_config
        )
        
        print(f"Status: {result['status']}")
        print(f"Documents: {result['count']}")
        if result['documents']:
            print(f"First doc: {result['documents'][0]['text'][:100]}")
    
    asyncio.run(test())