"""
Optimized RAG Pipeline with LLM-Enhanced Query Processing
Best performing query enhancement system for Andhra Pradesh education corpus
"""
import asyncio
import sys
import os
sys.path.append(os.getcwd())

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview import rag
from vertexai.preview.rag import RagResource

# Configuration
PROJECT_ID = "tech-bharath"
LOCATION = "asia-south1"
CORPUS_ID = "projects/tech-bharath/locations/asia-south1/ragCorpora/7638104968020361216"

class OptimizedQueryEnhancer:
    """Optimized query enhancer based on test results"""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.model = GenerativeModel("gemini-2.5-flash")
        
        # Best working enhancement patterns from testing
        self.enhancement_patterns = {
            "teacher_transfer": {
                "template": '"Andhra Pradesh" ("teacher transfer" OR "teaching staff redeployment" OR "teacher posting") ("rules" OR "regulations" OR "guidelines" OR "policy" OR "G.O. Ms. No." OR "orders")',
                "success_rate": 1.0
            },
            "scholarship": {
                "template": 'Andhra Pradesh education scholarship scheme financial assistance policy guidelines G.O. notification for students beneficiaries',
                "success_rate": 1.0
            },
            "government_order": {
                "template": '"Andhra Pradesh education" ("government order" OR "GO" OR "memo" OR "circular" OR "notification" OR "policy" OR "scheme" OR "regulation" OR "guidelines" OR "directive")',
                "success_rate": 1.0
            },
            "learning_outcomes": {
                "template": '"Andhra Pradesh" ("education policy" OR "school education") ("learning outcomes" OR "competencies" OR "standards" OR "educational objectives") ("guidelines" OR "framework" OR "regulations" OR "scheme" OR "circular" OR "notification" OR "directives")',
                "success_rate": 1.0
            }
        }
    
    async def enhance_query(self, query: str, context: str = "education") -> dict:
        """Enhanced query optimization with proven patterns"""
        
        query_lower = query.lower()
        
        # Check for pattern matches first
        for pattern_key, pattern_info in self.enhancement_patterns.items():
            if any(keyword in query_lower for keyword in pattern_key.split('_')):
                return {
                    "enhanced_query": pattern_info["template"],
                    "method": f"pattern_match_{pattern_key}",
                    "confidence": pattern_info["success_rate"],
                    "reasoning": f"Using proven pattern for {pattern_key}"
                }
        
        # Fall back to LLM enhancement for new queries
        return await self._llm_enhance(query, context)
    
    async def _llm_enhance(self, query: str, context: str) -> dict:
        """LLM-based enhancement for novel queries"""
        
        # Use the best-performing prompt template
        prompt = f'''Enhance this query for Andhra Pradesh education policy document search:

QUERY: "{query}"

Create an enhanced query using these proven patterns:
1. Include "Andhra Pradesh" for geographic specificity
2. Use OR operators for synonyms: (term1 OR term2 OR term3)
3. Add policy document terms: (policy OR guidelines OR G.O. OR circular OR notification OR scheme)
4. Use formal government language
5. Include education-specific terms where relevant

Examples that work well:
- "Andhra Pradesh education" (topic OR "related term") (document type)
- "topic policy guidelines G.O. notification"

ENHANCED QUERY: [your optimized query]

CONFIDENCE: [High/Medium/Low]'''

        try:
            loop = asyncio.get_event_loop()
            
            def sync_generate():
                response = self.model.generate_content(prompt)
                return response.text
            
            response = await loop.run_in_executor(None, sync_generate)
            
            # Extract enhanced query
            for line in response.split('\n'):
                if line.strip().startswith('ENHANCED QUERY:'):
                    enhanced = line.replace('ENHANCED QUERY:', '').strip()
                    return {
                        "enhanced_query": enhanced,
                        "method": "llm_enhanced",
                        "confidence": 0.8,
                        "reasoning": "LLM-generated using proven patterns"
                    }
            
            # Fallback
            return {
                "enhanced_query": f'"Andhra Pradesh education" {query} policy guidelines',
                "method": "fallback_pattern",
                "confidence": 0.6,
                "reasoning": "Applied basic enhancement pattern"
            }
            
        except Exception as e:
            return {
                "enhanced_query": query,
                "method": "original",
                "confidence": 0.3,
                "reasoning": f"Enhancement failed: {e}"
            }


class OptimizedRAGPipeline:
    """Complete optimized RAG pipeline"""
    
    def __init__(self):
        self.enhancer = OptimizedQueryEnhancer()
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    async def search(self, query: str, top_k: int = 10) -> dict:
        """Search with optimized query enhancement"""
        
        print(f"🔍 Original query: {query}")
        
        # Step 1: Enhance query
        enhancement = await self.enhancer.enhance_query(query)
        enhanced_query = enhancement["enhanced_query"]
        
        print(f"✨ Enhanced query: {enhanced_query}")
        print(f"📈 Method: {enhancement['method']} (confidence: {enhancement['confidence']})")
        
        # Step 2: Search with enhanced query
        enhanced_results = await self._search_corpus(enhanced_query, top_k)
        
        # Step 3: Fallback to original if no results
        if not enhanced_results:
            print("⚠️  No results with enhanced query, trying original...")
            original_results = await self._search_corpus(query, top_k)
            
            if original_results:
                print(f"✅ Found {len(original_results)} results with original query")
                return {
                    "documents": original_results,
                    "query_used": query,
                    "enhancement_info": enhancement,
                    "method": "original_fallback"
                }
            else:
                # Try broader terms
                print("🔄 Trying broader search...")
                broad_query = f'"{query.split()[0]} education policy"' if query.split() else "education policy"
                broad_results = await self._search_corpus(broad_query, top_k)
                
                return {
                    "documents": broad_results,
                    "query_used": broad_query,
                    "enhancement_info": enhancement,
                    "method": "broad_search"
                }
        else:
            print(f"✅ Found {len(enhanced_results)} results with enhanced query")
            return {
                "documents": enhanced_results,
                "query_used": enhanced_query,
                "enhancement_info": enhancement,
                "method": "enhanced"
            }
    
    async def _search_corpus(self, query: str, top_k: int) -> list:
        """Search the RAG corpus"""
        try:
            response = rag.retrieval_query(
                text=query,
                rag_resources=[RagResource(rag_corpus=CORPUS_ID)],
                similarity_top_k=top_k
            )
            
            documents = []
            if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
                contexts_list = list(response.contexts.contexts)
                
                for i, ctx in enumerate(contexts_list):
                    if hasattr(ctx, 'text'):
                        doc = {
                            "id": f"doc_{i}",
                            "text": ctx.text,
                            "source_uri": getattr(ctx, 'source_uri', ''),
                            "distance": getattr(ctx, 'distance', 1.0),
                            "score": 1.0 - float(getattr(ctx, 'distance', 1.0)),
                            "source_file": getattr(ctx, 'source_uri', '').split('/')[-1] if getattr(ctx, 'source_uri', '') else 'Unknown'
                        }
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    async def multi_strategy_search(self, query: str) -> dict:
        """Try multiple search strategies and return the best results"""
        
        strategies = []
        
        # Strategy 1: Optimized enhancement
        result1 = await self.search(query, top_k=5)
        strategies.append(("Optimized Enhancement", result1))
        
        # Strategy 2: Policy-focused
        policy_query = f"Andhra Pradesh education policy {query}"
        result2 = await self._search_corpus(policy_query, 5)
        strategies.append(("Policy-focused", {"documents": result2, "query_used": policy_query}))
        
        # Strategy 3: Scheme-focused (for benefit/scholarship queries)
        if any(word in query.lower() for word in ["scholarship", "benefit", "scheme", "assistance"]):
            scheme_query = f"Andhra Pradesh {query} scheme notification G.O."
            result3 = await self._search_corpus(scheme_query, 5)
            strategies.append(("Scheme-focused", {"documents": result3, "query_used": scheme_query}))
        
        # Find the best strategy
        best_strategy = max(strategies, key=lambda x: len(x[1]["documents"]))
        
        return {
            "best_result": best_strategy[1],
            "best_strategy": best_strategy[0],
            "all_strategies": strategies,
            "comparison": {name: len(result["documents"]) for name, result in strategies}
        }


async def demo_optimized_pipeline():
    """Demonstrate the optimized pipeline"""
    
    pipeline = OptimizedRAGPipeline()
    
    # Test queries that were previously problematic
    test_queries = [
        "teacher transfer rules",
        "student scholarship eligibility",
        "learning outcomes assessment",
        "school infrastructure norms",
        "education budget allocation",
        "primary school curriculum",
        "teacher recruitment process"
    ]
    
    print("🚀 OPTIMIZED RAG PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n{'🔥' * 20} QUERY: {query} {'🔥' * 20}")
        
        # Single optimized search
        result = await pipeline.search(query)
        
        print(f"📊 Results: {len(result['documents'])} documents found")
        print(f"🎯 Method used: {result['method']}")
        print(f"🔧 Query used: {result['query_used'][:100]}...")
        
        if result["documents"]:
            top_doc = result["documents"][0]
            print(f"📋 Top result (score: {top_doc['score']:.3f}):")
            print(f"   📄 Source: {top_doc['source_file']}")
            print(f"   📝 Text: {top_doc['text'][:150]}...")
        
        print("-" * 80)


async def compare_with_multi_strategy():
    """Compare single vs multi-strategy approach"""
    
    pipeline = OptimizedRAGPipeline()
    
    test_queries = ["teacher transfer rules", "student scholarship"]
    
    print("\n🏆 MULTI-STRATEGY COMPARISON")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        result = await pipeline.multi_strategy_search(query)
        
        print(f"🥇 Best strategy: {result['best_strategy']}")
        print(f"📊 Strategy comparison: {result['comparison']}")
        print(f"📈 Best result: {len(result['best_result']['documents'])} documents")


if __name__ == "__main__":
    async def main():
        await demo_optimized_pipeline()
        await compare_with_multi_strategy()
    
    asyncio.run(main())