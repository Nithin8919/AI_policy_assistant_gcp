"""
Test enhanced query system with LLM optimization
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview import rag
from vertexai.preview.rag import RagResource

# Simple logger for this test
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

logger = SimpleLogger()

# Configuration
PROJECT_ID = "tech-bharath"
LOCATION = "asia-south1"
CORPUS_ID = "projects/tech-bharath/locations/asia-south1/ragCorpora/7638104968020361216"

class LLMQueryEnhancer:
    """LLM-powered query enhancement"""
    
    def __init__(self):
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.model = GenerativeModel("gemini-2.5-flash")
    
    async def enhance_query(self, query: str, agent_type: str = "education") -> dict:
        """Enhance query using LLM"""
        
        prompt = f"""You are an expert at optimizing search queries for Andhra Pradesh education policy document retrieval.

TASK: Enhance the following query for better document retrieval from education policy corpus.

ORIGINAL QUERY: "{query}"

AGENT TYPE: {agent_type}

INSTRUCTIONS:
1. Analyze the intent and make it more searchable
2. Add relevant synonyms and related terms commonly found in official documents
3. Include domain-specific terminology used in government/education documents
4. Remove overly specific terms that might limit results
5. Focus on terms likely to appear in official policy documents

FORMAT YOUR RESPONSE AS:

ENHANCED QUERY: [Your enhanced query here]

SEARCH TERMS: [List 3-5 key search terms separated by commas]

REASONING: [Brief explanation of why this enhancement will improve retrieval]

CONFIDENCE: [High/Medium/Low confidence in this enhancement]"""

        try:
            loop = asyncio.get_event_loop()
            
            def sync_generate():
                response = self.model.generate_content(prompt)
                return response.text
            
            enhanced_response = await loop.run_in_executor(None, sync_generate)
            return self._parse_response(enhanced_response, query)
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return {
                "enhanced_query": query,
                "search_terms": [query],
                "reasoning": "LLM failed, using original",
                "confidence": "Low"
            }
    
    def _parse_response(self, response: str, original: str) -> dict:
        """Parse LLM response"""
        result = {
            "enhanced_query": original,
            "search_terms": [original],
            "reasoning": "Parsing failed",
            "confidence": "Low"
        }
        
        try:
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("ENHANCED QUERY:"):
                    result["enhanced_query"] = line.replace("ENHANCED QUERY:", "").strip()
                elif line.startswith("SEARCH TERMS:"):
                    terms = line.replace("SEARCH TERMS:", "").strip()
                    result["search_terms"] = [t.strip() for t in terms.split(',')]
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.replace("REASONING:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    result["confidence"] = line.replace("CONFIDENCE:", "").strip()
        except:
            pass
            
        return result

async def test_rag_with_enhancements():
    """Test RAG retrieval with different query enhancement approaches"""
    
    enhancer = LLMQueryEnhancer()
    
    # Test queries
    test_queries = [
        "teacher transfer rules",
        "student scholarship eligibility", 
        "primary school curriculum",
        "education budget allocation",
        "school infrastructure guidelines"
    ]
    
    print("=" * 80)
    print("TESTING RAG WITH LLM-ENHANCED QUERIES")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"TESTING QUERY: {query}")
        print(f"{'='*60}")
        
        # 1. Test original query
        print("\n1. ORIGINAL QUERY RESULTS:")
        original_results = await search_rag(query)
        print(f"   Found: {len(original_results)} documents")
        if original_results:
            print(f"   Sample: {original_results[0]['text'][:80]}...")
        
        # 2. Test LLM-enhanced query
        print("\n2. LLM ENHANCEMENT:")
        enhancement = await enhancer.enhance_query(query, "education")
        enhanced_query = enhancement["enhanced_query"]
        
        print(f"   Original: {query}")
        print(f"   Enhanced: {enhanced_query}")
        print(f"   Reasoning: {enhancement['reasoning']}")
        print(f"   Confidence: {enhancement['confidence']}")
        
        print("\n3. ENHANCED QUERY RESULTS:")
        enhanced_results = await search_rag(enhanced_query)
        print(f"   Found: {len(enhanced_results)} documents")
        if enhanced_results:
            print(f"   Sample: {enhanced_results[0]['text'][:80]}...")
        
        # 3. Compare results
        print(f"\n4. IMPROVEMENT:")
        improvement = len(enhanced_results) - len(original_results)
        if improvement > 0:
            print(f"   ✅ +{improvement} more documents found")
        elif improvement == 0:
            print(f"   ➡️  Same number of documents")
        else:
            print(f"   ⚠️  {abs(improvement)} fewer documents")
        
        # 4. Test each search term individually
        print(f"\n5. INDIVIDUAL SEARCH TERMS:")
        for term in enhancement["search_terms"][:3]:  # Test top 3 terms
            term_results = await search_rag(term)
            print(f"   '{term}': {len(term_results)} docs")

async def search_rag(query: str) -> list:
    """Search RAG corpus"""
    try:
        response = rag.retrieval_query(
            text=query,
            rag_resources=[RagResource(rag_corpus=CORPUS_ID)],
            similarity_top_k=5
        )
        
        contexts = []
        if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
            contexts_list = list(response.contexts.contexts)
            for ctx in contexts_list:
                if hasattr(ctx, 'text'):
                    contexts.append({
                        'text': ctx.text,
                        'source': getattr(ctx, 'source_uri', ''),
                        'distance': getattr(ctx, 'distance', 1.0)
                    })
        return contexts
        
    except Exception as e:
        logger.error(f"RAG search failed for '{query}': {e}")
        return []

async def find_best_enhancements():
    """Find the best query enhancements for problematic queries"""
    
    enhancer = LLMQueryEnhancer()
    
    # Queries that previously returned 0 results
    problematic_queries = [
        "teacher transfer rules",
        "government order",
        "student scholarship",
        "learning outcomes"
    ]
    
    print("\n" + "=" * 80)
    print("FINDING BEST ENHANCEMENTS FOR PROBLEMATIC QUERIES")
    print("=" * 80)
    
    best_enhancements = {}
    
    for query in problematic_queries:
        print(f"\nOptimizing: {query}")
        
        # Try multiple enhancement approaches
        enhancements = []
        
        # Approach 1: General enhancement
        enh1 = await enhancer.enhance_query(query, "education")
        enhancements.append(("General", enh1))
        
        # Approach 2: Policy-focused enhancement
        policy_query = f"policy guidelines for {query}"
        enh2 = await enhancer.enhance_query(policy_query, "education")
        enhancements.append(("Policy-focused", enh2))
        
        # Approach 3: Regulation-focused enhancement
        reg_query = f"regulations and rules about {query}"
        enh3 = await enhancer.enhance_query(reg_query, "education")
        enhancements.append(("Regulation-focused", enh3))
        
        # Test each enhancement
        best_count = 0
        best_enhancement = None
        
        for name, enh in enhancements:
            results = await search_rag(enh["enhanced_query"])
            count = len(results)
            print(f"  {name}: {count} results - '{enh['enhanced_query'][:50]}...'")
            
            if count > best_count:
                best_count = count
                best_enhancement = (name, enh)
        
        if best_enhancement:
            best_enhancements[query] = best_enhancement
            print(f"  ✅ Best: {best_enhancement[0]} with {best_count} results")
        else:
            print(f"  ❌ No working enhancement found")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - BEST ENHANCEMENTS")
    print(f"{'='*60}")
    
    for query, (approach, enhancement) in best_enhancements.items():
        print(f"\nQuery: {query}")
        print(f"Best approach: {approach}")
        print(f"Enhanced: {enhancement['enhanced_query']}")
        print(f"Reasoning: {enhancement['reasoning']}")

if __name__ == "__main__":
    async def main():
        await test_rag_with_enhancements()
        await find_best_enhancements()
    
    asyncio.run(main())