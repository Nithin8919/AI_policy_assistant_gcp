"""
Test Vertex AI RAG's native generation capabilities
Check if we can use rag.generate_answer() instead of just retrieval
"""
import vertexai
from vertexai.preview import rag
from vertexai.preview.rag import RagResource
import asyncio

# Configuration
PROJECT_ID = "tech-bharath"
LOCATION = "asia-south1"
CORPUS_ID = "projects/tech-bharath/locations/asia-south1/ragCorpora/7638104968020361216"

async def test_rag_generation():
    """Test if Vertex AI RAG supports end-to-end generation"""
    
    print("🔍 TESTING VERTEX AI RAG GENERATION CAPABILITIES")
    print("=" * 60)
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Test query
    query = "What are the guidelines for teacher recruitment in Andhra Pradesh?"
    print(f"Query: {query}")
    
    print("\n1. Testing retrieval_query (current method):")
    try:
        retrieval_response = rag.retrieval_query(
            text=query,
            rag_resources=[RagResource(rag_corpus=CORPUS_ID)],
            similarity_top_k=5
        )
        
        if hasattr(retrieval_response, 'contexts') and hasattr(retrieval_response.contexts, 'contexts'):
            contexts = list(retrieval_response.contexts.contexts)
            print(f"✅ Retrieval successful: {len(contexts)} contexts")
        else:
            print("❌ Retrieval failed: no contexts")
            
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
    
    print("\n2. Testing rag.generate_answer (end-to-end):")
    try:
        # Check if generate_answer exists
        if hasattr(rag, 'generate_answer'):
            print("✅ generate_answer method found!")
            
            # Try to use it
            generation_response = rag.generate_answer(
                text=query,
                rag_resources=[RagResource(rag_corpus=CORPUS_ID)],
                similarity_top_k=5
            )
            
            print(f"✅ Generation successful!")
            print(f"Response type: {type(generation_response)}")
            
            # Check response structure
            if hasattr(generation_response, 'answer'):
                print(f"📝 Answer: {generation_response.answer[:200]}...")
            
            if hasattr(generation_response, 'supporting_references'):
                print(f"📚 Supporting references: {len(generation_response.supporting_references)}")
                
            if hasattr(generation_response, 'answerable_probability'):
                print(f"🎯 Answerable probability: {generation_response.answerable_probability}")
            
            print(f"\n📋 Full response attributes:")
            print([attr for attr in dir(generation_response) if not attr.startswith('_')])
            
            return generation_response
            
        else:
            print("❌ generate_answer method not available")
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Checking available RAG methods:")
    rag_methods = [method for method in dir(rag) if not method.startswith('_')]
    print(f"Available methods: {rag_methods}")
    
    # Check for other generation methods
    generation_methods = [method for method in rag_methods if 'generate' in method.lower()]
    print(f"Generation-related methods: {generation_methods}")

async def test_advanced_rag_options():
    """Test advanced RAG options for better generation"""
    
    print(f"\n{'='*60}")
    print("🚀 TESTING ADVANCED RAG OPTIONS")
    print("=" * 60)
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    query = "What are the teacher transfer rules in Andhra Pradesh education department?"
    
    # Test with different parameters
    test_configs = [
        {
            "name": "Basic retrieval",
            "params": {
                "text": query,
                "rag_resources": [RagResource(rag_corpus=CORPUS_ID)],
                "similarity_top_k": 5
            }
        },
        {
            "name": "With ranking",
            "params": {
                "text": query,
                "rag_resources": [RagResource(rag_corpus=CORPUS_ID)],
                "similarity_top_k": 10,
                "ranking_config": {
                    "ranker": f"projects/{PROJECT_ID}/locations/{LOCATION}/rankingConfigs/default_ranking_config"
                }
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n🔬 Testing: {config['name']}")
        try:
            response = rag.retrieval_query(**config['params'])
            
            if hasattr(response, 'contexts') and hasattr(response.contexts, 'contexts'):
                contexts = list(response.contexts.contexts)
                print(f"✅ {len(contexts)} contexts retrieved")
                
                if contexts:
                    # Show context quality
                    top_context = contexts[0]
                    if hasattr(top_context, 'text'):
                        print(f"📝 Top context: {top_context.text[:150]}...")
                    if hasattr(top_context, 'distance'):
                        print(f"🎯 Distance: {top_context.distance}")
        except Exception as e:
            print(f"❌ Error: {e}")

async def compare_architectures():
    """Compare current architecture vs Vertex AI RAG native generation"""
    
    print(f"\n{'='*60}")
    print("📊 ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    print("""
🏗️  CURRENT ARCHITECTURE (Your System):
    Query → Enhanced Query → RAG Retrieval → Documents → Gemini Synthesis → Answer
    
    ✅ Pros:
    - Full control over prompt engineering
    - Custom citation format
    - Agent-based post-processing
    - Multi-engine fusion
    - Enhanced query optimization
    
    ⚠️  Cons:
    - More complex pipeline
    - Multiple API calls
    - Custom synthesis logic to maintain

🏗️  VERTEX AI RAG NATIVE (Alternative):
    Query → RAG Generate Answer → Answer with Citations
    
    ✅ Pros:
    - Single API call
    - Built-in citation handling
    - Optimized for retrieval-generation
    - Less complexity
    
    ⚠️  Cons:
    - Less control over prompts
    - Fixed citation format
    - No multi-engine support
    - Limited customization
    """)

if __name__ == "__main__":
    async def main():
        await test_rag_generation()
        await test_advanced_rag_options()
        await compare_architectures()
        
        print(f"\n{'🎯' * 20}")
        print("RECOMMENDATION:")
        print("Your current architecture is actually BETTER for your use case!")
        print("✅ Multi-engine support")
        print("✅ Custom agents with domain expertise") 
        print("✅ Enhanced query optimization")
        print("✅ Flexible citation formats")
        print("✅ Full control over the pipeline")
        print(f"{'🎯' * 20}")
    
    asyncio.run(main())