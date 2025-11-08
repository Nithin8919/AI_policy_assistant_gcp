"""
LLM-powered Query Enhancement for RAG Retrieval
Optimizes queries using Gemini to improve document retrieval
"""
from typing import Dict, List, Any, Tuple
import vertexai
from vertexai.generative_models import GenerativeModel
import asyncio
from utils.logging import get_logger

logger = get_logger()


class LLMQueryEnhancer:
    """LLM-powered query enhancement for better RAG retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_id = config["project"]["gcp_project_id"]
        self.location = config["project"]["location"]
        self.model_name = config.get("models", {}).get("llm", "gemini-2.5-flash")
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
        
        logger.info(f"Initialized LLM Query Enhancer with {self.model_name}")
    
    async def enhance_query_for_rag(
        self,
        original_query: str,
        agent_type: str,
        parsed_query_info: Dict[str, Any] = None,
        corpus_context: str = "Andhra Pradesh education policy documents"
    ) -> Dict[str, Any]:
        """
        Use LLM to enhance query for better RAG retrieval
        
        Args:
            original_query: User's original query
            agent_type: Type of agent (education, schemes, judicial, etc.)
            parsed_query_info: Structured info from agent analysis
            corpus_context: Description of the document corpus
        
        Returns:
            Dict with enhanced query and reasoning
        """
        
        # Build enhancement prompt
        prompt = self._build_enhancement_prompt(
            original_query, agent_type, parsed_query_info, corpus_context
        )
        
        try:
            # Get enhancement from LLM
            loop = asyncio.get_event_loop()
            
            def sync_generate():
                response = self.model.generate_content(prompt)
                return response.text
            
            enhanced_response = await loop.run_in_executor(None, sync_generate)
            
            # Parse the response
            enhancement_result = self._parse_enhancement_response(enhanced_response)
            
            logger.info(
                f"Enhanced query '{original_query[:30]}...' -> "
                f"'{enhancement_result['enhanced_query'][:50]}...'"
            )
            
            return enhancement_result
            
        except Exception as e:
            logger.error(f"LLM query enhancement failed: {e}")
            # Fallback to original query
            return {
                "enhanced_query": original_query,
                "search_terms": [original_query],
                "reasoning": "LLM enhancement failed, using original query",
                "confidence": 0.5
            }
    
    def _build_enhancement_prompt(
        self,
        query: str,
        agent_type: str,
        parsed_info: Dict[str, Any],
        corpus_context: str
    ) -> str:
        """Build the enhancement prompt for the LLM"""
        
        # Agent-specific context
        agent_contexts = {
            "education": "educational policies, curriculum guidelines, teacher regulations, student affairs, academic standards",
            "schemes": "government schemes, welfare programs, scholarships, benefits, eligibility criteria, application processes",
            "judicial": "court judgments, legal precedents, case law, judicial orders, legal principles",
            "legal": "acts, rules, constitutional provisions, legal statutes, regulatory frameworks",
            "data_report": "educational statistics, enrollment data, performance metrics, budget allocations, infrastructure data"
        }
        
        agent_context = agent_contexts.get(agent_type, "government policy documents")
        
        # Build structured info context
        parsed_context = ""
        if parsed_info:
            parsed_context = f"""
Structured Analysis:
- Education Level: {parsed_info.get('education_level', 'N/A')}
- Subject Area: {parsed_info.get('subject_area', 'N/A')}
- Policy Type: {parsed_info.get('policy_type', 'N/A')}
- Stakeholder: {parsed_info.get('stakeholder', 'N/A')}
- Scope: {parsed_info.get('regulation_scope', 'N/A')}
"""
        
        prompt = f"""You are an expert at optimizing search queries for document retrieval from {corpus_context}.

TASK: Enhance the following query for better document retrieval from a corpus containing {agent_context}.

ORIGINAL QUERY: "{query}"

AGENT TYPE: {agent_type}

{parsed_context}

INSTRUCTIONS:
1. Analyze the intent and information need
2. Identify key concepts that should be emphasized
3. Add relevant synonyms and related terms commonly found in official documents
4. Include domain-specific terminology used in {agent_type} documents
5. Remove overly specific terms that might limit results
6. Ensure the enhanced query would match document language and style

IMPORTANT GUIDELINES:
- Keep the enhanced query natural and readable
- Focus on terms likely to appear in official documents
- Include both specific and general terms
- Consider how government documents are typically written
- Add terms that provide context without being too verbose

FORMAT YOUR RESPONSE AS:

ENHANCED QUERY: [Your enhanced query here]

SEARCH TERMS: [List 3-5 key search terms separated by commas]

REASONING: [Brief explanation of why this enhancement will improve retrieval]

CONFIDENCE: [High/Medium/Low confidence in this enhancement]"""
        
        return prompt
    
    def _parse_enhancement_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM enhancement response"""
        try:
            lines = response.strip().split('\n')
            result = {
                "enhanced_query": "",
                "search_terms": [],
                "reasoning": "",
                "confidence": 0.5
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if line.startswith("ENHANCED QUERY:"):
                    result["enhanced_query"] = line.replace("ENHANCED QUERY:", "").strip()
                elif line.startswith("SEARCH TERMS:"):
                    terms = line.replace("SEARCH TERMS:", "").strip()
                    result["search_terms"] = [t.strip() for t in terms.split(',') if t.strip()]
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.replace("REASONING:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    conf_text = line.replace("CONFIDENCE:", "").strip().lower()
                    if "high" in conf_text:
                        result["confidence"] = 0.9
                    elif "medium" in conf_text:
                        result["confidence"] = 0.7
                    else:
                        result["confidence"] = 0.5
            
            # Fallback if parsing failed
            if not result["enhanced_query"]:
                result["enhanced_query"] = response.strip()
                result["reasoning"] = "Used raw LLM response"
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing enhancement response: {e}")
            return {
                "enhanced_query": response.strip(),
                "search_terms": [],
                "reasoning": "Parsing failed, using raw response",
                "confidence": 0.3
            }
    
    async def batch_enhance_queries(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance multiple queries in parallel"""
        
        tasks = [
            self.enhance_query_for_rag(
                original_query=q["query"],
                agent_type=q.get("agent_type", "education"),
                parsed_query_info=q.get("parsed_info"),
                corpus_context=q.get("corpus_context", "Andhra Pradesh education policy documents")
            )
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Enhancement failed for query {i}: {result}")
                # Fallback
                final_results.append({
                    "enhanced_query": queries[i]["query"],
                    "search_terms": [queries[i]["query"]],
                    "reasoning": f"Enhancement failed: {result}",
                    "confidence": 0.1
                })
            else:
                final_results.append(result)
        
        return final_results


async def enhance_agent_query(
    query: str,
    agent,  # Education, Schemes, Judicial, etc. agent
    features: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced query pipeline combining agent analysis + LLM optimization
    
    Args:
        query: Original query
        agent: Specialized agent (EducationAgent, etc.)
        features: Query analysis features
        config: System configuration
    
    Returns:
        Tuple of (enhanced_query, enhancement_info)
    """
    
    # Step 1: Agent analysis
    parsed_query = agent.analyze_query(query, features)
    
    # Step 2: Agent enhancement (existing method)
    agent_enhanced = agent.enhance_query(query, parsed_query)
    
    # Step 3: LLM optimization
    enhancer = LLMQueryEnhancer(config)
    
    # Convert parsed query to dict for LLM
    parsed_info = {
        "education_level": getattr(parsed_query, 'education_level', 'N/A'),
        "subject_area": getattr(parsed_query, 'subject_area', 'N/A'),
        "stakeholder": getattr(parsed_query, 'stakeholder', 'N/A'),
        "policy_type": getattr(parsed_query, 'policy_type', 'N/A'),
        "regulation_scope": getattr(parsed_query, 'regulation_scope', 'N/A'),
        "scheme_category": getattr(parsed_query, 'scheme_category', 'N/A'),
        "benefit_type": getattr(parsed_query, 'benefit_type', 'N/A')
    }
    
    # Get agent type from class name
    agent_type = agent.__class__.__name__.lower().replace('agent', '')
    
    llm_result = await enhancer.enhance_query_for_rag(
        original_query=query,
        agent_type=agent_type,
        parsed_query_info=parsed_info
    )
    
    # Combine enhancements
    final_enhanced = llm_result["enhanced_query"]
    
    enhancement_info = {
        "original_query": query,
        "agent_enhanced": agent_enhanced,
        "llm_enhanced": final_enhanced,
        "parsed_query": parsed_query,
        "llm_reasoning": llm_result["reasoning"],
        "llm_confidence": llm_result["confidence"],
        "search_terms": llm_result["search_terms"]
    }
    
    return final_enhanced, enhancement_info


if __name__ == "__main__":
    # Test the enhancer
    from config import load_config
    from agents.education import EducationAgent
    
    async def test():
        config = load_config()
        
        # Test with education agent
        agent = EducationAgent(config)
        features = {"entities": {}, "query_type": "policy"}
        
        test_queries = [
            "teacher transfer rules",
            "student scholarship eligibility",
            "primary school curriculum guidelines"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing: {query}")
            print(f"{'='*60}")
            
            enhanced, info = await enhance_agent_query(query, agent, features, config)
            
            print(f"Original: {info['original_query']}")
            print(f"Agent Enhanced: {info['agent_enhanced']}")
            print(f"LLM Enhanced: {info['llm_enhanced']}")
            print(f"Reasoning: {info['llm_reasoning']}")
            print(f"Confidence: {info['llm_confidence']}")
            print(f"Search Terms: {', '.join(info['search_terms'])}")
    
    asyncio.run(test())