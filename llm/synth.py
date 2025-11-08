"""
Synthesis - Generate grounded answers with citations using Gemini
Enforces strict citation format and evidence-based reasoning
"""
from typing import List, Dict, Any, Tuple
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import re
from utils.logging import get_logger

logger = get_logger()


async def synthesize_answer(
    query: str,
    documents: List[Dict[str, Any]],
    features: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a grounded, citation-backed answer
    
    Args:
        query: Original user query
        documents: Final selected documents (post-fusion)
        features: Query analysis features
        config: System configuration
    
    Returns:
        Dictionary with answer, citations, confidence, etc.
    """
    if not documents:
        return {
            "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or check if the query is within the scope of available data.",
            "citations": [],
            "confidence": 0.0,
            "evidence": []
        }
    
    logger.info(f"Synthesizing answer from {len(documents)} documents")
    
    # Initialize Vertex AI
    project_id = config["project"]["gcp_project_id"]
    location = config["project"]["location"]
    vertexai.init(project=project_id, location=location)
    
    # Prepare context and prompt
    context = _build_context(documents, config)
    prompt = _build_prompt(query, context, features, config)
    
    # Generate answer
    model_name = config["models"]["llm"]
    model = GenerativeModel(model_name)
    
    generation_config = {
        "temperature": config["models"].get("temperature", 0.1),
        "max_output_tokens": config["models"].get("max_tokens", 2048),
    }
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        answer_text = response.text
        
        # Parse citations from answer
        citations = _extract_citations(answer_text, documents)
        
        # Calculate confidence
        confidence = _calculate_confidence(answer_text, citations, documents)
        
        # Extract key evidence spans
        evidence = _extract_evidence(answer_text, documents, config)
        
        logger.info(
            f"Answer generated: {len(answer_text)} chars, "
            f"{len(citations)} citations, confidence={confidence:.2f}"
        )
        
        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": confidence,
            "evidence": evidence
        }
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "evidence": []
        }


def _build_context(documents: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    """Build context string from documents"""
    context_parts = []
    
    for i, doc in enumerate(documents):
        vertical = doc.get("vertical", "unknown")
        doc_id = doc.get("id", f"doc_{i}")
        locator = doc.get("locator", "")
        text = doc.get("text", "")
        source_date = doc.get("source_date", "")
        
        # Format: [vertical:doc_id:locator] - text
        citation_tag = f"[{vertical}:{doc_id}:{locator}]"
        date_info = f" (Date: {source_date})" if source_date else ""
        
        context_part = f"{citation_tag}{date_info}\n{text}\n"
        context_parts.append(context_part)
    
    return "\n---\n".join(context_parts)


def _build_prompt(
    query: str,
    context: str,
    features: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """Build the synthesis prompt"""
    
    citation_format = config["synthesis"]["citation_format"]
    show_conflicts = config["synthesis"].get("show_conflicts", True)
    
    prompt = f"""You are a policy reasoning assistant for Andhra Pradesh government data.

Your task is to answer the following query using ONLY the provided context documents. You must:

1. **Ground every claim in evidence**: Each factual statement must be cited
2. **Use strict citation format**: {citation_format}
3. **Be precise and factual**: No speculation or information beyond the documents
4. **Highlight conflicts**: If sources disagree, state both views and identify the controlling authority
5. **Note gaps**: If information is incomplete, explicitly say so
6. **Quote key clauses**: Include short verbatim quotes (< 200 chars) from official documents

**Query**: {query}

**Query Type**: {features.get('query_type', 'general')}

**Entities Mentioned**:
{_format_entities(features.get('entities', {}))}

**Context Documents**:
{context}

**Instructions**:
- Start with a direct answer to the query
- Support each claim with citations in the format {citation_format}
- If multiple documents support a point, cite all relevant sources
- If sources conflict, explain the conflict and cite the superseding authority
- End with any important caveats or limitations

**Answer**:"""
    
    return prompt


def _format_entities(entities: Dict[str, List[str]]) -> str:
    """Format extracted entities for prompt"""
    if not entities or not any(entities.values()):
        return "None"
    
    formatted = []
    for entity_type, items in entities.items():
        if items:
            formatted.append(f"- {entity_type}: {', '.join(items)}")
    
    return "\n".join(formatted) if formatted else "None"


def _extract_citations(answer: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract citation tags from the answer
    Format: [vertical:doc_id:locator]
    """
    citation_pattern = r'\[([\w_]+):([\w_]+):([\w\d\s]*)\]'
    matches = re.findall(citation_pattern, answer)
    
    citations = []
    doc_lookup = {doc["id"]: doc for doc in documents}
    
    for vertical, doc_id, locator in matches:
        if doc_id in doc_lookup:
            doc = doc_lookup[doc_id]
            citation = {
                "vertical": vertical,
                "doc_id": doc_id,
                "locator": locator,
                "snippet": doc.get("text", "")[:200],  # First 200 chars
                "score": doc.get("rerank_score", doc.get("score", 0.0)),
                "source_date": doc.get("source_date", ""),
                "source_url": doc.get("source_uri", "")
            }
            citations.append(citation)
    
    return citations


def _calculate_confidence(
    answer: str,
    citations: List[Dict[str, Any]],
    documents: List[Dict[str, Any]]
) -> float:
    """
    Calculate confidence score for the answer
    Based on: citation coverage, source quality, evidence strength
    """
    if not answer or answer.startswith("Error"):
        return 0.0
    
    # Factor 1: Citation density (citations per 100 words)
    word_count = len(answer.split())
    citation_density = min((len(citations) / max(word_count / 100, 1)) / 3, 1.0)
    
    # Factor 2: Average source quality (rerank scores)
    if citations:
        avg_source_score = sum(c.get("score", 0.5) for c in citations) / len(citations)
    else:
        avg_source_score = 0.0
    
    # Factor 3: Document coverage (how many of top docs were cited)
    if documents and citations:
        top_docs_cited = len(set(c["doc_id"] for c in citations))
        coverage = min(top_docs_cited / min(len(documents), 5), 1.0)
    else:
        coverage = 0.0
    
    # Factor 4: Presence of hedging language (reduces confidence)
    hedge_words = ["may", "might", "possibly", "unclear", "limited information"]
    hedge_count = sum(1 for word in hedge_words if word in answer.lower())
    hedge_penalty = min(hedge_count * 0.1, 0.3)
    
    # Weighted combination
    confidence = (
        0.3 * citation_density +
        0.3 * avg_source_score +
        0.3 * coverage +
        0.1 * (1 - hedge_penalty)
    )
    
    return round(confidence, 2)


def _extract_evidence(
    answer: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract key evidence spans from documents"""
    evidence = []
    
    # Find quoted text in answer
    quote_pattern = r'"([^"]{20,200})"'
    quotes = re.findall(quote_pattern, answer)
    
    for quote in quotes:
        # Find source document
        for doc in documents:
            if quote.lower() in doc.get("text", "").lower():
                evidence.append({
                    "quote": quote,
                    "source": doc.get("citation", {}),
                    "context": doc.get("text", "")[:300]
                })
                break
    
    return evidence[:5]  # Limit to top 5 evidence spans


if __name__ == "__main__":
    # Test synthesis
    import asyncio
    from config import load_config
    
    config = load_config()
    
    async def test():
        test_docs = [
            {
                "id": "legal_doc_1",
                "vertical": "legal",
                "text": "Section 10 of the AP Education Service Rules states that teachers may request transfer after completing 3 years of service in their current location.",
                "locator": "Section 10",
                "score": 0.95,
                "source_date": "2023-01-15"
            },
            {
                "id": "go_doc_1",
                "vertical": "gos",
                "text": "GO Ms No 45 dated 2024-03-20 implements the transfer policy and specifies the application process and timeline.",
                "locator": "GO Ms No 45",
                "score": 0.90,
                "source_date": "2024-03-20"
            }
        ]
        
        test_features = {
            "query_type": "procedural",
            "entities": {
                "legal_refs": ["Section 10"],
                "go_numbers": ["GO Ms No 45"]
            }
        }
        
        result = await synthesize_answer(
            query="What are the transfer rules for teachers?",
            documents=test_docs,
            features=test_features,
            config=config
        )
        
        print("Answer:")
        print(result["answer"])
        print(f"\nConfidence: {result['confidence']}")
        print(f"Citations: {len(result['citations'])}")
    
    asyncio.run(test())