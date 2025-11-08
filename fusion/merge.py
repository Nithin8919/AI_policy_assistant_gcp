"""
Merge - Final merging and trimming of documents for synthesis
"""
from typing import List, Dict, Any
from utils.logging import get_logger

logger = get_logger()


def merge_and_trim(
    documents: List[Dict[str, Any]],
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Final merging and selection of top-k documents
    
    Args:
        documents: Reranked documents
        top_k: Number of documents to keep for synthesis
    
    Returns:
        Final set of documents with metadata
    """
    if not documents:
        return []
    
    logger.info(f"Merging and trimming to top {top_k} from {len(documents)} documents")
    
    # Take top-k
    selected = documents[:top_k]
    
    # Enrich with final metadata
    final_docs = []
    for rank, doc in enumerate(selected):
        enriched_doc = doc.copy()
        enriched_doc["final_rank"] = rank
        enriched_doc["selected"] = True
        
        # Prepare citation metadata
        enriched_doc["citation"] = {
            "vertical": doc.get("vertical", "unknown"),
            "doc_id": doc.get("id", "unknown"),
            "locator": doc.get("locator", ""),
            "score": doc.get("rerank_score", doc.get("score", 0.0)),
            "source_uri": doc.get("source_uri", ""),
            "source_date": doc.get("source_date", "")
        }
        
        final_docs.append(enriched_doc)
    
    # Log vertical distribution
    vertical_dist = {}
    for doc in final_docs:
        v = doc.get("vertical", "unknown")
        vertical_dist[v] = vertical_dist.get(v, 0) + 1
    
    logger.info(f"Final set distribution: {vertical_dist}")
    
    return final_docs


def ensure_vertical_coverage(
    documents: List[Dict[str, Any]],
    min_per_vertical: int = 1
) -> List[Dict[str, Any]]:
    """
    Ensure minimum representation from each vertical present
    
    This is useful to guarantee diverse evidence even if one engine
    dominates the ranking
    """
    # Group by vertical
    by_vertical = {}
    for doc in documents:
        v = doc.get("vertical", "unknown")
        if v not in by_vertical:
            by_vertical[v] = []
        by_vertical[v].append(doc)
    
    # Take at least min_per_vertical from each
    balanced = []
    for vertical, docs in by_vertical.items():
        count = min(min_per_vertical, len(docs))
        balanced.extend(docs[:count])
    
    # Add remaining documents up to original length
    remaining_slots = len(documents) - len(balanced)
    if remaining_slots > 0:
        # Add highest scoring docs not yet included
        included_ids = {doc["id"] for doc in balanced}
        remaining = [doc for doc in documents if doc["id"] not in included_ids]
        balanced.extend(remaining[:remaining_slots])
    
    # Re-sort by score
    balanced.sort(key=lambda d: d.get("rerank_score", d.get("score", 0.0)), reverse=True)
    
    return balanced


def prioritize_explicit_refs(
    documents: List[Dict[str, Any]],
    entities: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Boost documents that match explicit entity references in the query
    
    E.g., if query mentions "GO Ms No 45", boost docs with that GO number
    """
    if not entities:
        return documents
    
    boosted = []
    for doc in documents:
        doc_copy = doc.copy()
        boost = 0.0
        
        text_lower = doc.get("text", "").lower()
        
        # Check for GO number matches
        for go_num in entities.get("go_numbers", []):
            if go_num.lower() in text_lower:
                boost += 0.2
        
        # Check for legal ref matches
        for legal_ref in entities.get("legal_refs", []):
            if legal_ref.lower() in text_lower:
                boost += 0.2
        
        # Check for case citation matches
        for case_cite in entities.get("case_citations", []):
            if case_cite.lower() in text_lower:
                boost += 0.2
        
        # Apply boost
        if boost > 0:
            original_score = doc.get("rerank_score", doc.get("score", 0.0))
            doc_copy["boosted_score"] = original_score + boost
            doc_copy["entity_boost"] = boost
        else:
            doc_copy["boosted_score"] = doc.get("rerank_score", doc.get("score", 0.0))
            doc_copy["entity_boost"] = 0.0
        
        boosted.append(doc_copy)
    
    # Re-sort by boosted score
    boosted.sort(key=lambda d: d["boosted_score"], reverse=True)
    
    return boosted


if __name__ == "__main__":
    # Test merge
    test_docs = [
        {"id": "1", "text": "Doc 1", "score": 0.9, "vertical": "legal", "rerank_score": 0.95},
        {"id": "2", "text": "Doc 2", "score": 0.8, "vertical": "gos", "rerank_score": 0.85},
        {"id": "3", "text": "Doc 3", "score": 0.7, "vertical": "legal", "rerank_score": 0.75},
        {"id": "4", "text": "Doc 4", "score": 0.6, "vertical": "data_report", "rerank_score": 0.65},
    ]
    
    merged = merge_and_trim(test_docs, top_k=3)
    print(f"Selected {len(merged)} documents:")
    for doc in merged:
        print(f"  Rank {doc['final_rank']}: {doc['vertical']} - {doc['text']}")