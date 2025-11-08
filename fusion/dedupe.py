"""
Deduplication - Remove duplicate and near-duplicate documents
Uses both exact matching (URL/hash) and semantic similarity
"""
from typing import List, Dict, Any, Set
import hashlib
from collections import defaultdict
from utils.logging import get_logger

logger = get_logger()


def deduplicate_docs(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicates using multiple strategies
    
    1. Exact duplicates: same source URI
    2. Content duplicates: same content hash
    3. Near-duplicates: high semantic similarity (optional, can be expensive)
    
    Args:
        documents: List of documents from multiple engines
    
    Returns:
        Deduplicated list of documents
    """
    if not documents:
        return []
    
    logger.info(f"Deduplicating {len(documents)} documents")
    
    # Strategy 1: Remove exact URL duplicates (keep highest score)
    url_groups = defaultdict(list)
    for doc in documents:
        url = doc.get("source_uri", "")
        if url:
            url_groups[url].append(doc)
        else:
            # No URL, keep it for now
            url_groups[doc.get("id", "no_url")].append(doc)
    
    # Keep best scoring document from each URL group
    deduped_by_url = []
    for url, docs_group in url_groups.items():
        best_doc = max(docs_group, key=lambda d: d.get("score", 0.0))
        deduped_by_url.append(best_doc)
    
    logger.info(f"After URL dedup: {len(deduped_by_url)} documents")
    
    # Strategy 2: Remove content hash duplicates
    deduped_by_content = _dedupe_by_content_hash(deduped_by_url)
    
    logger.info(f"After content dedup: {len(deduped_by_content)} documents")
    
    # Strategy 3: Optional - semantic near-duplicate removal
    # This is expensive, only use if you have compute budget
    # deduped_final = _dedupe_by_semantic_similarity(deduped_by_content)
    
    return deduped_by_content


def _dedupe_by_content_hash(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove documents with identical content using hash"""
    seen_hashes: Set[str] = set()
    deduped = []
    
    for doc in documents:
        content = doc.get("text", "")
        if not content:
            # Keep documents without text content
            deduped.append(doc)
            continue
        
        # Compute content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduped.append(doc)
    
    return deduped


def _dedupe_by_semantic_similarity(
    documents: List[Dict[str, Any]],
    threshold: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicates using semantic similarity
    
    This is optional and computationally expensive.
    Only use if you have the budget and embedding model available.
    
    Args:
        documents: List of documents
        threshold: Cosine similarity threshold (0.95 = very similar)
    
    Returns:
        Deduplicated documents
    """
    # TODO: Implement if needed using embeddings
    # For now, just return input
    logger.warning("Semantic deduplication not implemented, skipping")
    return documents


def merge_duplicate_sources(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge documents from same source but different sections/pages
    Useful for combining evidence from different parts of the same document
    """
    source_groups = defaultdict(list)
    
    for doc in documents:
        # Group by source URI base (without page/section)
        source = doc.get("source_uri", "").split("#")[0]
        source_groups[source].append(doc)
    
    merged = []
    for source, docs_group in source_groups.items():
        if len(docs_group) == 1:
            # Single document from this source
            merged.append(docs_group[0])
        else:
            # Multiple snippets from same source
            # Keep them separate but mark as related
            for doc in docs_group:
                doc["related_count"] = len(docs_group) - 1
                merged.append(doc)
    
    return merged


if __name__ == "__main__":
    # Test deduplication
    test_docs = [
        {
            "id": "1",
            "source_uri": "gs://bucket/doc1.pdf",
            "text": "Teacher transfer rules",
            "score": 0.9,
            "vertical": "legal"
        },
        {
            "id": "2",
            "source_uri": "gs://bucket/doc1.pdf",  # Duplicate URL
            "text": "Teacher transfer rules",
            "score": 0.85,
            "vertical": "gos"
        },
        {
            "id": "3",
            "source_uri": "gs://bucket/doc2.pdf",
            "text": "Teacher transfer rules",  # Duplicate content
            "score": 0.8,
            "vertical": "legal"
        },
        {
            "id": "4",
            "source_uri": "gs://bucket/doc3.pdf",
            "text": "Different content about budgets",
            "score": 0.7,
            "vertical": "data_report"
        }
    ]
    
    deduped = deduplicate_docs(test_docs)
    print(f"Original: {len(test_docs)} documents")
    print(f"After dedup: {len(deduped)} documents")
    for doc in deduped:
        print(f"  - {doc['id']}: {doc['text'][:30]}...")