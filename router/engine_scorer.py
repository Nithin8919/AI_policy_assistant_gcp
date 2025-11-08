"""
Engine Scorer - Scores RAG engines based on query analysis
Uses semantic similarity, rules, and entity overlap to determine relevance
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from config import load_config, get_all_facets


class EngineScorer:
    """Scores RAG engines for a given analyzed query"""
    
    def __init__(self):
        self.config = load_config()
        self.engines = self.config["engines"]
        self.engine_facets = get_all_facets()
        
    def score_engines(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Score all engines based on query features
        Returns sorted list of (engine_name, score) tuples
        """
        scores = {}
        
        for engine_name in self.engines.keys():
            score = self._compute_engine_score(engine_name, features)
            scores[engine_name] = score
        
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_scores
    
    def _compute_engine_score(self, engine_name: str, features: Dict[str, Any]) -> float:
        """
        Compute score for a single engine using multiple signals
        Score = base_weight + facet_match + entity_boost + recency_boost + rule_bonus
        """
        engine_config = self.engines[engine_name]
        
        # Start with base weight from config
        score = engine_config["weight"]
        
        # Facet matching score
        facet_score = self._score_facet_match(engine_name, features.get("facets", []))
        score += facet_score * 0.3
        
        # Entity-based boosting
        entity_boost = self._score_entity_overlap(engine_name, features.get("entities", {}))
        score += entity_boost * 0.2
        
        # Recency boost for temporal queries
        if features.get("temporal", {}).get("has_temporal"):
            recency_boost = self._score_recency(engine_name, features)
            score += recency_boost * 0.15
        
        # Rule-based bonuses
        rule_bonus = self._apply_rules(engine_name, features)
        score += rule_bonus
        
        # Normalize to 0-1 range
        return min(max(score, 0.0), 1.0)
    
    def _score_facet_match(self, engine_name: str, query_facets: List[str]) -> float:
        """Score based on facet overlap"""
        if not query_facets:
            return 0.0
        
        engine_facets = self.engine_facets.get(engine_name, [])
        
        if not engine_facets:
            # Engines without facets get a small base score
            return 0.1
        
        # Compute Jaccard similarity
        intersection = set(query_facets) & set(engine_facets)
        union = set(query_facets) | set(engine_facets)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _score_entity_overlap(self, engine_name: str, entities: Dict[str, List[str]]) -> float:
        """Boost score based on entity types relevant to engine"""
        entity_relevance = {
            "legal": ["legal_refs", "case_citations"],
            "judicial": ["case_citations", "legal_refs"],
            "gos": ["go_numbers"],
            "data_report": ["metrics"],
            "schemes": ["schemes"]
        }
        
        relevant_entity_types = entity_relevance.get(engine_name, [])
        
        if not relevant_entity_types:
            return 0.0
        
        # Count how many relevant entities are present
        entity_count = sum(len(entities.get(et, [])) for et in relevant_entity_types)
        
        # Normalize (diminishing returns)
        return min(entity_count * 0.2, 0.5)
    
    def _score_recency(self, engine_name: str, features: Dict[str, Any]) -> float:
        """Boost engines that typically have recent data for temporal queries"""
        # GOs and Data Reports are typically more time-sensitive
        recency_engines = {"gos": 0.3, "data_report": 0.25, "schemes": 0.2}
        
        return recency_engines.get(engine_name, 0.0)
    
    def _apply_rules(self, engine_name: str, features: Dict[str, Any]) -> float:
        """Apply domain-specific rules for scoring"""
        bonus = 0.0
        
        entities = features.get("entities", {})
        query_type = features.get("query_type", "general")
        query_lower = features.get("normalized_query", "").lower()
        
        # Rule 1: Legal queries with sections/acts strongly favor legal + gos
        if entities.get("legal_refs") and engine_name in ["legal", "gos"]:
            bonus += 0.3
        
        # Rule 2: Transfer queries need legal + gos
        if "transfer" in query_lower and engine_name in ["legal", "gos"]:
            bonus += 0.25
        
        # Rule 3: Case citations need judicial + legal
        if entities.get("case_citations") and engine_name in ["judicial", "legal"]:
            bonus += 0.3
        
        # Rule 4: Statistical queries need data_report
        if query_type == "statistical" and engine_name == "data_report":
            bonus += 0.25
        
        # Rule 5: GO numbers explicitly mentioned
        if entities.get("go_numbers"):
            if engine_name == "gos":
                bonus += 0.4  # Strong signal
            elif engine_name == "legal":
                bonus += 0.15  # Supporting context
        
        # Rule 6: Scheme queries
        if entities.get("schemes") and engine_name in ["schemes", "gos"]:
            bonus += 0.2
        
        # Rule 7: RTE/Constitutional queries
        if any(term in query_lower for term in ["rte", "right to education", "article", "constitution"]):
            if engine_name == "legal":
                bonus += 0.3
        
        # Rule 8: Service rules
        if "service" in query_lower and engine_name in ["legal", "gos"]:
            bonus += 0.25
        
        return bonus


def select_engines(
    scores: List[Tuple[str, float]], 
    max_engines: int = 3, 
    min_score: float = 0.25
) -> List[str]:
    """
    Select top engines based on scores and constraints
    Returns list of engine names
    """
    # Filter by minimum score
    qualified = [(name, score) for name, score in scores if score >= min_score]
    
    # Take top N
    selected = qualified[:max_engines]
    
    return [name for name, _ in selected]


def apply_forced_pairs(
    selected: List[str], 
    features: Dict[str, Any]
) -> List[str]:
    """
    Add forced engine pairs based on configuration and query features
    E.g., if 'legal' is selected and query has GO numbers, force 'gos' as well
    """
    config = load_config()
    force_pairs = config["routing"].get("force_pairs", [])
    
    additional = set()
    
    for pair in force_pairs:
        engine_a, engine_b = pair
        
        # If one is selected, consider adding the other
        if engine_a in selected and engine_b not in selected:
            # Check if query features justify the pairing
            if _should_force_pair(engine_b, features):
                additional.add(engine_b)
        
        elif engine_b in selected and engine_a not in selected:
            if _should_force_pair(engine_a, features):
                additional.add(engine_a)
    
    return list(set(selected) | additional)


def _should_force_pair(engine: str, features: Dict[str, Any]) -> bool:
    """Determine if forced pairing is justified by query features"""
    entities = features.get("entities", {})
    
    # Force 'gos' if legal query with GO references
    if engine == "gos" and entities.get("go_numbers"):
        return True
    
    # Force 'legal' if judicial query with legal refs
    if engine == "legal" and entities.get("case_citations"):
        return True
    
    # Force 'gos' if schemes query
    if engine == "gos" and entities.get("schemes"):
        return True
    
    return False


if __name__ == "__main__":
    # Test engine scoring
    scorer = EngineScorer()
    
    test_features = {
        "facets": ["transfer", "service"],
        "entities": {
            "legal_refs": ["Section 10"],
            "go_numbers": ["GO Ms No 45"]
        },
        "query_type": "procedural",
        "normalized_query": "What are the transfer rules for teachers?",
        "temporal": {"has_temporal": False}
    }
    
    scores = scorer.score_engines(test_features)
    print("\nEngine Scores:")
    for engine, score in scores:
        print(f"  {engine}: {score:.3f}")
    
    selected = select_engines(scores, max_engines=3, min_score=0.25)
    print(f"\nSelected engines: {selected}")
    
    with_pairs = apply_forced_pairs(selected, test_features)
    print(f"After forced pairs: {with_pairs}")