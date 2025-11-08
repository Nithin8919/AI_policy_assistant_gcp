"""
Planner - Creates execution plan for multi-engine retrieval
Combines query analysis and engine scoring into actionable plan
"""
from typing import Dict, List, Any
import uuid
from datetime import datetime

from router.query_analyzer import QueryAnalyzer
from router.engine_scorer import EngineScorer, select_engines, apply_forced_pairs
from config import load_config


class QueryPlanner:
    """Creates detailed execution plans for multi-engine RAG queries"""
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.scorer = EngineScorer()
        self.config = load_config()
        
    def create_plan(
        self, 
        query: str, 
        max_engines: int = None,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a complete execution plan
        Returns plan with engines, filters, budgets, and metadata
        """
        # Use config default if not specified
        if max_engines is None:
            max_engines = self.config["routing"]["max_engines"]
        
        min_score = self.config["routing"]["min_score"]
        
        # Step 1: Analyze query
        features = self.analyzer.analyze(query)
        
        # Step 2: Score engines
        engine_scores = self.scorer.score_engines(features)
        
        # Step 3: Select engines
        selected = select_engines(engine_scores, max_engines, min_score)
        
        # Step 4: Apply forced pairs if configured
        if self.config["routing"].get("force_pairs"):
            selected = apply_forced_pairs(selected, features)
        
        # Step 5: Build detailed plan
        plan = self._build_plan(
            query=query,
            features=features,
            engine_scores=dict(engine_scores),
            selected_engines=selected,
            user_context=user_context or {}
        )
        
        return plan
    
    def _build_plan(
        self,
        query: str,
        features: Dict[str, Any],
        engine_scores: Dict[str, float],
        selected_engines: List[str],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construct the full execution plan"""
        
        plan_id = str(uuid.uuid4())
        
        # Build per-engine configurations
        engine_configs = {}
        for engine_name in selected_engines:
            engine_configs[engine_name] = self._build_engine_config(
                engine_name, features
            )
        
        # Determine retrieval strategy
        parallel = self.config["routing"].get("parallel_retrieval", True)
        
        # Build rationale
        rationale = self._generate_rationale(
            selected_engines, engine_scores, features
        )
        
        plan = {
            "plan_id": plan_id,
            "query": query,
            "created_at": datetime.utcnow().isoformat(),
            
            # Analysis results
            "features": features,
            "entities": features["entities"],
            "facets": features["facets"],
            "query_type": features["query_type"],
            
            # Engine selection
            "all_engine_scores": engine_scores,
            "selected_engines": selected_engines,
            "engine_configs": engine_configs,
            
            # Execution strategy
            "parallel_retrieval": parallel,
            "max_engines": len(selected_engines),
            
            # Metadata
            "routing_rationale": rationale,
            "user_context": user_context,
            "constraints": features["constraints"],
            "temporal": features["temporal"]
        }
        
        return plan
    
    def _build_engine_config(
        self, 
        engine_name: str, 
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build configuration for a specific engine"""
        
        config = {
            "engine_name": engine_name,
            "top_k": self.config["ranking"]["top_k_per_engine"],
            "filters": {},
            "facet_hints": []
        }
        
        # Add relevant facets as hints
        engine_facets = self.config["engines"][engine_name].get("facets", [])
        query_facets = features.get("facets", [])
        relevant_facets = list(set(engine_facets) & set(query_facets))
        config["facet_hints"] = relevant_facets
        
        # Add temporal filters if present
        temporal = features.get("temporal", {})
        if temporal.get("has_temporal"):
            if temporal.get("years"):
                config["filters"]["years"] = temporal["years"]
            if temporal.get("fiscal_years"):
                config["filters"]["fiscal_years"] = temporal["fiscal_years"]
        
        # Add jurisdiction filters
        jurisdiction = features.get("jurisdiction", "Andhra Pradesh")
        config["filters"]["jurisdiction"] = jurisdiction
        
        # Add entity-specific filters
        entities = features.get("entities", {})
        
        if engine_name == "gos" and entities.get("go_numbers"):
            config["filters"]["go_numbers"] = entities["go_numbers"]
        
        if engine_name == "judicial" and entities.get("case_citations"):
            config["filters"]["case_citations"] = entities["case_citations"]
        
        if engine_name == "legal" and entities.get("legal_refs"):
            config["filters"]["legal_refs"] = entities["legal_refs"]
        
        # Add constraints
        constraints = features.get("constraints", {})
        if constraints.get("districts"):
            config["filters"]["districts"] = constraints["districts"]
        
        if constraints.get("school_types"):
            config["filters"]["school_types"] = constraints["school_types"]
        
        return config
    
    def _generate_rationale(
        self,
        selected_engines: List[str],
        engine_scores: Dict[str, float],
        features: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation for engine selection"""
        
        reasons = []
        
        # Explain primary engine
        if selected_engines:
            primary = selected_engines[0]
            primary_score = engine_scores[primary]
            reasons.append(
                f"Primary engine '{primary}' (score: {primary_score:.2f})"
            )
        
        # Explain entity matches
        entities = features.get("entities", {})
        if entities.get("legal_refs"):
            reasons.append(f"Legal references detected: {entities['legal_refs']}")
        
        if entities.get("go_numbers"):
            reasons.append(f"GO numbers mentioned: {entities['go_numbers']}")
        
        if entities.get("case_citations"):
            reasons.append(f"Case law cited: {entities['case_citations']}")
        
        # Explain facet matches
        if features.get("facets"):
            reasons.append(f"Relevant facets: {', '.join(features['facets'])}")
        
        # Explain forced pairs
        if len(selected_engines) > 1:
            reasons.append(
                f"Additional engines: {', '.join(selected_engines[1:])} "
                f"(complementary context)"
            )
        
        return "; ".join(reasons)
    
    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """Get a concise summary of the plan"""
        engines = ", ".join(plan["selected_engines"])
        query_type = plan["query_type"]
        entity_count = sum(len(v) for v in plan["entities"].values())
        
        return (
            f"Plan {plan['plan_id'][:8]}: "
            f"Query type={query_type}, "
            f"Engines=[{engines}], "
            f"Entities={entity_count}"
        )


if __name__ == "__main__":
    # Test the planner
    planner = QueryPlanner()
    
    test_queries = [
        "What are the transfer rules under GO Ms No 45?",
        "Show UDISE enrollment data for Krishna district",
        "Explain RTE Section 12(1)(c) implementation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        plan = planner.create_plan(query)
        print(planner.get_plan_summary(plan))
        print(f"Engines: {plan['selected_engines']}")
        print(f"Rationale: {plan['routing_rationale']}")