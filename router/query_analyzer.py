"""
Query Analyzer - Extracts entities, constraints, and expansions from user queries
Performs NER for legal citations, GO numbers, case law, metrics, dates, etc.
"""
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from config import load_config


class QueryAnalyzer:
    """Analyzes policy queries to extract structured information"""
    
    def __init__(self):
        self.config = load_config()
        self.patterns = self.config.get("entity_extraction", {})
        self.synonyms = self.config.get("synonyms", {})
        self.temporal_keywords = self.config.get("temporal_keywords", [])
        
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Main analysis pipeline
        Returns structured information about the query
        """
        query_lower = query.lower()
        
        return {
            "original_query": query,
            "normalized_query": self._normalize(query),
            "entities": self._extract_entities(query),
            "facets": self._identify_facets(query_lower),
            "constraints": self._extract_constraints(query),
            "expansions": self._expand_query(query_lower),
            "temporal": self._extract_temporal(query),
            "jurisdiction": self._extract_jurisdiction(query),
            "query_type": self._classify_query(query_lower)
        }
    
    def _normalize(self, query: str) -> str:
        """Normalize query text"""
        # Remove extra whitespace
        query = " ".join(query.split())
        # Standardize common abbreviations
        replacements = {
            "G.O.": "GO",
            "G O": "GO",
            "Govt.": "Government",
            "Sec.": "Section",
            "Art.": "Article"
        }
        for old, new in replacements.items():
            query = query.replace(old, new)
        return query
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract legal, administrative, and domain entities"""
        entities = {
            "legal_refs": [],
            "go_numbers": [],
            "case_citations": [],
            "metrics": [],
            "schemes": []
        }
        
        # Extract legal references (Acts, Sections, Articles)
        if "legal_patterns" in self.patterns:
            for pattern in self.patterns["legal_patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities["legal_refs"].extend(matches)
        
        # Extract GO numbers
        if "go_patterns" in self.patterns:
            for pattern in self.patterns["go_patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities["go_numbers"].extend(matches)
        
        # Extract case citations
        if "case_patterns" in self.patterns:
            for pattern in self.patterns["case_patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities["case_citations"].extend(matches)
        
        # Extract metrics (UDISE, GER, ASER, NAS)
        if "metric_patterns" in self.patterns:
            for pattern in self.patterns["metric_patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    entities["metrics"].append(pattern)
        
        # Extract scheme names (keyword-based)
        scheme_keywords = ["PM POSHAN", "KGBV", "Bala badi", "scholarship", "MDM", "Samagra Shiksha"]
        for keyword in scheme_keywords:
            if keyword.lower() in query.lower():
                entities["schemes"].append(keyword)
        
        return entities
    
    def _identify_facets(self, query_lower: str) -> List[str]:
        """Identify which facets are relevant for this query"""
        facets = []
        
        # Map keywords to facets
        facet_keywords = {
            "aser": ["aser", "learning level", "reading level"],
            "budget": ["budget", "allocation", "expenditure", "fund"],
            "financial": ["financial", "audit", "accounts"],
            "nas": ["nas", "national achievement"],
            "ses": ["socio-economic", "survey", "ses"],
            "teacher_data": ["teacher", "faculty", "staff data"],
            "udise": ["udise", "school statistics", "enrollment"],
            "ap_edu": ["ap education", "state education"],
            "constitution": ["constitution", "article"],
            "ntce": ["ncte", "teacher education"],
            "rte": ["rte", "right to education", "section 12"],
            "service": ["service rule", "service regulation"],
            "transfer": ["transfer", "posting", "deployment"]
        }
        
        for facet, keywords in facet_keywords.items():
            if any(kw in query_lower for kw in keywords):
                facets.append(facet)
        
        return facets
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract filters and constraints"""
        constraints = {
            "districts": [],
            "mandals": [],
            "school_types": [],
            "date_range": None
        }
        
        # Extract district names (basic pattern)
        district_pattern = r'\b(?:in|for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+district\b'
        districts = re.findall(district_pattern, query)
        constraints["districts"] = districts
        
        # Extract school types
        school_types = ["primary", "upper primary", "secondary", "higher secondary", "high school"]
        for stype in school_types:
            if stype in query.lower():
                constraints["school_types"].append(stype)
        
        return constraints
    
    def _expand_query(self, query_lower: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expansions = []
        
        for term, synonyms_list in self.synonyms.items():
            if term in query_lower:
                expansions.extend(synonyms_list)
        
        return list(set(expansions))
    
    def _extract_temporal(self, query: str) -> Dict[str, Any]:
        """Extract temporal constraints"""
        temporal = {
            "has_temporal": False,
            "references": [],
            "date_range": None
        }
        
        # Check for temporal keywords
        for keyword in self.temporal_keywords:
            if keyword.lower() in query.lower():
                temporal["has_temporal"] = True
                temporal["references"].append(keyword)
        
        # Extract specific years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        if years:
            temporal["has_temporal"] = True
            temporal["years"] = years
        
        # Extract FY patterns
        fy_pattern = r'FY\s*(20\d{2})[-\s]*(20)?\d{2}'
        fy_matches = re.findall(fy_pattern, query, re.IGNORECASE)
        if fy_matches:
            temporal["has_temporal"] = True
            temporal["fiscal_years"] = fy_matches
        
        return temporal
    
    def _extract_jurisdiction(self, query: str) -> str:
        """Extract geographic jurisdiction"""
        # Default to Andhra Pradesh
        if "telangana" in query.lower():
            return "Telangana"
        return "Andhra Pradesh"
    
    def _classify_query(self, query_lower: str) -> str:
        """Classify the type of query"""
        if any(word in query_lower for word in ["what", "define", "explain", "meaning"]):
            return "definitional"
        elif any(word in query_lower for word in ["how many", "statistics", "data", "number"]):
            return "statistical"
        elif any(word in query_lower for word in ["when", "date", "year"]):
            return "temporal"
        elif any(word in query_lower for word in ["who", "which authority", "responsible"]):
            return "authority"
        elif any(word in query_lower for word in ["how", "process", "procedure"]):
            return "procedural"
        elif any(word in query_lower for word in ["can", "allowed", "permitted", "legal"]):
            return "legal_validity"
        else:
            return "general"


# Utility functions
def highlight_entities(query: str, entities: Dict[str, List[str]]) -> str:
    """Highlight extracted entities in the original query"""
    highlighted = query
    for entity_type, items in entities.items():
        for item in items:
            if item in highlighted:
                highlighted = highlighted.replace(item, f"**{item}**")
    return highlighted


if __name__ == "__main__":
    # Test the analyzer
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "What are the transfer rules for teachers under GO Ms No 45?",
        "Show me UDISE data for primary schools in Krishna district for FY 2023-24",
        "Explain RTE Section 12(1)(c) implementation in AP",
        "What did the Supreme Court say in AIR 2020 about teacher appointments?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = analyzer.analyze(query)
        print(f"Entities: {result['entities']}")
        print(f"Facets: {result['facets']}")
        print(f"Type: {result['query_type']}")