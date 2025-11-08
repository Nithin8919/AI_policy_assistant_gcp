"""
Legal Agent - Handles Acts, rules, constitution, education codes
Specializes in legal document analysis and statutory interpretation
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger()


@dataclass
class LegalQuery:
    """Structured representation of legal query"""
    legal_refs: List[str]  # Acts, sections, articles
    query_type: str  # constitutional, statutory, procedural, interpretive
    jurisdiction: str  # central, state, local
    subject_area: str  # education, service, transfer, etc.
    temporal_scope: Optional[str]  # current, historical, amendments
    citation_context: bool  # whether seeking specific citations


class LegalAgent:
    """Agent specialized in legal instruments and statutory analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.facets = config["engines"]["legal"]["facets"]
        self.keywords = config["engines"]["legal"]["keywords"]
        
        # Legal document hierarchy and relationships
        self.legal_hierarchy = {
            "constitution": {
                "level": 1,
                "articles": ["Article 21A", "Article 45", "Article 46", "Article 15", "Article 14"],
                "keywords": ["fundamental right", "directive principles", "equality", "education"]
            },
            "central_acts": {
                "level": 2,
                "acts": ["RTE Act 2009", "NCTE Act", "University Grants Commission Act"],
                "keywords": ["right to education", "teacher education", "university"]
            },
            "state_acts": {
                "level": 3,
                "acts": ["AP Education Act", "AP Reorganization Act", "AP Service Rules"],
                "keywords": ["state education", "service rules", "transfer policy"]
            },
            "rules_regulations": {
                "level": 4,
                "types": ["service rules", "transfer rules", "recruitment rules", "conduct rules"],
                "keywords": ["procedure", "eligibility", "process", "conditions"]
            }
        }
        
        # Legal citation patterns
        self.citation_patterns = {
            "section": r'(?:Section|Sec\.?)\s*(\d+[A-Z]?(?:\(\d+\))?)',
            "article": r'(?:Article|Art\.?)\s*(\d+[A-Z]?(?:\(\d+\))?)',
            "rule": r'(?:Rule|r\.)\s*(\d+[A-Z]?(?:\(\d+\))?)',
            "subsection": r'\((\d+)\)',
            "act_year": r'\b(19|20)\d{2}\b',
            "chapter": r'(?:Chapter|Ch\.?)\s*(\d+|[IVX]+)'
        }
        
        # Subject area mappings
        self.subject_areas = {
            "education": ["education", "school", "teacher", "student", "curriculum", "examination"],
            "service": ["service", "appointment", "promotion", "disciplinary", "retirement"],
            "transfer": ["transfer", "posting", "deployment", "cadre", "locality"],
            "constitutional": ["fundamental", "directive", "constitutional", "amendment"],
            "administrative": ["procedure", "appeal", "review", "grievance", "tribunal"]
        }
        
        logger.info(f"Initialized LegalAgent with facets: {self.facets}")
    
    def analyze_query(self, query: str, features: Dict[str, Any]) -> LegalQuery:
        """Analyze query and extract legal requirements"""
        query_lower = query.lower()
        
        # Extract legal references
        legal_refs = self._extract_legal_references(query, features)
        
        # Classify query type
        query_type = self._classify_legal_query(query_lower)
        
        # Determine jurisdiction
        jurisdiction = self._determine_jurisdiction(query_lower)
        
        # Identify subject area
        subject_area = self._identify_subject_area(query_lower)
        
        # Determine temporal scope
        temporal_scope = self._determine_temporal_scope(query, features)
        
        # Check if seeking specific citations
        citation_context = self._needs_citations(query_lower)
        
        return LegalQuery(
            legal_refs=legal_refs,
            query_type=query_type,
            jurisdiction=jurisdiction,
            subject_area=subject_area,
            temporal_scope=temporal_scope,
            citation_context=citation_context
        )
    
    def build_search_filters(self, parsed_query: LegalQuery, features: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters for RAG retrieval"""
        filters = {
            "facets": self._map_subject_to_facets(parsed_query.subject_area),
            "document_types": self._get_document_types(parsed_query.query_type),
            "jurisdiction": parsed_query.jurisdiction
        }
        
        # Add legal reference filters
        if parsed_query.legal_refs:
            filters["legal_references"] = parsed_query.legal_refs
        
        # Add temporal filters
        if parsed_query.temporal_scope and parsed_query.temporal_scope != "current":
            filters["temporal_scope"] = parsed_query.temporal_scope
        
        # Extract from features
        constraints = features.get("constraints", {})
        if constraints.get("districts"):
            filters["applicable_districts"] = constraints["districts"]
        
        return filters
    
    def enhance_query(self, original_query: str, parsed_query: LegalQuery) -> str:
        """Enhance query with legal terminology and context"""
        enhancements = []
        
        # Add hierarchical context
        if parsed_query.query_type == "constitutional":
            enhancements.extend(["fundamental rights", "directive principles", "constitution"])
        elif parsed_query.query_type == "statutory":
            enhancements.extend(["act", "statute", "legislation", "provision"])
        elif parsed_query.query_type == "procedural":
            enhancements.extend(["procedure", "process", "rules", "guidelines"])
        
        # Add subject-specific terms
        if parsed_query.subject_area in self.subject_areas:
            subject_terms = self.subject_areas[parsed_query.subject_area]
            enhancements.extend(subject_terms[:3])
        
        # Add jurisdiction context
        if parsed_query.jurisdiction == "state":
            enhancements.extend(["Andhra Pradesh", "state government", "state rules"])
        elif parsed_query.jurisdiction == "central":
            enhancements.extend(["central government", "union", "national"])
        
        # Add legal relationship terms
        if parsed_query.legal_refs:
            enhancements.extend(["as per", "under", "accordance with", "provision of"])
        
        # Build enhanced query
        if enhancements:
            return f"{original_query} {' '.join(set(enhancements[:8]))}"
        
        return original_query
    
    def postprocess_results(self, documents: List[Dict[str, Any]], parsed_query: LegalQuery) -> List[Dict[str, Any]]:
        """Post-process and prioritize results based on legal hierarchy and relevance"""
        processed_docs = []
        
        for doc in documents:
            # Score based on legal hierarchy
            hierarchy_score = self._score_legal_hierarchy(doc, parsed_query)
            
            # Score based on citation relevance
            citation_score = self._score_citation_relevance(doc, parsed_query)
            
            # Score based on subject area alignment
            subject_score = self._score_subject_alignment(doc, parsed_query)
            
            # Score based on jurisdictional relevance
            jurisdiction_score = self._score_jurisdiction_relevance(doc, parsed_query)
            
            # Composite score
            composite_score = (
                hierarchy_score * 0.3 +
                citation_score * 0.3 +
                subject_score * 0.25 +
                jurisdiction_score * 0.15
            )
            
            doc_copy = doc.copy()
            doc_copy["legal_score"] = composite_score
            doc_copy["query_type"] = parsed_query.query_type
            doc_copy["subject_area"] = parsed_query.subject_area
            doc_copy["agent"] = "legal"
            
            # Extract and highlight legal citations
            doc_copy["extracted_citations"] = self._extract_citations_from_doc(doc)
            
            processed_docs.append(doc_copy)
        
        # Sort by composite score
        processed_docs.sort(key=lambda x: x["legal_score"], reverse=True)
        
        return processed_docs
    
    def extract_legal_structure(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract legal structure and relationships from documents"""
        legal_structure = {
            "primary_sources": [],
            "secondary_sources": [],
            "citations": [],
            "cross_references": [],
            "amendments": [],
            "interpretations": []
        }
        
        for doc in documents:
            text = doc.get("text", "")
            
            # Extract citations
            citations = self._extract_all_citations(text)
            legal_structure["citations"].extend(citations)
            
            # Identify primary vs secondary sources
            if self._is_primary_source(doc):
                legal_structure["primary_sources"].append({
                    "doc_id": doc.get("id"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "type": self._identify_document_type(text)
                })
            else:
                legal_structure["secondary_sources"].append({
                    "doc_id": doc.get("id"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "type": "commentary"
                })
            
            # Extract cross-references
            cross_refs = self._extract_cross_references(text)
            legal_structure["cross_references"].extend(cross_refs)
        
        return legal_structure
    
    def _extract_legal_references(self, query: str, features: Dict[str, Any]) -> List[str]:
        """Extract legal references from query"""
        legal_refs = []
        
        # From features
        entities = features.get("entities", {})
        if entities.get("legal_refs"):
            legal_refs.extend(entities["legal_refs"])
        
        # Extract using patterns
        for ref_type, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                legal_refs.append(f"{ref_type.title()} {match}")
        
        # Extract act names
        act_pattern = r'\b([A-Z][^.]*?)\s*(?:Act|Code|Rules?)\s*(?:,?\s*(19|20)\d{2})?\b'
        act_matches = re.findall(act_pattern, query)
        for match in act_matches:
            act_name = match[0].strip()
            year = match[1] if len(match) > 1 and match[1] else ""
            if len(act_name) > 3:  # Avoid single words
                legal_refs.append(f"{act_name} Act {year}".strip())
        
        return list(set(legal_refs))
    
    def _classify_legal_query(self, query_lower: str) -> str:
        """Classify the type of legal query"""
        if any(term in query_lower for term in ["constitution", "fundamental", "directive", "article"]):
            return "constitutional"
        elif any(term in query_lower for term in ["act", "statute", "section", "provision"]):
            return "statutory"
        elif any(term in query_lower for term in ["procedure", "process", "how to", "steps"]):
            return "procedural"
        elif any(term in query_lower for term in ["interpret", "meaning", "define", "explain"]):
            return "interpretive"
        elif any(term in query_lower for term in ["valid", "legal", "permissible", "authority"]):
            return "validity"
        else:
            return "general"
    
    def _determine_jurisdiction(self, query_lower: str) -> str:
        """Determine the jurisdiction scope"""
        if any(term in query_lower for term in ["central", "union", "national", "india"]):
            return "central"
        elif any(term in query_lower for term in ["state", "andhra pradesh", "ap", "telangana"]):
            return "state"
        elif any(term in query_lower for term in ["local", "municipal", "panchayat"]):
            return "local"
        else:
            return "state"  # Default for AP context
    
    def _identify_subject_area(self, query_lower: str) -> str:
        """Identify the subject area of the legal query"""
        for subject, terms in self.subject_areas.items():
            if any(term in query_lower for term in terms):
                return subject
        return "general"
    
    def _determine_temporal_scope(self, query: str, features: Dict[str, Any]) -> Optional[str]:
        """Determine temporal scope of legal inquiry"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["current", "existing", "present", "latest"]):
            return "current"
        elif any(term in query_lower for term in ["original", "enacted", "initial"]):
            return "original"
        elif any(term in query_lower for term in ["amendment", "modified", "changed"]):
            return "amendments"
        elif any(term in query_lower for term in ["historical", "previous", "old"]):
            return "historical"
        
        # Check for specific years
        temporal = features.get("temporal", {})
        if temporal.get("years"):
            return f"year_{temporal['years'][0]}"
        
        return "current"  # Default
    
    def _needs_citations(self, query_lower: str) -> bool:
        """Check if query is seeking specific citations"""
        citation_indicators = [
            "cite", "reference", "section", "article", "provision",
            "under which", "as per", "according to", "legal basis"
        ]
        return any(indicator in query_lower for indicator in citation_indicators)
    
    def _map_subject_to_facets(self, subject_area: str) -> List[str]:
        """Map subject area to appropriate facets"""
        subject_facet_mapping = {
            "education": ["ap_edu", "rte", "constitution"],
            "service": ["service", "ap_edu"],
            "transfer": ["transfer", "service"],
            "constitutional": ["constitution"],
            "administrative": ["service", "ap_edu"]
        }
        return subject_facet_mapping.get(subject_area, ["ap_edu"])
    
    def _get_document_types(self, query_type: str) -> List[str]:
        """Get relevant document types for query"""
        type_mapping = {
            "constitutional": ["constitution", "constitutional"],
            "statutory": ["act", "statute", "legislation"],
            "procedural": ["rules", "regulations", "guidelines"],
            "interpretive": ["commentary", "interpretation", "guidelines"],
            "validity": ["act", "rules", "guidelines"]
        }
        return type_mapping.get(query_type, ["act", "rules"])
    
    def _score_legal_hierarchy(self, doc: Dict[str, Any], parsed_query: LegalQuery) -> float:
        """Score document based on legal hierarchy relevance"""
        text = doc.get("text", "").lower()
        
        # Constitutional sources get highest weight for constitutional queries
        if parsed_query.query_type == "constitutional":
            if any(term in text for term in ["constitution", "article", "fundamental"]):
                return 1.0
            return 0.3
        
        # Statutory sources for statutory queries
        elif parsed_query.query_type == "statutory":
            if any(term in text for term in ["act", "section", "statute"]):
                return 1.0
            return 0.4
        
        # Rules and regulations for procedural queries
        elif parsed_query.query_type == "procedural":
            if any(term in text for term in ["rules", "procedure", "process"]):
                return 1.0
            return 0.5
        
        return 0.5  # Neutral score
    
    def _score_citation_relevance(self, doc: Dict[str, Any], parsed_query: LegalQuery) -> float:
        """Score document based on citation relevance"""
        if not parsed_query.legal_refs:
            return 0.5  # Neutral if no specific references
        
        text = doc.get("text", "").lower()
        
        # Count matches with legal references
        matches = 0
        for legal_ref in parsed_query.legal_refs:
            # Extract key parts of the reference
            ref_lower = legal_ref.lower()
            ref_parts = ref_lower.split()
            
            # Check for partial matches
            if any(part in text for part in ref_parts if len(part) > 2):
                matches += 1
        
        if not parsed_query.legal_refs:
            return 0.5
        
        return min(matches / len(parsed_query.legal_refs), 1.0)
    
    def _score_subject_alignment(self, doc: Dict[str, Any], parsed_query: LegalQuery) -> float:
        """Score document based on subject area alignment"""
        if parsed_query.subject_area == "general":
            return 0.5
        
        text = doc.get("text", "").lower()
        subject_terms = self.subject_areas.get(parsed_query.subject_area, [])
        
        matches = sum(1 for term in subject_terms if term in text)
        return min(matches / len(subject_terms) if subject_terms else 0, 1.0)
    
    def _score_jurisdiction_relevance(self, doc: Dict[str, Any], parsed_query: LegalQuery) -> float:
        """Score document based on jurisdictional relevance"""
        text = doc.get("text", "").lower()
        metadata = doc.get("metadata", {})
        
        # Check for jurisdiction indicators
        if parsed_query.jurisdiction == "state":
            if any(term in text for term in ["andhra pradesh", "state government", "state"]):
                return 1.0
        elif parsed_query.jurisdiction == "central":
            if any(term in text for term in ["central", "union", "government of india"]):
                return 1.0
        
        # Check metadata
        if metadata.get("jurisdiction") == parsed_query.jurisdiction:
            return 1.0
        
        return 0.7  # Default relevance
    
    def _extract_citations_from_doc(self, doc: Dict[str, Any]) -> List[str]:
        """Extract legal citations from document text"""
        text = doc.get("text", "")
        citations = []
        
        # Extract various citation formats
        for ref_type, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                citations.append(f"{ref_type.title()} {match}")
        
        return list(set(citations))
    
    def _extract_all_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract all legal citations with context"""
        citations = []
        
        for ref_type, pattern in self.citation_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                context = text[max(0, start-50):min(len(text), end+50)]
                
                citations.append({
                    "type": ref_type,
                    "reference": match.group(0),
                    "context": context.strip(),
                    "position": start
                })
        
        return citations
    
    def _is_primary_source(self, doc: Dict[str, Any]) -> bool:
        """Determine if document is a primary legal source"""
        text = doc.get("text", "").lower()
        metadata = doc.get("metadata", {})
        
        # Check for primary source indicators
        primary_indicators = [
            "act", "statute", "constitution", "rules", "regulation",
            "government order", "notification", "circular"
        ]
        
        if any(indicator in text[:200] for indicator in primary_indicators):
            return True
        
        # Check metadata
        doc_type = metadata.get("document_type", "").lower()
        if doc_type in ["act", "constitution", "rules", "go", "notification"]:
            return True
        
        return False
    
    def _identify_document_type(self, text: str) -> str:
        """Identify the type of legal document"""
        text_lower = text[:200].lower()  # Check first 200 chars
        
        if "constitution" in text_lower:
            return "constitution"
        elif any(term in text_lower for term in ["act", "statute"]):
            return "act"
        elif "rules" in text_lower:
            return "rules"
        elif any(term in text_lower for term in ["go", "government order"]):
            return "government_order"
        elif "notification" in text_lower:
            return "notification"
        else:
            return "other"
    
    def _extract_cross_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract cross-references to other legal documents"""
        cross_refs = []
        
        # Pattern for "as per", "under", "in accordance with"
        ref_pattern = r'(?:as per|under|in accordance with|subject to)\s+([^.]{10,100})'
        
        for match in re.finditer(ref_pattern, text, re.IGNORECASE):
            cross_refs.append({
                "reference": match.group(1).strip(),
                "context": match.group(0),
                "position": match.start()
            })
        
        return cross_refs


if __name__ == "__main__":
    # Test the legal agent
    from config import load_config
    
    config = load_config()
    agent = LegalAgent(config)
    
    test_query = "What are the teacher transfer rules under Section 10 of AP Education Act?"
    test_features = {
        "entities": {"legal_refs": ["Section 10", "AP Education Act"]},
        "facets": ["transfer", "service"],
        "query_type": "procedural"
    }
    
    parsed = agent.analyze_query(test_query, test_features)
    print(f"Parsed query: {parsed}")
    
    filters = agent.build_search_filters(parsed, test_features)
    print(f"Search filters: {filters}")
    
    enhanced = agent.enhance_query(test_query, parsed)
    print(f"Enhanced query: {enhanced}")
