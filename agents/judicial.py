"""
Judicial Agent - Handles court judgments, orders, and case law
Specializes in judicial precedent analysis and case law interpretation
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger()


@dataclass
class JudicialQuery:
    """Structured representation of judicial query"""
    case_citations: List[str]  # specific case references
    court_level: str  # supreme, high, district, tribunal
    case_type: str  # writ, appeal, civil, criminal, administrative
    legal_issues: List[str]  # constitutional, statutory, procedural issues
    precedent_scope: str  # binding, persuasive, distinguishable
    temporal_relevance: str  # current, historical, landmark


class JudicialAgent:
    """Agent specialized in judicial decisions and case law analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keywords = config["engines"]["judicial"]["keywords"]
        
        # Court hierarchy and jurisdiction
        self.court_hierarchy = {
            "supreme_court": {
                "level": 1,
                "binding_scope": "national",
                "keywords": ["supreme court", "sc", "hon'ble supreme court"],
                "citation_patterns": [r'\b(19|20)\d{2}\s+\d+\s+SCC', r'AIR\s+(19|20)\d{2}\s+SC']
            },
            "high_court": {
                "level": 2,
                "binding_scope": "state",
                "keywords": ["high court", "hc", "andhra pradesh high court", "hyderabad high court"],
                "citation_patterns": [r'AIR\s+(19|20)\d{2}\s+AP', r'\b(19|20)\d{2}\s+\d+\s+ALD']
            },
            "district_court": {
                "level": 3,
                "binding_scope": "district",
                "keywords": ["district court", "sessions court", "judicial magistrate"],
                "citation_patterns": []
            },
            "tribunal": {
                "level": 3,
                "binding_scope": "specialized",
                "keywords": ["tribunal", "appellate tribunal", "administrative tribunal"],
                "citation_patterns": []
            }
        }
        
        # Case type patterns and characteristics
        self.case_types = {
            "writ": {
                "patterns": [r'WP\s*(?:No\.?)?\s*\d+', r'writ petition', r'habeas corpus', r'mandamus'],
                "issues": ["fundamental rights", "administrative action", "constitutional validity"],
                "remedies": ["mandamus", "certiorari", "prohibition", "quo warranto"]
            },
            "appeal": {
                "patterns": [r'(?:SLP|SCA|CrLA|CivilA)\s*(?:No\.?)?\s*\d+', r'appeal', r'revision'],
                "issues": ["procedural", "substantive law", "evidence"],
                "remedies": ["allowed", "dismissed", "remand", "set aside"]
            },
            "PIL": {
                "patterns": [r'PIL\s*(?:No\.?)?\s*\d+', r'public interest litigation'],
                "issues": ["public policy", "social justice", "environmental"],
                "remedies": ["directions", "guidelines", "monitoring"]
            },
            "service": {
                "patterns": [r'service matter', r'employment', r'transfer', r'promotion'],
                "issues": ["service law", "administrative law", "employment rights"],
                "remedies": ["reinstatement", "promotion", "transfer", "compensation"]
            }
        }
        
        # Legal principles and doctrines
        self.legal_principles = {
            "constitutional": ["fundamental rights", "separation of powers", "federal structure"],
            "administrative": ["natural justice", "procedural fairness", "reasonableness"],
            "educational": ["right to education", "academic freedom", "equal opportunity"],
            "service": ["due process", "legitimate expectation", "equality in service"]
        }
        
        logger.info(f"Initialized JudicialAgent with keywords: {self.keywords}")
    
    def analyze_query(self, query: str, features: Dict[str, Any]) -> JudicialQuery:
        """Analyze query and extract judicial requirements"""
        query_lower = query.lower()
        
        # Extract case citations
        case_citations = self._extract_case_citations(query, features)
        
        # Identify court level
        court_level = self._identify_court_level(query_lower)
        
        # Classify case type
        case_type = self._classify_case_type(query_lower)
        
        # Extract legal issues
        legal_issues = self._extract_legal_issues(query_lower)
        
        # Determine precedent scope
        precedent_scope = self._determine_precedent_scope(query_lower, court_level)
        
        # Assess temporal relevance
        temporal_relevance = self._assess_temporal_relevance(query, features)
        
        return JudicialQuery(
            case_citations=case_citations,
            court_level=court_level,
            case_type=case_type,
            legal_issues=legal_issues,
            precedent_scope=precedent_scope,
            temporal_relevance=temporal_relevance
        )
    
    def build_search_filters(self, parsed_query: JudicialQuery, features: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters for RAG retrieval"""
        filters = {
            "court_level": parsed_query.court_level,
            "case_type": parsed_query.case_type,
            "precedent_scope": parsed_query.precedent_scope
        }
        
        # Add case citation filters
        if parsed_query.case_citations:
            filters["case_citations"] = parsed_query.case_citations
        
        # Add legal issue filters
        if parsed_query.legal_issues:
            filters["legal_issues"] = parsed_query.legal_issues
        
        # Add temporal filters
        if parsed_query.temporal_relevance != "current":
            filters["temporal_relevance"] = parsed_query.temporal_relevance
        
        # Extract from features
        temporal = features.get("temporal", {})
        if temporal.get("years"):
            filters["decision_years"] = temporal["years"]
        
        return filters
    
    def enhance_query(self, original_query: str, parsed_query: JudicialQuery) -> str:
        """Enhance query with judicial terminology and case law context"""
        enhancements = []
        
        # Add court-specific terms
        if parsed_query.court_level in self.court_hierarchy:
            court_keywords = self.court_hierarchy[parsed_query.court_level]["keywords"]
            enhancements.extend(court_keywords[:2])
        
        # Add case type specific terms
        if parsed_query.case_type in self.case_types:
            case_issues = self.case_types[parsed_query.case_type]["issues"]
            enhancements.extend(case_issues[:2])
        
        # Add legal principle terms
        for issue in parsed_query.legal_issues:
            if issue in self.legal_principles:
                enhancements.extend(self.legal_principles[issue][:2])
        
        # Add precedent context
        if parsed_query.precedent_scope == "binding":
            enhancements.extend(["binding precedent", "ratio decidendi", "authoritative"])
        elif parsed_query.precedent_scope == "persuasive":
            enhancements.extend(["persuasive", "guidance", "considered"])
        
        # Add judicial terminology
        enhancements.extend(["held", "observed", "judgment", "bench", "petitioner", "respondent"])
        
        # Build enhanced query
        if enhancements:
            return f"{original_query} {' '.join(set(enhancements[:8]))}"
        
        return original_query
    
    def postprocess_results(self, documents: List[Dict[str, Any]], parsed_query: JudicialQuery) -> List[Dict[str, Any]]:
        """Post-process and prioritize results based on judicial hierarchy and precedent value"""
        processed_docs = []
        
        for doc in documents:
            # Score based on court hierarchy
            hierarchy_score = self._score_court_hierarchy(doc, parsed_query)
            
            # Score based on citation relevance
            citation_score = self._score_citation_relevance(doc, parsed_query)
            
            # Score based on legal issue alignment
            issue_score = self._score_legal_issues(doc, parsed_query)
            
            # Score based on precedent value
            precedent_score = self._score_precedent_value(doc, parsed_query)
            
            # Score based on temporal relevance
            temporal_score = self._score_temporal_relevance(doc, parsed_query)
            
            # Composite score
            composite_score = (
                hierarchy_score * 0.25 +
                citation_score * 0.25 +
                issue_score * 0.25 +
                precedent_score * 0.15 +
                temporal_score * 0.10
            )
            
            doc_copy = doc.copy()
            doc_copy["judicial_score"] = composite_score
            doc_copy["court_level"] = parsed_query.court_level
            doc_copy["case_type"] = parsed_query.case_type
            doc_copy["agent"] = "judicial"
            
            # Extract judicial elements
            doc_copy["extracted_holdings"] = self._extract_holdings(doc)
            doc_copy["extracted_ratio"] = self._extract_ratio_decidendi(doc)
            
            processed_docs.append(doc_copy)
        
        # Sort by composite score
        processed_docs.sort(key=lambda x: x["judicial_score"], reverse=True)
        
        return processed_docs
    
    def extract_case_law_structure(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract case law structure and precedent relationships"""
        case_structure = {
            "binding_precedents": [],
            "persuasive_authorities": [],
            "ratio_decidendi": [],
            "obiter_dicta": [],
            "distinguished_cases": [],
            "overruled_cases": []
        }
        
        for doc in documents:
            text = doc.get("text", "")
            
            # Extract holdings and ratios
            holdings = self._extract_holdings(doc)
            case_structure["ratio_decidendi"].extend(holdings)
            
            # Identify case relationships
            relationships = self._identify_case_relationships(text)
            case_structure["distinguished_cases"].extend(relationships.get("distinguished", []))
            case_structure["overruled_cases"].extend(relationships.get("overruled", []))
            
            # Categorize by precedent value
            court_level = self._identify_court_from_text(text)
            if court_level in ["supreme_court", "high_court"]:
                case_structure["binding_precedents"].append({
                    "doc_id": doc.get("id"),
                    "court": court_level,
                    "citation": self._extract_primary_citation(text)
                })
            else:
                case_structure["persuasive_authorities"].append({
                    "doc_id": doc.get("id"),
                    "court": court_level,
                    "citation": self._extract_primary_citation(text)
                })
        
        return case_structure
    
    def _extract_case_citations(self, query: str, features: Dict[str, Any]) -> List[str]:
        """Extract case citations from query"""
        citations = []
        
        # From features
        entities = features.get("entities", {})
        if entities.get("case_citations"):
            citations.extend(entities["case_citations"])
        
        # Extract using court-specific patterns
        for court_type, court_info in self.court_hierarchy.items():
            for pattern in court_info["citation_patterns"]:
                matches = re.findall(pattern, query, re.IGNORECASE)
                citations.extend(matches)
        
        # Extract general case patterns
        general_patterns = [
            r'\b(19|20)\d{2}\s+\d+\s+[A-Z]{2,4}\s+\d+',  # Year Volume Reporter Page
            r'AIR\s+(19|20)\d{2}\s+[A-Z]{2,4}\s+\d+',     # AIR citations
            r'WP\s*(?:No\.?)?\s*\d+(?:/\d{4})?',          # Writ Petition numbers
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _identify_court_level(self, query_lower: str) -> str:
        """Identify the court level from query"""
        for court_type, court_info in self.court_hierarchy.items():
            if any(keyword in query_lower for keyword in court_info["keywords"]):
                return court_type
        
        # Default based on case indicators
        if any(term in query_lower for term in ["writ", "constitutional", "fundamental"]):
            return "high_court"  # Typical for constitutional matters
        elif any(term in query_lower for term in ["appeal", "slp", "supreme"]):
            return "supreme_court"
        else:
            return "high_court"  # Default for AP context
    
    def _classify_case_type(self, query_lower: str) -> str:
        """Classify the type of case from query"""
        for case_type, case_info in self.case_types.items():
            if any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in case_info["patterns"]):
                return case_type
        
        # Infer from context
        if any(term in query_lower for term in ["transfer", "service", "employment"]):
            return "service"
        elif any(term in query_lower for term in ["constitutional", "rights", "violation"]):
            return "writ"
        else:
            return "general"
    
    def _extract_legal_issues(self, query_lower: str) -> List[str]:
        """Extract legal issues from query"""
        issues = []
        
        for issue_type, keywords in self.legal_principles.items():
            if any(keyword in query_lower for keyword in keywords):
                issues.append(issue_type)
        
        # Additional issue identification
        issue_indicators = {
            "procedural": ["procedure", "due process", "natural justice"],
            "substantive": ["rights", "entitlement", "validity"],
            "constitutional": ["constitutional", "fundamental", "article"],
            "administrative": ["administrative", "government action", "discretion"]
        }
        
        for issue, indicators in issue_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                if issue not in issues:
                    issues.append(issue)
        
        return issues
    
    def _determine_precedent_scope(self, query_lower: str, court_level: str) -> str:
        """Determine the precedent scope based on court and query"""
        if court_level == "supreme_court":
            return "binding"
        elif court_level == "high_court":
            return "binding"  # Within state jurisdiction
        elif any(term in query_lower for term in ["guidance", "persuasive", "consider"]):
            return "persuasive"
        else:
            return "binding"  # Default for legal precedent
    
    def _assess_temporal_relevance(self, query: str, features: Dict[str, Any]) -> str:
        """Assess temporal relevance of the case law inquiry"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["recent", "latest", "current"]):
            return "current"
        elif any(term in query_lower for term in ["landmark", "leading", "seminal"]):
            return "landmark"
        elif any(term in query_lower for term in ["historical", "evolution", "development"]):
            return "historical"
        
        # Check for specific time periods
        temporal = features.get("temporal", {})
        if temporal.get("years"):
            year = int(temporal["years"][0])
            current_year = datetime.now().year
            if current_year - year <= 5:
                return "current"
            elif current_year - year <= 20:
                return "recent"
            else:
                return "historical"
        
        return "current"  # Default
    
    def _score_court_hierarchy(self, doc: Dict[str, Any], parsed_query: JudicialQuery) -> float:
        """Score document based on court hierarchy relevance"""
        text = doc.get("text", "").lower()
        metadata = doc.get("metadata", {})
        
        # Identify court from document
        doc_court = self._identify_court_from_text(text)
        
        # Higher court decisions generally have higher precedent value
        if doc_court == parsed_query.court_level:
            return 1.0
        elif doc_court == "supreme_court":
            return 1.0  # Supreme Court decisions are always highly relevant
        elif doc_court == "high_court" and parsed_query.court_level != "supreme_court":
            return 0.8
        else:
            return 0.5
    
    def _score_citation_relevance(self, doc: Dict[str, Any], parsed_query: JudicialQuery) -> float:
        """Score document based on case citation relevance"""
        if not parsed_query.case_citations:
            return 0.5  # Neutral if no specific citations
        
        text = doc.get("text", "")
        
        # Check for direct citation matches
        matches = 0
        for citation in parsed_query.case_citations:
            # Clean and compare citations
            citation_clean = re.sub(r'\s+', ' ', citation.strip())
            if citation_clean in text:
                matches += 1
        
        if matches > 0:
            return min(matches / len(parsed_query.case_citations), 1.0)
        
        return 0.3  # Low score for no matches
    
    def _score_legal_issues(self, doc: Dict[str, Any], parsed_query: JudicialQuery) -> float:
        """Score document based on legal issue alignment"""
        if not parsed_query.legal_issues:
            return 0.5
        
        text = doc.get("text", "").lower()
        
        # Count issue matches
        matches = 0
        for issue in parsed_query.legal_issues:
            issue_keywords = self.legal_principles.get(issue, [issue])
            if any(keyword in text for keyword in issue_keywords):
                matches += 1
        
        return min(matches / len(parsed_query.legal_issues) if parsed_query.legal_issues else 0, 1.0)
    
    def _score_precedent_value(self, doc: Dict[str, Any], parsed_query: JudicialQuery) -> float:
        """Score document based on precedent value"""
        text = doc.get("text", "").lower()
        
        # Look for precedent indicators
        precedent_indicators = {
            "high": ["binding", "authoritative", "settled law", "ratio decidendi"],
            "medium": ["persuasive", "guidance", "considered", "followed"],
            "low": ["distinguished", "not applicable", "obiter"]
        }
        
        if any(indicator in text for indicator in precedent_indicators["high"]):
            return 1.0
        elif any(indicator in text for indicator in precedent_indicators["medium"]):
            return 0.7
        elif any(indicator in text for indicator in precedent_indicators["low"]):
            return 0.3
        else:
            return 0.5  # Neutral
    
    def _score_temporal_relevance(self, doc: Dict[str, Any], parsed_query: JudicialQuery) -> float:
        """Score document based on temporal relevance"""
        metadata = doc.get("metadata", {})
        doc_date = metadata.get("date", metadata.get("judgment_date", ""))
        
        if not doc_date:
            return 0.5  # Neutral if no date
        
        # Extract year from document date
        year_match = re.search(r'(19|20)\d{2}', doc_date)
        if not year_match:
            return 0.5
        
        doc_year = int(year_match.group(0))
        current_year = datetime.now().year
        
        if parsed_query.temporal_relevance == "current":
            # Prefer recent decisions
            age = current_year - doc_year
            if age <= 5:
                return 1.0
            elif age <= 10:
                return 0.8
            elif age <= 20:
                return 0.6
            else:
                return 0.4
        elif parsed_query.temporal_relevance == "landmark":
            # Landmark cases may be older but still highly relevant
            return 0.9
        elif parsed_query.temporal_relevance == "historical":
            # Historical analysis may prefer older cases
            age = current_year - doc_year
            if age >= 20:
                return 1.0
            elif age >= 10:
                return 0.8
            else:
                return 0.6
        
        return 0.5
    
    def _extract_holdings(self, doc: Dict[str, Any]) -> List[str]:
        """Extract judicial holdings from document"""
        text = doc.get("text", "")
        holdings = []
        
        # Pattern for holdings
        holding_patterns = [
            r'(?:held|observed|ruled|decided)\s*:?\s*([^.]{50,300})',
            r'(?:ratio|principle|law)\s*:?\s*([^.]{50,300})',
            r'(?:it is|we|court)\s+(?:held|find|rule)\s+(?:that\s+)?([^.]{50,300})'
        ]
        
        for pattern in holding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            holdings.extend([match.strip() for match in matches if len(match.strip()) > 20])
        
        return holdings[:5]  # Return top 5 holdings
    
    def _extract_ratio_decidendi(self, doc: Dict[str, Any]) -> List[str]:
        """Extract ratio decidendi (binding legal principle) from document"""
        text = doc.get("text", "")
        ratios = []
        
        # Look for ratio indicators
        ratio_patterns = [
            r'(?:ratio\s+decidendi|legal\s+principle|binding\s+precedent)\s*:?\s*([^.]{50,200})',
            r'(?:thus|therefore|hence)\s*,?\s*(?:the|it\s+is)\s+(?:law|held|settled)\s+(?:that\s+)?([^.]{50,200})'
        ]
        
        for pattern in ratio_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ratios.extend([match.strip() for match in matches if len(match.strip()) > 20])
        
        return ratios[:3]  # Return top 3 ratios
    
    def _identify_case_relationships(self, text: str) -> Dict[str, List[str]]:
        """Identify relationships with other cases (distinguished, overruled, etc.)"""
        relationships = {
            "distinguished": [],
            "overruled": [],
            "followed": [],
            "considered": []
        }
        
        # Patterns for different relationships
        relationship_patterns = {
            "distinguished": r'(?:distinguished|not\s+applicable)\s+([^.]{20,100})',
            "overruled": r'(?:overruled|overturned)\s+([^.]{20,100})',
            "followed": r'(?:followed|applied)\s+([^.]{20,100})',
            "considered": r'(?:considered|relied\s+upon)\s+([^.]{20,100})'
        }
        
        for relationship, pattern in relationship_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            relationships[relationship] = [match.strip() for match in matches]
        
        return relationships
    
    def _identify_court_from_text(self, text: str) -> str:
        """Identify court level from document text"""
        text_lower = text.lower()
        
        for court_type, court_info in self.court_hierarchy.items():
            if any(keyword in text_lower for keyword in court_info["keywords"]):
                return court_type
        
        return "unknown"
    
    def _extract_primary_citation(self, text: str) -> str:
        """Extract the primary citation from document text"""
        # Look for standard citation formats
        citation_patterns = [
            r'\b(19|20)\d{2}\s+\d+\s+[A-Z]{2,4}\s+\d+',
            r'AIR\s+(19|20)\d{2}\s+[A-Z]{2,4}\s+\d+',
            r'WP\s*(?:No\.?)?\s*\d+(?:/\d{4})?'
        ]
        
        for pattern in citation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "No citation found"


if __name__ == "__main__":
    # Test the judicial agent
    from config import load_config
    
    config = load_config()
    agent = JudicialAgent(config)
    
    test_query = "What did the Supreme Court say in AIR 2020 SC 1234 about teacher appointment rights?"
    test_features = {
        "entities": {"case_citations": ["AIR 2020 SC 1234"]},
        "query_type": "authority",
        "temporal": {"years": ["2020"]}
    }
    
    parsed = agent.analyze_query(test_query, test_features)
    print(f"Parsed query: {parsed}")
    
    filters = agent.build_search_filters(parsed, test_features)
    print(f"Search filters: {filters}")
    
    enhanced = agent.enhance_query(test_query, parsed)
    print(f"Enhanced query: {enhanced}")
