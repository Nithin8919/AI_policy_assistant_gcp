"""
Schemes Agent - Handles government schemes, welfare programs, and benefits
Specializes in scheme eligibility, application processes, and benefit disbursement
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger()


@dataclass
class SchemeQuery:
    """Structured representation of scheme query"""
    scheme_category: str  # education, welfare, employment, infrastructure
    beneficiary_type: str  # students, teachers, schools, general_public
    scheme_scope: str  # central, state, district, local
    benefit_type: str  # financial, material, service, infrastructure
    eligibility_focus: str  # economic, social, geographic, academic
    application_stage: str  # information, eligibility, application, status


class SchemesAgent:
    """Agent specialized in government schemes, welfare programs, and benefits"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keywords = config["engines"]["schemes"]["keywords"]
        
        # Scheme categories and classifications
        self.scheme_categories = {
            "education": {
                "keywords": ["scholarship", "education", "student", "school", "books", "uniform", "fee"],
                "subcategories": ["scholarships", "infrastructure", "teacher welfare", "student support"],
                "common_schemes": ["jagananna vidya deevena", "fee reimbursement", "textbook distribution"]
            },
            "welfare": {
                "keywords": ["welfare", "pension", "assistance", "support", "subsidy", "benefit"],
                "subcategories": ["social security", "housing", "healthcare", "food security"],
                "common_schemes": ["pension schemes", "housing schemes", "healthcare schemes"]
            },
            "employment": {
                "keywords": ["employment", "job", "skill", "training", "livelihood", "self help"],
                "subcategories": ["skill development", "employment generation", "entrepreneurship"],
                "common_schemes": ["mgnrega", "skill development", "startup schemes"]
            },
            "infrastructure": {
                "keywords": ["infrastructure", "development", "construction", "facility", "building"],
                "subcategories": ["roads", "water supply", "electricity", "digital infrastructure"],
                "common_schemes": ["infrastructure development", "digital connectivity", "facility upgrades"]
            },
            "agriculture": {
                "keywords": ["agriculture", "farmer", "crop", "irrigation", "subsidy", "kisan"],
                "subcategories": ["crop insurance", "subsidies", "irrigation", "market support"],
                "common_schemes": ["pm kisan", "crop insurance", "irrigation schemes"]
            }
        }
        
        # Beneficiary types and characteristics
        self.beneficiary_types = {
            "students": {
                "keywords": ["student", "pupil", "learner", "scholar", "candidate"],
                "categories": ["sc/st", "obc", "minority", "ews", "general", "differently abled"],
                "education_levels": ["primary", "secondary", "higher", "professional"],
                "eligibility_factors": ["income", "category", "academic performance", "attendance"]
            },
            "teachers": {
                "keywords": ["teacher", "faculty", "instructor", "educator", "staff"],
                "categories": ["government", "private", "contract", "retired"],
                "benefits": ["salary", "pension", "training", "housing", "medical"],
                "eligibility_factors": ["service period", "qualification", "performance"]
            },
            "schools": {
                "keywords": ["school", "institution", "college", "university", "educational institute"],
                "categories": ["government", "aided", "private", "residential"],
                "benefits": ["grants", "infrastructure", "equipment", "maintenance"],
                "eligibility_factors": ["recognition", "enrollment", "performance", "location"]
            },
            "general_public": {
                "keywords": ["citizen", "public", "people", "community", "beneficiary"],
                "categories": ["bpl", "apl", "rural", "urban", "tribal", "women"],
                "benefits": ["financial assistance", "subsidies", "services", "facilities"],
                "eligibility_factors": ["income", "residence", "age", "family size"]
            }
        }
        
        # Scheme scope and implementing agencies
        self.scheme_scopes = {
            "central": {
                "implementing_agencies": ["ministry", "central government", "government of india"],
                "coverage": "nationwide",
                "keywords": ["central", "national", "all india", "pm", "pradhan mantri", "ministry"],
                "funding": "central funds"
            },
            "state": {
                "implementing_agencies": ["state government", "andhra pradesh", "ap government"],
                "coverage": "state-wide",
                "keywords": ["state", "andhra pradesh", "ap", "chief minister", "jagananna", "amma"],
                "funding": "state funds"
            },
            "district": {
                "implementing_agencies": ["district collector", "zilla panchayat", "district administration"],
                "coverage": "district-level",
                "keywords": ["district", "collector", "zp", "mandal", "local"],
                "funding": "district funds"
            },
            "local": {
                "implementing_agencies": ["gram panchayat", "municipality", "local body"],
                "coverage": "local-level",
                "keywords": ["village", "gram panchayat", "municipality", "ward", "local"],
                "funding": "local funds"
            }
        }
        
        # Benefit types and characteristics
        self.benefit_types = {
            "financial": {
                "patterns": [r'amount', r'money', r'cash', r'rupees', r'payment', r'reimbursement'],
                "forms": ["direct cash transfer", "reimbursement", "subsidy", "scholarship"],
                "disbursement": ["bank transfer", "check", "cash", "digital payment"]
            },
            "material": {
                "patterns": [r'material', r'goods', r'items', r'kit', r'supplies', r'equipment'],
                "forms": ["textbooks", "uniforms", "bicycles", "laptops", "food grains"],
                "distribution": ["schools", "centers", "door-to-door", "collection points"]
            },
            "service": {
                "patterns": [r'service', r'facility', r'access', r'provision', r'delivery'],
                "forms": ["healthcare", "education", "training", "counseling", "guidance"],
                "delivery": ["centers", "online", "mobile units", "institutions"]
            },
            "infrastructure": {
                "patterns": [r'building', r'construction', r'facility', r'infrastructure', r'development'],
                "forms": ["school buildings", "hostels", "laboratories", "libraries", "digital infrastructure"],
                "implementation": ["contractors", "departments", "agencies", "committees"]
            }
        }
        
        # Eligibility criteria patterns
        self.eligibility_patterns = {
            "economic": {
                "indicators": ["income", "bpl", "apl", "family income", "annual income", "poverty"],
                "thresholds": ["below poverty line", "economically weaker", "low income", "annual income"]
            },
            "social": {
                "indicators": ["caste", "category", "sc", "st", "obc", "minority", "gender"],
                "categories": ["scheduled caste", "scheduled tribe", "other backward class", "minority"]
            },
            "geographic": {
                "indicators": ["rural", "urban", "tribal", "district", "mandal", "village"],
                "locations": ["rural areas", "tribal areas", "backward districts", "remote areas"]
            },
            "academic": {
                "indicators": ["marks", "percentage", "grade", "performance", "attendance"],
                "criteria": ["minimum marks", "academic performance", "regular attendance"]
            }
        }
        
        logger.info(f"Initialized SchemesAgent with keywords: {self.keywords}")
    
    def analyze_query(self, query: str, features: Dict[str, Any]) -> SchemeQuery:
        """Analyze query and extract scheme requirements"""
        query_lower = query.lower()
        
        # Identify scheme category
        scheme_category = self._identify_scheme_category(query_lower)
        
        # Identify beneficiary type
        beneficiary_type = self._identify_beneficiary_type(query_lower)
        
        # Determine scheme scope
        scheme_scope = self._determine_scheme_scope(query_lower)
        
        # Classify benefit type
        benefit_type = self._classify_benefit_type(query_lower)
        
        # Identify eligibility focus
        eligibility_focus = self._identify_eligibility_focus(query_lower)
        
        # Determine application stage
        application_stage = self._determine_application_stage(query_lower, features)
        
        return SchemeQuery(
            scheme_category=scheme_category,
            beneficiary_type=beneficiary_type,
            scheme_scope=scheme_scope,
            benefit_type=benefit_type,
            eligibility_focus=eligibility_focus,
            application_stage=application_stage
        )
    
    def build_search_filters(self, parsed_query: SchemeQuery, features: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters for RAG retrieval"""
        filters = {
            "scheme_category": parsed_query.scheme_category,
            "beneficiary_type": parsed_query.beneficiary_type,
            "scheme_scope": parsed_query.scheme_scope,
            "benefit_type": parsed_query.benefit_type,
            "eligibility_focus": parsed_query.eligibility_focus
        }
        
        # Add application stage if specific
        if parsed_query.application_stage != "information":
            filters["application_stage"] = parsed_query.application_stage
        
        # Extract entity-based filters
        entities = features.get("entities", {})
        if entities.get("scheme_names"):
            filters["scheme_names"] = entities["scheme_names"]
        if entities.get("government_departments"):
            filters["departments"] = entities["government_departments"]
        
        # Add temporal filters
        temporal = features.get("temporal", {})
        if temporal.get("years"):
            filters["scheme_years"] = temporal["years"]
        
        return filters
    
    def enhance_query(self, original_query: str, parsed_query: SchemeQuery) -> str:
        """Enhance query with scheme terminology and benefit context"""
        enhancements = []
        
        # Add category-specific terms
        if parsed_query.scheme_category in self.scheme_categories:
            category_keywords = self.scheme_categories[parsed_query.scheme_category]["keywords"]
            enhancements.extend(category_keywords[:3])
        
        # Add beneficiary-specific terms
        if parsed_query.beneficiary_type in self.beneficiary_types:
            beneficiary_keywords = self.beneficiary_types[parsed_query.beneficiary_type]["keywords"]
            enhancements.extend(beneficiary_keywords[:2])
        
        # Add scope-specific terms
        if parsed_query.scheme_scope in self.scheme_scopes:
            scope_keywords = self.scheme_scopes[parsed_query.scheme_scope]["keywords"]
            enhancements.extend(scope_keywords[:2])
        
        # Add benefit-specific terms
        if parsed_query.benefit_type in self.benefit_types:
            benefit_forms = self.benefit_types[parsed_query.benefit_type]["forms"]
            enhancements.extend(benefit_forms[:2])
        
        # Add application stage terms
        stage_terms = {
            "eligibility": ["eligibility", "criteria", "qualification", "requirements"],
            "application": ["application", "form", "procedure", "process", "how to apply"],
            "status": ["status", "tracking", "progress", "disbursement", "payment"]
        }
        if parsed_query.application_stage in stage_terms:
            enhancements.extend(stage_terms[parsed_query.application_stage][:2])
        
        # Add general scheme terminology
        enhancements.extend([
            "scheme", "government", "benefit", "welfare", "eligibility", 
            "application", "guidelines", "procedure"
        ])
        
        # Build enhanced query
        if enhancements:
            return f"{original_query} {' '.join(set(enhancements[:10]))}"
        
        return original_query
    
    def postprocess_results(self, documents: List[Dict[str, Any]], parsed_query: SchemeQuery) -> List[Dict[str, Any]]:
        """Post-process and prioritize results based on scheme relevance and applicability"""
        processed_docs = []
        
        for doc in documents:
            # Score based on scheme category relevance
            category_score = self._score_scheme_category(doc, parsed_query)
            
            # Score based on beneficiary type alignment
            beneficiary_score = self._score_beneficiary_type(doc, parsed_query)
            
            # Score based on scheme scope relevance
            scope_score = self._score_scheme_scope(doc, parsed_query)
            
            # Score based on benefit type alignment
            benefit_score = self._score_benefit_type(doc, parsed_query)
            
            # Score based on eligibility information
            eligibility_score = self._score_eligibility_information(doc, parsed_query)
            
            # Score based on application stage relevance
            stage_score = self._score_application_stage(doc, parsed_query)
            
            # Composite score
            composite_score = (
                category_score * 0.25 +
                beneficiary_score * 0.20 +
                scope_score * 0.15 +
                benefit_score * 0.15 +
                eligibility_score * 0.15 +
                stage_score * 0.10
            )
            
            doc_copy = doc.copy()
            doc_copy["scheme_score"] = composite_score
            doc_copy["scheme_category"] = parsed_query.scheme_category
            doc_copy["beneficiary_type"] = parsed_query.beneficiary_type
            doc_copy["agent"] = "schemes"
            
            # Extract scheme-specific elements
            doc_copy["extracted_schemes"] = self._extract_scheme_names(doc)
            doc_copy["eligibility_criteria"] = self._extract_eligibility_criteria(doc)
            doc_copy["application_process"] = self._extract_application_process(doc)
            doc_copy["benefits_offered"] = self._extract_benefits_offered(doc)
            
            processed_docs.append(doc_copy)
        
        # Sort by composite score
        processed_docs.sort(key=lambda x: x["scheme_score"], reverse=True)
        
        return processed_docs
    
    def extract_scheme_structure(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract scheme structure and implementation details"""
        scheme_structure = {
            "central_schemes": [],
            "state_schemes": [],
            "eligibility_matrix": {},
            "application_procedures": [],
            "benefit_structures": [],
            "implementing_agencies": []
        }
        
        for doc in documents:
            text = doc.get("text", "")
            
            # Extract scheme names
            schemes = self._extract_scheme_names(doc)
            
            # Categorize by scope
            scope = self._determine_scheme_scope_from_text(text)
            if scope == "central":
                scheme_structure["central_schemes"].extend(schemes)
            else:
                scheme_structure["state_schemes"].extend(schemes)
            
            # Extract eligibility criteria
            eligibility = self._extract_eligibility_criteria(doc)
            for scheme in schemes:
                if scheme not in scheme_structure["eligibility_matrix"]:
                    scheme_structure["eligibility_matrix"][scheme] = []
                scheme_structure["eligibility_matrix"][scheme].extend(eligibility)
            
            # Extract application procedures
            procedures = self._extract_application_process(doc)
            scheme_structure["application_procedures"].extend(procedures)
            
            # Extract benefit structures
            benefits = self._extract_benefits_offered(doc)
            scheme_structure["benefit_structures"].extend(benefits)
            
            # Extract implementing agencies
            agencies = self._extract_implementing_agencies(text)
            scheme_structure["implementing_agencies"].extend(agencies)
        
        return scheme_structure
    
    def _identify_scheme_category(self, query_lower: str) -> str:
        """Identify scheme category from query"""
        for category, category_info in self.scheme_categories.items():
            if any(keyword in query_lower for keyword in category_info["keywords"]):
                return category
        
        return "general"  # Default
    
    def _identify_beneficiary_type(self, query_lower: str) -> str:
        """Identify beneficiary type from query"""
        for beneficiary, beneficiary_info in self.beneficiary_types.items():
            if any(keyword in query_lower for keyword in beneficiary_info["keywords"]):
                return beneficiary
        
        return "general_public"  # Default
    
    def _determine_scheme_scope(self, query_lower: str) -> str:
        """Determine scheme scope from query"""
        for scope, scope_info in self.scheme_scopes.items():
            if any(keyword in query_lower for keyword in scope_info["keywords"]):
                return scope
        
        # Default to state for AP context
        return "state"
    
    def _classify_benefit_type(self, query_lower: str) -> str:
        """Classify benefit type from query"""
        for benefit_type, benefit_info in self.benefit_types.items():
            if any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in benefit_info["patterns"]):
                return benefit_type
        
        return "financial"  # Default (most common)
    
    def _identify_eligibility_focus(self, query_lower: str) -> str:
        """Identify eligibility focus from query"""
        for focus, focus_info in self.eligibility_patterns.items():
            if any(indicator in query_lower for indicator in focus_info["indicators"]):
                return focus
        
        return "general"  # Default
    
    def _determine_application_stage(self, query_lower: str, features: Dict[str, Any]) -> str:
        """Determine application stage from query"""
        if any(term in query_lower for term in ["how to apply", "application", "procedure", "form"]):
            return "application"
        elif any(term in query_lower for term in ["eligible", "eligibility", "criteria", "qualification"]):
            return "eligibility"
        elif any(term in query_lower for term in ["status", "track", "payment", "disbursement"]):
            return "status"
        else:
            return "information"  # Default
    
    def _score_scheme_category(self, doc: Dict[str, Any], parsed_query: SchemeQuery) -> float:
        """Score document based on scheme category relevance"""
        text = doc.get("text", "").lower()
        
        if parsed_query.scheme_category == "general":
            return 0.5
        
        category_info = self.scheme_categories.get(parsed_query.scheme_category, {})
        keywords = category_info.get("keywords", [])
        
        matches = sum(1 for keyword in keywords if keyword in text)
        if matches > 0:
            return min(matches / len(keywords), 1.0)
        
        return 0.3
    
    def _score_beneficiary_type(self, doc: Dict[str, Any], parsed_query: SchemeQuery) -> float:
        """Score document based on beneficiary type relevance"""
        text = doc.get("text", "").lower()
        
        beneficiary_info = self.beneficiary_types.get(parsed_query.beneficiary_type, {})
        keywords = beneficiary_info.get("keywords", [])
        
        matches = sum(1 for keyword in keywords if keyword in text)
        if matches > 0:
            return min(matches / len(keywords), 1.0)
        
        return 0.4
    
    def _score_scheme_scope(self, doc: Dict[str, Any], parsed_query: SchemeQuery) -> float:
        """Score document based on scheme scope relevance"""
        text = doc.get("text", "").lower()
        
        scope_info = self.scheme_scopes.get(parsed_query.scheme_scope, {})
        keywords = scope_info.get("keywords", [])
        
        matches = sum(1 for keyword in keywords if keyword in text)
        if matches > 0:
            return min(matches / len(keywords), 1.0)
        
        return 0.5
    
    def _score_benefit_type(self, doc: Dict[str, Any], parsed_query: SchemeQuery) -> float:
        """Score document based on benefit type relevance"""
        text = doc.get("text", "").lower()
        
        benefit_info = self.benefit_types.get(parsed_query.benefit_type, {})
        patterns = benefit_info.get("patterns", [])
        
        matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
        if matches > 0:
            return min(matches / len(patterns), 1.0)
        
        return 0.4
    
    def _score_eligibility_information(self, doc: Dict[str, Any], parsed_query: SchemeQuery) -> float:
        """Score document based on eligibility information relevance"""
        text = doc.get("text", "").lower()
        
        if parsed_query.eligibility_focus == "general":
            return 0.5
        
        eligibility_info = self.eligibility_patterns.get(parsed_query.eligibility_focus, {})
        indicators = eligibility_info.get("indicators", [])
        
        matches = sum(1 for indicator in indicators if indicator in text)
        if matches > 0:
            return min(matches / len(indicators), 1.0)
        
        return 0.3
    
    def _score_application_stage(self, doc: Dict[str, Any], parsed_query: SchemeQuery) -> float:
        """Score document based on application stage relevance"""
        text = doc.get("text", "").lower()
        
        stage_keywords = {
            "information": ["scheme", "about", "details", "overview"],
            "eligibility": ["eligible", "criteria", "qualification", "requirements"],
            "application": ["apply", "application", "form", "procedure", "process"],
            "status": ["status", "track", "payment", "disbursement", "implementation"]
        }
        
        keywords = stage_keywords.get(parsed_query.application_stage, [])
        matches = sum(1 for keyword in keywords if keyword in text)
        
        if matches > 0:
            return min(matches / len(keywords), 1.0)
        
        return 0.5
    
    def _extract_scheme_names(self, doc: Dict[str, Any]) -> List[str]:
        """Extract scheme names from document"""
        text = doc.get("text", "")
        schemes = []
        
        # Common scheme name patterns
        scheme_patterns = [
            r'([A-Z][^.]{10,80}(?:scheme|yojana|program(?:me)?|initiative))',
            r'((?:jagananna|amma|pm|pradhan mantri)\s+[^.]{5,50})',
            r'([A-Z][^.]{5,40}(?:scholarship|pension|assistance))',
            r'(fee\s+reimbursement[^.]{0,30})',
            r'(mid\s+day\s+meal[^.]{0,20})'
        ]
        
        for pattern in scheme_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            schemes.extend([match.strip() for match in matches if len(match.strip()) > 8])
        
        return list(set(schemes))[:5]  # Return top 5 unique schemes
    
    def _extract_eligibility_criteria(self, doc: Dict[str, Any]) -> List[str]:
        """Extract eligibility criteria from document"""
        text = doc.get("text", "")
        criteria = []
        
        criteria_patterns = [
            r'(?:eligible|qualification|criteria|requirement)\s*:?\s*([^.]{30,150})',
            r'(?:must|should|shall)\s+(?:be|have)\s+([^.]{20,100})',
            r'(?:income|family income)\s*:?\s*([^.]{10,80})',
            r'(?:age|category|caste)\s*:?\s*([^.]{10,60})'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return criteria[:5]
    
    def _extract_application_process(self, doc: Dict[str, Any]) -> List[str]:
        """Extract application process steps from document"""
        text = doc.get("text", "")
        process_steps = []
        
        process_patterns = [
            r'(?:step|procedure|process)\s*\d*\s*:?\s*([^.]{20,120})',
            r'(?:submit|apply|fill)\s+([^.]{15,100})',
            r'(?:visit|go to|contact)\s+([^.]{15,80})',
            r'(?:required documents|documents needed)\s*:?\s*([^.]{30,150})'
        ]
        
        for pattern in process_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            process_steps.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return process_steps[:5]
    
    def _extract_benefits_offered(self, doc: Dict[str, Any]) -> List[str]:
        """Extract benefits offered from document"""
        text = doc.get("text", "")
        benefits = []
        
        benefit_patterns = [
            r'(?:benefit|amount|assistance)\s*:?\s*([^.]{20,100})',
            r'(?:�|rs\.?|rupees)\s*([^.]{10,80})',
            r'(?:will\s+(?:get|receive|be\s+given))\s+([^.]{15,100})',
            r'(?:provides?|offers?|gives?)\s+([^.]{15,100})'
        ]
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            benefits.extend([match.strip() for match in matches if len(match.strip()) > 8])
        
        return benefits[:5]
    
    def _extract_implementing_agencies(self, text: str) -> List[str]:
        """Extract implementing agencies from document text"""
        agencies = []
        text_lower = text.lower()
        
        # Collect all agencies from scheme scopes
        all_agencies = []
        for scope_info in self.scheme_scopes.values():
            all_agencies.extend(scope_info["implementing_agencies"])
        
        # Check for agency mentions
        for agency in all_agencies:
            if agency in text_lower:
                agencies.append(agency)
        
        # Extract additional agencies using patterns
        agency_patterns = [
            r'(?:implemented\s+by|under)\s+([^.]{10,60}(?:department|ministry|office|committee))',
            r'([^.]{10,50}(?:collector|commissioner|secretary|director))',
        ]
        
        for pattern in agency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            agencies.extend([match.strip() for match in matches])
        
        return list(set(agencies))
    
    def _determine_scheme_scope_from_text(self, text: str) -> str:
        """Determine scheme scope from document text"""
        text_lower = text.lower()
        
        for scope, scope_info in self.scheme_scopes.items():
            if any(keyword in text_lower for keyword in scope_info["keywords"]):
                return scope
        
        return "state"  # Default for AP context


if __name__ == "__main__":
    # Test the schemes agent
    from config import load_config
    
    config = load_config()
    agent = SchemesAgent(config)
    
    test_query = "What are the eligibility criteria for Jagananna Vidya Deevena scholarship?"
    test_features = {
        "entities": {"scheme_names": ["Jagananna Vidya Deevena"]},
        "query_type": "eligibility",
        "temporal": {"years": ["2023"]}
    }
    
    parsed = agent.analyze_query(test_query, test_features)
    print(f"Parsed query: {parsed}")
    
    filters = agent.build_search_filters(parsed, test_features)
    print(f"Search filters: {filters}")
    
    enhanced = agent.enhance_query(test_query, parsed)
    print(f"Enhanced query: {enhanced}")