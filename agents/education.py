"""
Education Agent - Handles educational policies, regulations, and academic matters
Specializes in educational administration, curriculum, and student affairs
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger()


@dataclass
class EducationQuery:
    """Structured representation of education query"""
    education_level: str  # primary, secondary, higher, professional
    subject_area: str  # academic, administrative, policy, infrastructure
    stakeholder: str  # students, teachers, administrators, parents
    policy_type: str  # curriculum, assessment, admission, welfare
    regulation_scope: str  # central, state, district, institutional
    temporal_context: str  # current, historical, upcoming


class EducationAgent:
    """Agent specialized in educational policies, regulations, and academic matters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keywords = config["engines"]["education"]["keywords"]
        
        # Education levels and classifications
        self.education_levels = {
            "primary": {
                "keywords": ["primary school", "elementary", "classes 1-5", "lower primary", "upper primary"],
                "grades": ["1", "2", "3", "4", "5"],
                "focus_areas": ["basic literacy", "numeracy", "foundational skills", "child development"]
            },
            "secondary": {
                "keywords": ["secondary school", "high school", "classes 6-12", "ssc", "intermediate"],
                "grades": ["6", "7", "8", "9", "10", "11", "12"],
                "focus_areas": ["subject specialization", "board exams", "career guidance", "skill development"]
            },
            "higher": {
                "keywords": ["college", "university", "undergraduate", "postgraduate", "degree"],
                "grades": ["ug", "pg", "phd", "diploma"],
                "focus_areas": ["specialization", "research", "professional skills", "employment"]
            },
            "professional": {
                "keywords": ["medical", "engineering", "law", "teaching", "professional courses"],
                "grades": ["professional", "technical", "vocational"],
                "focus_areas": ["professional competency", "licensing", "practice", "specialization"]
            }
        }
        
        # Subject areas and domains
        self.subject_areas = {
            "academic": {
                "patterns": [r'curriculum', r'syllabus', r'subjects', r'academic', r'teaching', r'learning'],
                "topics": ["curriculum design", "pedagogy", "assessment", "academic standards", "textbooks"],
                "stakeholders": ["teachers", "students", "curriculum committee", "education board"]
            },
            "administrative": {
                "patterns": [r'administration', r'management', r'governance', r'policy', r'regulations'],
                "topics": ["school management", "teacher recruitment", "infrastructure", "resource allocation"],
                "stakeholders": ["administrators", "education department", "government", "school management"]
            },
            "student_affairs": {
                "patterns": [r'student', r'admission', r'scholarship', r'welfare', r'discipline'],
                "topics": ["admissions", "scholarships", "student welfare", "counseling", "extracurricular"],
                "stakeholders": ["students", "parents", "counselors", "welfare officers"]
            },
            "infrastructure": {
                "patterns": [r'building', r'facilities', r'equipment', r'infrastructure', r'resources'],
                "topics": ["school buildings", "laboratories", "libraries", "digital infrastructure"],
                "stakeholders": ["administrators", "maintenance", "government", "contractors"]
            }
        }
        
        # Policy types and frameworks
        self.policy_types = {
            "curriculum": {
                "patterns": [r'curriculum', r'syllabus', r'course', r'subject', r'ncf', r'scf'],
                "regulations": ["NCERT guidelines", "state curriculum framework", "assessment guidelines"],
                "authorities": ["NCERT", "SCERT", "education board", "university"]
            },
            "admission": {
                "patterns": [r'admission', r'enrollment', r'seat', r'quota', r'reservation', r'counseling'],
                "regulations": ["admission policy", "reservation rules", "eligibility criteria"],
                "authorities": ["admission committee", "education department", "university"]
            },
            "assessment": {
                "patterns": [r'exam', r'assessment', r'evaluation', r'grading', r'marks', r'ccf'],
                "regulations": ["examination rules", "continuous assessment", "grading system"],
                "authorities": ["board of education", "university", "examination committee"]
            },
            "welfare": {
                "patterns": [r'welfare', r'scholarship', r'fee', r'assistance', r'support', r'meal'],
                "regulations": ["scholarship schemes", "fee structure", "mid-day meal", "student support"],
                "authorities": ["welfare department", "education department", "social welfare"]
            }
        }
        
        # Regulation scope and jurisdictions
        self.regulation_scopes = {
            "central": {
                "authorities": ["mhrd", "ugc", "aicte", "ncte", "ncert", "cbse"],
                "coverage": "national",
                "keywords": ["central government", "ministry", "national policy", "all india"]
            },
            "state": {
                "authorities": ["state education department", "scert", "state board", "state university"],
                "coverage": "state-wide",
                "keywords": ["andhra pradesh", "state government", "state policy", "ap education"]
            },
            "district": {
                "authorities": ["district collector", "deo", "district education office", "zp"],
                "coverage": "district-level",
                "keywords": ["district", "mandal", "block", "local administration"]
            },
            "institutional": {
                "authorities": ["school committee", "college administration", "university senate"],
                "coverage": "institution-specific",
                "keywords": ["school rules", "college policy", "internal regulations", "institutional"]
            }
        }
        
        # Educational stakeholders
        self.stakeholders = {
            "students": ["learner", "pupil", "student", "candidate", "scholar"],
            "teachers": ["teacher", "faculty", "instructor", "educator", "trainer"],
            "administrators": ["principal", "headmaster", "administrator", "officer", "director"],
            "parents": ["parent", "guardian", "family", "community"],
            "government": ["official", "minister", "secretary", "commissioner", "officer"]
        }
        
        logger.info(f"Initialized EducationAgent with keywords: {self.keywords}")
    
    def analyze_query(self, query: str, features: Dict[str, Any]) -> EducationQuery:
        """Analyze query and extract educational requirements"""
        query_lower = query.lower()
        
        # Identify education level
        education_level = self._identify_education_level(query_lower, features)
        
        # Identify subject area
        subject_area = self._identify_subject_area(query_lower)
        
        # Identify primary stakeholder
        stakeholder = self._identify_stakeholder(query_lower)
        
        # Classify policy type
        policy_type = self._classify_policy_type(query_lower)
        
        # Determine regulation scope
        regulation_scope = self._determine_regulation_scope(query_lower)
        
        # Assess temporal context
        temporal_context = self._assess_temporal_context(query, features)
        
        return EducationQuery(
            education_level=education_level,
            subject_area=subject_area,
            stakeholder=stakeholder,
            policy_type=policy_type,
            regulation_scope=regulation_scope,
            temporal_context=temporal_context
        )
    
    def build_search_filters(self, parsed_query: EducationQuery, features: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters for RAG retrieval"""
        filters = {
            "education_level": parsed_query.education_level,
            "subject_area": parsed_query.subject_area,
            "stakeholder": parsed_query.stakeholder,
            "policy_type": parsed_query.policy_type,
            "regulation_scope": parsed_query.regulation_scope
        }
        
        # Add temporal filters
        if parsed_query.temporal_context != "current":
            filters["temporal_context"] = parsed_query.temporal_context
        
        # Extract from features
        temporal = features.get("temporal", {})
        if temporal.get("years"):
            filters["policy_years"] = temporal["years"]
        
        # Add entity-based filters
        entities = features.get("entities", {})
        if entities.get("educational_institutions"):
            filters["institutions"] = entities["educational_institutions"]
        if entities.get("education_authorities"):
            filters["authorities"] = entities["education_authorities"]
        
        return filters
    
    def enhance_query(self, original_query: str, parsed_query: EducationQuery) -> str:
        """Enhance query with educational terminology and policy context"""
        enhancements = []
        
        # Add level-specific terms
        if parsed_query.education_level in self.education_levels:
            level_keywords = self.education_levels[parsed_query.education_level]["keywords"]
            enhancements.extend(level_keywords[:2])
        
        # Add subject area terms
        if parsed_query.subject_area in self.subject_areas:
            area_topics = self.subject_areas[parsed_query.subject_area]["topics"]
            enhancements.extend(area_topics[:2])
        
        # Add policy type terms
        if parsed_query.policy_type in self.policy_types:
            policy_regulations = self.policy_types[parsed_query.policy_type]["regulations"]
            enhancements.extend(policy_regulations[:2])
        
        # Add authority terms
        if parsed_query.regulation_scope in self.regulation_scopes:
            authorities = self.regulation_scopes[parsed_query.regulation_scope]["authorities"]
            enhancements.extend(authorities[:2])
        
        # Add educational terminology
        enhancements.extend([
            "education policy", "guidelines", "regulations", "academic", 
            "institutional", "educational", "learning", "development"
        ])
        
        # Build enhanced query
        if enhancements:
            return f"{original_query} {' '.join(set(enhancements[:8]))}"
        
        return original_query
    
    def postprocess_results(self, documents: List[Dict[str, Any]], parsed_query: EducationQuery) -> List[Dict[str, Any]]:
        """Post-process and prioritize results based on educational relevance and authority"""
        processed_docs = []
        
        for doc in documents:
            # Score based on education level relevance
            level_score = self._score_education_level(doc, parsed_query)
            
            # Score based on subject area alignment
            subject_score = self._score_subject_area(doc, parsed_query)
            
            # Score based on stakeholder relevance
            stakeholder_score = self._score_stakeholder_relevance(doc, parsed_query)
            
            # Score based on policy authority
            authority_score = self._score_policy_authority(doc, parsed_query)
            
            # Score based on temporal relevance
            temporal_score = self._score_temporal_relevance(doc, parsed_query)
            
            # Composite score
            composite_score = (
                level_score * 0.25 +
                subject_score * 0.25 +
                stakeholder_score * 0.20 +
                authority_score * 0.20 +
                temporal_score * 0.10
            )
            
            doc_copy = doc.copy()
            doc_copy["education_score"] = composite_score
            doc_copy["education_level"] = parsed_query.education_level
            doc_copy["subject_area"] = parsed_query.subject_area
            doc_copy["agent"] = "education"
            
            # Extract educational elements
            doc_copy["extracted_policies"] = self._extract_policies(doc)
            doc_copy["extracted_guidelines"] = self._extract_guidelines(doc)
            doc_copy["identified_authorities"] = self._identify_authorities(doc)
            
            processed_docs.append(doc_copy)
        
        # Sort by composite score
        processed_docs.sort(key=lambda x: x["education_score"], reverse=True)
        
        return processed_docs
    
    def extract_policy_structure(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract educational policy structure and hierarchy"""
        policy_structure = {
            "central_policies": [],
            "state_policies": [],
            "institutional_policies": [],
            "guidelines": [],
            "regulations": [],
            "schemes": []
        }
        
        for doc in documents:
            text = doc.get("text", "")
            
            # Extract policies and regulations
            policies = self._extract_policies(doc)
            policy_structure["guidelines"].extend(policies)
            
            # Identify policy scope
            scope = self._determine_policy_scope_from_text(text)
            if scope == "central":
                policy_structure["central_policies"].append({
                    "doc_id": doc.get("id"),
                    "authority": self._extract_primary_authority(text),
                    "policy_ref": self._extract_policy_reference(text)
                })
            elif scope == "state":
                policy_structure["state_policies"].append({
                    "doc_id": doc.get("id"),
                    "authority": self._extract_primary_authority(text),
                    "policy_ref": self._extract_policy_reference(text)
                })
            else:
                policy_structure["institutional_policies"].append({
                    "doc_id": doc.get("id"),
                    "authority": self._extract_primary_authority(text),
                    "policy_ref": self._extract_policy_reference(text)
                })
            
            # Extract schemes
            schemes = self._extract_schemes(text)
            policy_structure["schemes"].extend(schemes)
        
        return policy_structure
    
    def _identify_education_level(self, query_lower: str, features: Dict[str, Any]) -> str:
        """Identify education level from query"""
        # Check explicit mentions
        for level, level_info in self.education_levels.items():
            if any(keyword in query_lower for keyword in level_info["keywords"]):
                return level
        
        # Check for grade/class mentions
        entities = features.get("entities", {})
        if entities.get("education_levels"):
            return entities["education_levels"][0]
        
        # Infer from context
        if any(term in query_lower for term in ["degree", "college", "university"]):
            return "higher"
        elif any(term in query_lower for term in ["school", "class", "grade"]):
            return "secondary"
        elif any(term in query_lower for term in ["primary", "elementary", "basic"]):
            return "primary"
        
        return "general"  # Default
    
    def _identify_subject_area(self, query_lower: str) -> str:
        """Identify subject area from query"""
        for area, area_info in self.subject_areas.items():
            if any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in area_info["patterns"]):
                return area
        
        return "academic"  # Default
    
    def _identify_stakeholder(self, query_lower: str) -> str:
        """Identify primary stakeholder from query"""
        for stakeholder, keywords in self.stakeholders.items():
            if any(keyword in query_lower for keyword in keywords):
                return stakeholder
        
        return "general"  # Default
    
    def _classify_policy_type(self, query_lower: str) -> str:
        """Classify policy type from query"""
        for policy_type, policy_info in self.policy_types.items():
            if any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in policy_info["patterns"]):
                return policy_type
        
        return "general"  # Default
    
    def _determine_regulation_scope(self, query_lower: str) -> str:
        """Determine regulation scope from query"""
        for scope, scope_info in self.regulation_scopes.items():
            if any(keyword in query_lower for keyword in scope_info["keywords"]):
                return scope
        
        # Default to state for AP context
        return "state"
    
    def _assess_temporal_context(self, query: str, features: Dict[str, Any]) -> str:
        """Assess temporal context of the educational inquiry"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["current", "latest", "new", "recent"]):
            return "current"
        elif any(term in query_lower for term in ["upcoming", "proposed", "draft", "future"]):
            return "upcoming"
        elif any(term in query_lower for term in ["historical", "old", "previous", "past"]):
            return "historical"
        
        # Check temporal features
        temporal = features.get("temporal", {})
        if temporal.get("years"):
            year = int(temporal["years"][0])
            current_year = datetime.now().year
            if year >= current_year:
                return "upcoming"
            elif current_year - year <= 3:
                return "current"
            else:
                return "historical"
        
        return "current"  # Default
    
    def _score_education_level(self, doc: Dict[str, Any], parsed_query: EducationQuery) -> float:
        """Score document based on education level relevance"""
        text = doc.get("text", "").lower()
        
        if parsed_query.education_level == "general":
            return 0.5  # Neutral for general queries
        
        level_info = self.education_levels.get(parsed_query.education_level, {})
        keywords = level_info.get("keywords", [])
        
        # Check for direct level mentions
        matches = sum(1 for keyword in keywords if keyword in text)
        if matches > 0:
            return min(matches / len(keywords), 1.0)
        
        return 0.3  # Low score for no matches
    
    def _score_subject_area(self, doc: Dict[str, Any], parsed_query: EducationQuery) -> float:
        """Score document based on subject area alignment"""
        text = doc.get("text", "").lower()
        
        area_info = self.subject_areas.get(parsed_query.subject_area, {})
        topics = area_info.get("topics", [])
        
        # Check for topic matches
        matches = sum(1 for topic in topics if topic.replace(" ", "") in text.replace(" ", ""))
        if matches > 0:
            return min(matches / len(topics), 1.0)
        
        return 0.4
    
    def _score_stakeholder_relevance(self, doc: Dict[str, Any], parsed_query: EducationQuery) -> float:
        """Score document based on stakeholder relevance"""
        text = doc.get("text", "").lower()
        
        if parsed_query.stakeholder == "general":
            return 0.5
        
        stakeholder_keywords = self.stakeholders.get(parsed_query.stakeholder, [])
        matches = sum(1 for keyword in stakeholder_keywords if keyword in text)
        
        if matches > 0:
            return min(matches / len(stakeholder_keywords), 1.0)
        
        return 0.3
    
    def _score_policy_authority(self, doc: Dict[str, Any], parsed_query: EducationQuery) -> float:
        """Score document based on policy authority"""
        text = doc.get("text", "").lower()
        
        scope_info = self.regulation_scopes.get(parsed_query.regulation_scope, {})
        authorities = scope_info.get("authorities", [])
        
        # Check for authority mentions
        matches = sum(1 for authority in authorities if authority in text)
        if matches > 0:
            return min(matches / len(authorities), 1.0)
        
        return 0.4
    
    def _score_temporal_relevance(self, doc: Dict[str, Any], parsed_query: EducationQuery) -> float:
        """Score document based on temporal relevance"""
        metadata = doc.get("metadata", {})
        doc_date = metadata.get("date", metadata.get("policy_date", ""))
        
        if not doc_date:
            return 0.5  # Neutral if no date
        
        # Extract year from document date
        year_match = re.search(r'(19|20)\d{2}', doc_date)
        if not year_match:
            return 0.5
        
        doc_year = int(year_match.group(0))
        current_year = datetime.now().year
        
        if parsed_query.temporal_context == "current":
            age = current_year - doc_year
            if age <= 2:
                return 1.0
            elif age <= 5:
                return 0.8
            else:
                return 0.6
        elif parsed_query.temporal_context == "upcoming":
            if doc_year >= current_year:
                return 1.0
            else:
                return 0.3
        elif parsed_query.temporal_context == "historical":
            age = current_year - doc_year
            if age >= 5:
                return 1.0
            elif age >= 3:
                return 0.8
            else:
                return 0.6
        
        return 0.5
    
    def _extract_policies(self, doc: Dict[str, Any]) -> List[str]:
        """Extract educational policies from document"""
        text = doc.get("text", "")
        policies = []
        
        # Pattern for policies
        policy_patterns = [
            r'(?:policy|guideline|regulation|rule)\s*:?\s*([^.]{50,200})',
            r'(?:as per|according to|under)\s+([^.]{30,150}(?:policy|guideline|act|rule))',
            r'(?:the|this)\s+([^.]{20,100}(?:policy|scheme|regulation))'
        ]
        
        for pattern in policy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            policies.extend([match.strip() for match in matches if len(match.strip()) > 15])
        
        return policies[:5]  # Return top 5 policies
    
    def _extract_guidelines(self, doc: Dict[str, Any]) -> List[str]:
        """Extract educational guidelines from document"""
        text = doc.get("text", "")
        guidelines = []
        
        guideline_patterns = [
            r'(?:guideline|instruction|direction)\s*:?\s*([^.]{30,150})',
            r'(?:shall|must|should)\s+([^.]{20,100})',
            r'(?:it is|students|teachers)\s+(?:required|expected|advised)\s+(?:to\s+)?([^.]{30,150})'
        ]
        
        for pattern in guideline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            guidelines.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return guidelines[:5]  # Return top 5 guidelines
    
    def _identify_authorities(self, doc: Dict[str, Any]) -> List[str]:
        """Identify educational authorities mentioned in document"""
        text = doc.get("text", "").lower()
        authorities = []
        
        # Collect all authorities from regulation scopes
        all_authorities = []
        for scope_info in self.regulation_scopes.values():
            all_authorities.extend(scope_info["authorities"])
        
        # Check for authority mentions
        for authority in all_authorities:
            if authority in text:
                authorities.append(authority)
        
        return list(set(authorities))
    
    def _determine_policy_scope_from_text(self, text: str) -> str:
        """Determine policy scope from document text"""
        text_lower = text.lower()
        
        for scope, scope_info in self.regulation_scopes.items():
            if any(keyword in text_lower for keyword in scope_info["keywords"]):
                return scope
        
        return "institutional"  # Default
    
    def _extract_primary_authority(self, text: str) -> str:
        """Extract primary authority from document text"""
        text_lower = text.lower()
        
        # Check for specific authority patterns
        for scope_info in self.regulation_scopes.values():
            for authority in scope_info["authorities"]:
                if authority in text_lower:
                    return authority
        
        return "education department"  # Default
    
    def _extract_policy_reference(self, text: str) -> str:
        """Extract policy reference from document text"""
        # Look for policy reference patterns
        ref_patterns = [
            r'(?:G\.O\.?\s*(?:Ms\.?|No\.?)?)\s*[A-Z]*\s*\d+',  # Government Order
            r'(?:Policy|Circular|Guideline)\s*(?:No\.?)?\s*[\w\-/]+',
            r'(?:Act|Rule)\s*(?:No\.?)?\s*\d+(?:/\d{4})?'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "No reference found"
    
    def _extract_schemes(self, text: str) -> List[str]:
        """Extract educational schemes from document text"""
        schemes = []
        
        scheme_patterns = [
            r'([^.]{10,80}(?:scheme|yojana|program(?:me)?|initiative))',
            r'(?:under|through)\s+([^.]{15,60}(?:scheme|program))'
        ]
        
        for pattern in scheme_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            schemes.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return schemes[:3]  # Return top 3 schemes


if __name__ == "__main__":
    # Test the education agent
    from config import load_config
    
    config = load_config()
    agent = EducationAgent(config)
    
    test_query = "What are the guidelines for teacher recruitment in primary schools?"
    test_features = {
        "entities": {"education_levels": ["primary"]},
        "query_type": "policy",
        "temporal": {"years": ["2023"]}
    }
    
    parsed = agent.analyze_query(test_query, test_features)
    print(f"Parsed query: {parsed}")
    
    filters = agent.build_search_filters(parsed, test_features)
    print(f"Search filters: {filters}")
    
    enhanced = agent.enhance_query(test_query, parsed)
    print(f"Enhanced query: {enhanced}")