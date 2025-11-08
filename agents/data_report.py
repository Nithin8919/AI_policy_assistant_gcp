"""
Data Report Agent - Handles educational statistics, UDISE data, budget reports
Specializes in quantitative analysis and reporting
"""
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger()


@dataclass
class DataReportQuery:
    """Structured representation of data report query"""
    metric_type: str  # udise, aser, nas, budget, ses
    indicators: List[str]  # specific metrics requested
    geographic_scope: List[str]  # districts, mandals
    time_period: Optional[str]  # academic year, FY
    comparisons: List[str]  # year-over-year, district comparisons
    aggregation: str  # state, district, mandal, school level


class DataReportAgent:
    """Agent specialized in educational data and statistics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.facets = config["engines"]["data_report"]["facets"]
        self.keywords = config["engines"]["data_report"]["keywords"]
        
        # Metric categories and indicators
        self.metric_definitions = {
            "udise": {
                "enrollment": ["total enrollment", "gross enrollment ratio", "ger", "net enrollment"],
                "infrastructure": ["schools with electricity", "computer lab", "library", "playground"],
                "teachers": ["pupil teacher ratio", "ptr", "trained teachers", "teacher qualifications"],
                "retention": ["dropout rate", "completion rate", "transition rate"],
                "equity": ["gender parity index", "gpi", "sc/st enrollment", "cwsn enrollment"]
            },
            "aser": {
                "reading": ["reading level", "story reading", "paragraph reading", "letter recognition"],
                "arithmetic": ["subtraction", "division", "number recognition", "basic math"],
                "english": ["english reading", "word reading", "sentence reading"],
                "rural_focus": ["rural children", "village schools", "government school"]
            },
            "nas": {
                "subjects": ["mathematics", "language", "science", "social science"],
                "grades": ["grade 3", "grade 5", "grade 8", "grade 10"],
                "competencies": ["learning outcomes", "competency levels", "performance"]
            },
            "budget": {
                "allocation": ["budget allocation", "fund utilization", "per student expenditure"],
                "schemes": ["samagra shiksha", "mdm allocation", "scheme funding"],
                "categories": ["teacher salary", "infrastructure", "material", "training"]
            },
            "ses": {
                "household": ["economic status", "income levels", "poverty indicators"],
                "education": ["parental education", "family literacy", "educational aspirations"],
                "access": ["school distance", "transportation", "availability"]
            }
        }
        
        logger.info(f"Initialized DataReportAgent with facets: {self.facets}")
    
    def analyze_query(self, query: str, features: Dict[str, Any]) -> DataReportQuery:
        """Analyze query and extract data report requirements"""
        query_lower = query.lower()
        
        # Identify metric type
        metric_type = self._identify_metric_type(query_lower)
        
        # Extract specific indicators
        indicators = self._extract_indicators(query_lower, metric_type)
        
        # Extract geographic scope
        geographic_scope = self._extract_geographic_scope(query, features)
        
        # Extract time period
        time_period = self._extract_time_period(query, features)
        
        # Identify comparison requests
        comparisons = self._identify_comparisons(query_lower)
        
        # Determine aggregation level
        aggregation = self._determine_aggregation_level(query_lower)
        
        return DataReportQuery(
            metric_type=metric_type,
            indicators=indicators,
            geographic_scope=geographic_scope,
            time_period=time_period,
            comparisons=comparisons,
            aggregation=aggregation
        )
    
    def build_search_filters(self, parsed_query: DataReportQuery, features: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters for RAG retrieval"""
        filters = {
            "facets": [parsed_query.metric_type],
            "indicators": parsed_query.indicators,
            "aggregation_level": parsed_query.aggregation
        }
        
        # Add geographic filters
        if parsed_query.geographic_scope:
            filters["districts"] = parsed_query.geographic_scope
        
        # Add temporal filters
        if parsed_query.time_period:
            filters["time_period"] = parsed_query.time_period
            
        # Extract specific years from features
        temporal = features.get("temporal", {})
        if temporal.get("fiscal_years"):
            filters["fiscal_years"] = temporal["fiscal_years"]
        if temporal.get("years"):
            filters["years"] = temporal["years"]
        
        return filters
    
    def enhance_query(self, original_query: str, parsed_query: DataReportQuery) -> str:
        """Enhance query with domain-specific terms for better retrieval"""
        enhancements = []
        
        # Add metric-specific terminology
        if parsed_query.metric_type in self.metric_definitions:
            for category, terms in self.metric_definitions[parsed_query.metric_type].items():
                if any(term in original_query.lower() for term in terms[:2]):  # Use first 2 terms
                    enhancements.extend(terms[:3])  # Add 3 related terms
        
        # Add aggregation context
        if parsed_query.aggregation == "district":
            enhancements.extend(["district-wise", "by district", "district comparison"])
        elif parsed_query.aggregation == "state":
            enhancements.extend(["state-level", "andhra pradesh", "overall state"])
        
        # Add comparison context
        if "comparison" in parsed_query.comparisons:
            enhancements.extend(["compared to", "versus", "change over time"])
        
        # Build enhanced query
        if enhancements:
            return f"{original_query} {' '.join(set(enhancements[:8]))}"
        
        return original_query
    
    def postprocess_results(self, documents: List[Dict[str, Any]], parsed_query: DataReportQuery) -> List[Dict[str, Any]]:
        """Post-process and prioritize results based on data report needs"""
        processed_docs = []
        
        for doc in documents:
            # Score based on metric type alignment
            metric_score = self._score_metric_alignment(doc, parsed_query)
            
            # Score based on aggregation level match
            aggregation_score = self._score_aggregation_match(doc, parsed_query)
            
            # Score based on temporal alignment
            temporal_score = self._score_temporal_alignment(doc, parsed_query)
            
            # Composite score
            composite_score = (
                metric_score * 0.4 +
                aggregation_score * 0.3 +
                temporal_score * 0.3
            )
            
            doc_copy = doc.copy()
            doc_copy["data_report_score"] = composite_score
            doc_copy["metric_type"] = parsed_query.metric_type
            doc_copy["agent"] = "data_report"
            
            processed_docs.append(doc_copy)
        
        # Sort by composite score
        processed_docs.sort(key=lambda x: x["data_report_score"], reverse=True)
        
        return processed_docs
    
    def extract_key_metrics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key numerical metrics from documents for summary"""
        key_metrics = {
            "numerical_values": [],
            "percentages": [],
            "trends": [],
            "comparisons": []
        }
        
        for doc in documents:
            text = doc.get("text", "")
            
            # Extract numerical values with units
            number_patterns = [
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|crore|thousand|million|%|percent)',
                r'(\d+(?:\.\d+)?)\s*(ratio|rate|index|score)'
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    key_metrics["numerical_values"].append({
                        "value": match[0],
                        "unit": match[1],
                        "source": doc.get("vertical", ""),
                        "doc_id": doc.get("id", "")
                    })
            
            # Extract percentages
            pct_pattern = r'(\d+(?:\.\d+)?)\s*(?:%|percent)'
            percentages = re.findall(pct_pattern, text)
            for pct in percentages:
                key_metrics["percentages"].append({
                    "value": float(pct),
                    "source": doc.get("vertical", ""),
                    "context": text[max(0, text.find(pct)-50):text.find(pct)+100]
                })
        
        return key_metrics
    
    def _identify_metric_type(self, query_lower: str) -> str:
        """Identify the primary metric type from query"""
        # Direct mentions
        if any(term in query_lower for term in ["udise", "unified district", "school statistics"]):
            return "udise"
        elif any(term in query_lower for term in ["aser", "annual status", "learning level"]):
            return "aser"
        elif any(term in query_lower for term in ["nas", "national achievement", "learning outcome"]):
            return "nas"
        elif any(term in query_lower for term in ["budget", "allocation", "expenditure", "fund"]):
            return "budget"
        elif any(term in query_lower for term in ["socio-economic", "ses", "household"]):
            return "ses"
        
        # Infer from indicators
        if any(term in query_lower for term in ["enrollment", "dropout", "ptr", "infrastructure"]):
            return "udise"
        elif any(term in query_lower for term in ["reading level", "arithmetic", "subtraction"]):
            return "aser"
        elif any(term in query_lower for term in ["grade", "competency", "subject wise"]):
            return "nas"
        
        return "general"  # Default
    
    def _extract_indicators(self, query_lower: str, metric_type: str) -> List[str]:
        """Extract specific indicators mentioned in query"""
        indicators = []
        
        if metric_type in self.metric_definitions:
            for category, terms in self.metric_definitions[metric_type].items():
                for term in terms:
                    if term in query_lower:
                        indicators.append(term)
        
        return indicators
    
    def _extract_geographic_scope(self, query: str, features: Dict[str, Any]) -> List[str]:
        """Extract geographic scope from query and features"""
        geographic_scope = []
        
        # From constraints
        constraints = features.get("constraints", {})
        if constraints.get("districts"):
            geographic_scope.extend(constraints["districts"])
        
        # Common AP districts (basic extraction)
        ap_districts = [
            "krishna", "guntur", "east godavari", "west godavari", "visakhapatnam",
            "srikakulam", "vizianagaram", "kurnool", "anantapur", "chittoor",
            "kadapa", "nellore", "prakasam"
        ]
        
        query_lower = query.lower()
        for district in ap_districts:
            if district in query_lower:
                geographic_scope.append(district.title())
        
        return list(set(geographic_scope))
    
    def _extract_time_period(self, query: str, features: Dict[str, Any]) -> Optional[str]:
        """Extract time period specifications"""
        temporal = features.get("temporal", {})
        
        if temporal.get("fiscal_years"):
            return f"FY {temporal['fiscal_years'][0][0]}-{temporal['fiscal_years'][0][1] or temporal['fiscal_years'][0][0][-2:]}"
        
        if temporal.get("years"):
            years = temporal["years"]
            if len(years) == 1:
                return f"AY {years[0]}-{int(years[0])+1}"
            else:
                return f"{years[0]}-{years[-1]}"
        
        # Extract academic year patterns
        ay_pattern = r'(?:AY|academic year)\s*(20\d{2})[-\s]*(20)?\d{2}'
        ay_match = re.search(ay_pattern, query, re.IGNORECASE)
        if ay_match:
            return f"AY {ay_match.group(1)}-{ay_match.group(2) or str(int(ay_match.group(1))+1)[-2:]}"
        
        return None
    
    def _identify_comparisons(self, query_lower: str) -> List[str]:
        """Identify types of comparisons requested"""
        comparisons = []
        
        comparison_indicators = {
            "temporal": ["over time", "year over year", "trend", "change from", "compared to previous"],
            "geographic": ["district wise", "by district", "state vs district", "compared to other"],
            "demographic": ["by gender", "sc/st", "urban vs rural", "government vs private"],
            "performance": ["top performing", "bottom", "above average", "below state"]
        }
        
        for comp_type, indicators in comparison_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                comparisons.append(comp_type)
        
        return comparisons
    
    def _determine_aggregation_level(self, query_lower: str) -> str:
        """Determine the level of data aggregation needed"""
        if any(term in query_lower for term in ["state level", "andhra pradesh", "overall", "total"]):
            return "state"
        elif any(term in query_lower for term in ["district", "district wise", "by district"]):
            return "district"
        elif any(term in query_lower for term in ["mandal", "block", "cluster"]):
            return "mandal"
        elif any(term in query_lower for term in ["school", "school wise", "individual school"]):
            return "school"
        else:
            return "district"  # Default to district level
    
    def _score_metric_alignment(self, doc: Dict[str, Any], parsed_query: DataReportQuery) -> float:
        """Score document based on metric type alignment"""
        text = doc.get("text", "").lower()
        score = 0.0
        
        # Direct metric type match
        if parsed_query.metric_type in text:
            score += 0.5
        
        # Indicator matches
        indicator_matches = sum(1 for indicator in parsed_query.indicators if indicator in text)
        if parsed_query.indicators:
            score += 0.5 * (indicator_matches / len(parsed_query.indicators))
        
        return min(score, 1.0)
    
    def _score_aggregation_match(self, doc: Dict[str, Any], parsed_query: DataReportQuery) -> float:
        """Score document based on aggregation level match"""
        text = doc.get("text", "").lower()
        metadata = doc.get("metadata", {})
        
        # Check if document contains data at requested aggregation level
        aggregation_terms = {
            "state": ["state", "andhra pradesh", "overall", "total"],
            "district": ["district", "district wise", "by district"],
            "mandal": ["mandal", "block", "cluster"],
            "school": ["school", "school wise"]
        }
        
        terms = aggregation_terms.get(parsed_query.aggregation, [])
        if any(term in text for term in terms):
            return 1.0
        
        # Check metadata
        if metadata.get("aggregation_level") == parsed_query.aggregation:
            return 1.0
        
        return 0.3  # Partial match
    
    def _score_temporal_alignment(self, doc: Dict[str, Any], parsed_query: DataReportQuery) -> float:
        """Score document based on temporal alignment"""
        if not parsed_query.time_period:
            return 0.5  # Neutral if no temporal requirement
        
        text = doc.get("text", "").lower()
        metadata = doc.get("metadata", {})
        
        # Extract years from time period
        time_period_lower = parsed_query.time_period.lower()
        
        # Check for year matches in text
        if any(year in text for year in re.findall(r'20\d{2}', time_period_lower)):
            return 1.0
        
        # Check metadata for date alignment
        doc_date = metadata.get("date", metadata.get("published_date", ""))
        if doc_date and any(year in doc_date for year in re.findall(r'20\d{2}', time_period_lower)):
            return 1.0
        
        return 0.2  # Low score for temporal mismatch


if __name__ == "__main__":
    # Test the data report agent
    from config import load_config
    
    config = load_config()
    agent = DataReportAgent(config)
    
    test_query = "Show me UDISE enrollment data for Krishna district for FY 2023-24"
    test_features = {
        "entities": {"metrics": ["UDISE", "enrollment"]},
        "facets": ["udise"],
        "constraints": {"districts": ["Krishna"]},
        "temporal": {"fiscal_years": [["2023", "24"]]}
    }
    
    parsed = agent.analyze_query(test_query, test_features)
    print(f"Parsed query: {parsed}")
    
    filters = agent.build_search_filters(parsed, test_features)
    print(f"Search filters: {filters}")
    
    enhanced = agent.enhance_query(test_query, parsed)
    print(f"Enhanced query: {enhanced}")
