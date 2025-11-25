"""
Advanced AI-powered analysis for Neo-OSINT
Integrates with Neo-Clone brain for enhanced intelligence
"""

import asyncio
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Try to import Neo-Clone integration
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from neo_clone import neo_clone
except ImportError:
    neo_clone = None

from ..core.config import NeoOSINTConfig, AIModelConfig


@dataclass
class ThreatArtifact:
    """Threat intelligence artifact"""
    type: str  # email, phone, crypto, domain, ip, etc.
    value: str
    confidence: float
    context: str
    source_url: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class AnalysisResult:
    """Result of AI analysis"""
    executive_summary: str
    key_findings: List[str]
    artifacts: List[ThreatArtifact]
    detailed_analysis: str
    threat_level: str
    next_steps: List[str]
    confidence_score: float
    models_used: List[str]


class AIAnalyzer:
    """Advanced AI analysis with Neo-Clone integration"""
    
    def __init__(self, config: NeoOSINTConfig):
        self.config = config
        self.logger = logging.getLogger("neo_osint.ai")
        
        # Initialize AI models
        self.models = self._initialize_models()
        self.active_models = []
        
        # Neo-Clone integration
        self.neo_clone_available = neo_clone is not None and config.use_neo_clone_skills
        
        # Artifact patterns
        self.artifact_patterns = self._initialize_artifact_patterns()
        
        # Performance tracking
        self.models_used = []
    
    def _initialize_models(self) -> List[AIModelConfig]:
        """Initialize AI models from configuration"""
        models = self.config.ai_models.copy()
        
        # If no models configured, use defaults
        if not models:
            models = self.config.get_default_ai_models()
        
        # Filter enabled models
        return [model for model in models if model.enabled]
    
    def _initialize_artifact_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for artifact extraction"""
        return {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            "bitcoin": re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'),
            "ethereum": re.compile(r'\b0x[a-fA-F0-9]{40}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "domain": re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'),
            "onion_url": re.compile(r'\b[a-z2-7]{16,56}\.onion\b'),
            "discord": re.compile(r'\bdiscord\.gg/[a-zA-Z0-9]+\b'),
            "telegram": re.compile(r'\bt\.me/[a-zA-Z0-9_]+\b'),
            "pgp_key": re.compile(r'-----BEGIN PGP [A-Z]+-----[\s\S]+?-----END PGP [A-Z]+-----'),
            "hash_sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),
            "hash_md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
            "hash_sha1": re.compile(r'\b[a-fA-F0-9]{40}\b')
        }
    
    async def refine_query(self, query: str) -> str:
        """Refine search query using AI"""
        self.logger.info(f"Refining query: {query}")
        
        # Try Neo-Clone first if available
        if self.neo_clone_available:
            try:
                result = await self._neo_clone_refine_query(query)
                if result:
                    self.models_used.append("neo-clone")
                    return result
            except Exception as e:
                self.logger.warning(f"Neo-Clone query refinement failed: {e}")
        
        # Fallback to traditional AI models
        for model in self.models:
            try:
                refined = await self._refine_query_with_model(query, model)
                if refined and refined != query:
                    self.models_used.append(f"{model.provider}:{model.model_name}")
                    return refined
            except Exception as e:
                self.logger.warning(f"Query refinement failed with {model.model_name}: {e}")
                continue
        
        # Return original query if all refinements fail
        return query
    
    async def _neo_clone_refine_query(self, query: str) -> Optional[str]:
        """Use Neo-Clone to refine query"""
        if not neo_clone:
            return None
        
        prompt = f"""
        As an expert OSINT investigator, refine this search query for dark web investigation:
        
        Original query: {query}
        
        Rules:
        1. Make the query more specific and effective for dark web search
        2. Add relevant keywords that threat actors might use
        3. Remove unnecessary words
        4. Use terminology common in cybercrime forums
        5. Return only the refined query, no explanation
        
        Refined query:
        """
        
        try:
            response = await asyncio.to_thread(
                neo_clone,
                message=prompt,
                mode="tool",
                timeout=30000
            )
            
            # Extract the refined query from response
            if response and isinstance(response, str):
                lines = response.strip().split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('#') and not line.startswith('//'):
                        return line.strip()
            
            return None
        except Exception as e:
            self.logger.error(f"Neo-Clone refinement error: {e}")
            return None
    
    async def _refine_query_with_model(self, query: str, model: AIModelConfig) -> Optional[str]:
        """Refine query using traditional AI model"""
        # This would integrate with OpenAI, Anthropic, etc.
        # For now, return a simple enhancement
        enhanced_terms = {
            "ransomware": ["ransomware", "extortion", "encryption", "bitcoin", "payment"],
            "malware": ["malware", "virus", "trojan", "backdoor", "payload"],
            "data breach": ["data breach", "leak", "database", "dump", "exposed"],
            "credentials": ["credentials", "password", "login", "access", "accounts"]
        }
        
        query_lower = query.lower()
        for key, terms in enhanced_terms.items():
            if key in query_lower:
                # Add relevant terms
                for term in terms:
                    if term not in query_lower:
                        return f"{query} {term}"
        
        return query
    
    async def filter_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and rank search results using AI"""
        self.logger.info(f"Filtering {len(results)} search results")
        
        if not results:
            return []
        
        # Try Neo-Clone first
        if self.neo_clone_available:
            try:
                filtered = await self._neo_clone_filter_results(query, results)
                if filtered:
                    self.models_used.append("neo-clone")
                    return filtered
            except Exception as e:
                self.logger.warning(f"Neo-Clone filtering failed: {e}")
        
        # Fallback to traditional filtering
        return await self._traditional_filter_results(query, results)
    
    async def _neo_clone_filter_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Use Neo-Clone to filter results"""
        if not neo_clone:
            return None
        
        # Format results for analysis
        results_text = "\n".join([
            f"{i+1}. {r['title']} - {r['url']}"
            for i, r in enumerate(results[:50])  # Limit to 50 for analysis
        ])
        
        prompt = f"""
        As a cybercrime threat intelligence expert, analyze these search results for the query: "{query}"
        
        Select the TOP 20 most relevant results that would be valuable for investigation.
        
        Results:
        {results_text}
        
        Return only the indices (numbers) of the most relevant results, separated by commas:
        """
        
        try:
            response = await asyncio.to_thread(
                neo_clone,
                message=prompt,
                mode="tool",
                timeout=30000
            )
            
            if response and isinstance(response, str):
                # Extract indices
                indices = []
                for match in re.findall(r'\d+', response):
                    idx = int(match) - 1  # Convert to 0-based
                    if 0 <= idx < len(results):
                        indices.append(idx)
                
                if indices:
                    return [results[i] for i in indices[:20]]
            
            return None
        except Exception as e:
            self.logger.error(f"Neo-Clone filtering error: {e}")
            return None
    
    async def _traditional_filter_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Traditional result filtering based on keywords and relevance"""
        query_terms = query.lower().split()
        
        scored_results = []
        for result in results:
            score = 0.0
            
            # Title matching
            title_lower = result['title'].lower()
            for term in query_terms:
                if term in title_lower:
                    score += 2.0
            
            # URL matching
            url_lower = result['url'].lower()
            for term in query_terms:
                if term in url_lower:
                    score += 1.0
            
            # Snippet matching
            snippet_lower = result.get('snippet', '').lower()
            for term in query_terms:
                if term in snippet_lower:
                    score += 1.5
            
            # Onion site bonus
            if '.onion' in url_lower:
                score += 1.0
            
            # Freshness bonus (if timestamp available)
            if result.get('timestamp'):
                age = datetime.now().timestamp() - result['timestamp']
                if age < 86400:  # Less than 1 day
                    score += 0.5
            
            result['relevance_score'] = score
            scored_results.append(result)
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_results[:20]
    
    async def analyze_content(
        self,
        original_query: str,
        refined_query: str,
        filtered_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str]
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis of scraped content"""
        self.logger.info("Performing comprehensive content analysis")
        
        # Extract artifacts
        artifacts = await self._extract_artifacts(scraped_content)
        
        # Generate analysis using Neo-Clone if available
        if self.neo_clone_available:
            try:
                analysis = await self._neo_clone_analyze_content(
                    original_query,
                    refined_query,
                    filtered_results,
                    scraped_content,
                    artifacts
                )
                if analysis:
                    self.models_used.append("neo-clone")
                    return analysis
            except Exception as e:
                self.logger.warning(f"Neo-Clone analysis failed: {e}")
        
        # Fallback to traditional analysis
        return await self._traditional_analyze_content(
            original_query,
            refined_query,
            filtered_results,
            scraped_content,
            artifacts
        )
    
    async def _extract_artifacts(self, scraped_content: Dict[str, str]) -> List[ThreatArtifact]:
        """Extract threat intelligence artifacts from content"""
        artifacts = []
        
        for url, content in scraped_content.items():
            for artifact_type, pattern in self.artifact_patterns.items():
                matches = pattern.findall(content)
                for match in matches:
                    # Calculate confidence based on context
                    confidence = self._calculate_artifact_confidence(artifact_type, match, content)
                    
                    artifact = ThreatArtifact(
                        type=artifact_type,
                        value=match,
                        confidence=confidence,
                        context=self._get_artifact_context(match, content),
                        source_url=url,
                        severity=self._determine_artifact_severity(artifact_type, match)
                    )
                    artifacts.append(artifact)
        
        # Remove duplicates and sort by confidence
        unique_artifacts = {}
        for artifact in artifacts:
            key = f"{artifact.type}:{artifact.value}"
            if key not in unique_artifacts or artifact.confidence > unique_artifacts[key].confidence:
                unique_artifacts[key] = artifact
        
        return sorted(unique_artifacts.values(), key=lambda x: x.confidence, reverse=True)
    
    def _calculate_artifact_confidence(self, artifact_type: str, value: str, content: str) -> float:
        """Calculate confidence score for an artifact"""
        base_confidence = {
            "email": 0.7,
            "phone": 0.6,
            "bitcoin": 0.8,
            "ethereum": 0.8,
            "ip_address": 0.5,
            "domain": 0.6,
            "onion_url": 0.9,
            "discord": 0.8,
            "telegram": 0.8,
            "pgp_key": 0.9,
            "hash_sha256": 0.7,
            "hash_md5": 0.6,
            "hash_sha1": 0.6
        }.get(artifact_type, 0.5)
        
        # Boost confidence based on context
        context_indicators = [
            "contact", "email", "phone", "address", "wallet", "payment",
            "server", "host", "domain", "website", "forum", "market"
        ]
        
        context_words = content.lower().split()
        for indicator in context_indicators:
            if indicator in context_words:
                base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _get_artifact_context(self, artifact: str, content: str) -> str:
        """Get context around an artifact"""
        # Find artifact in content and extract surrounding text
        index = content.find(artifact)
        if index == -1:
            return ""
        
        start = max(0, index - 50)
        end = min(len(content), index + len(artifact) + 50)
        
        context = content[start:end].replace('\n', ' ').strip()
        return context
    
    def _determine_artifact_severity(self, artifact_type: str, value: str) -> str:
        """Determine severity level for an artifact"""
        high_severity_types = ["bitcoin", "ethereum", "pgp_key", "onion_url"]
        medium_severity_types = ["email", "phone", "discord", "telegram"]
        
        if artifact_type in high_severity_types:
            return "high"
        elif artifact_type in medium_severity_types:
            return "medium"
        else:
            return "low"
    
    async def _neo_clone_analyze_content(
        self,
        original_query: str,
        refined_query: str,
        filtered_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        artifacts: List[ThreatArtifact]
    ) -> Optional[Dict[str, Any]]:
        """Use Neo-Clone for advanced content analysis"""
        if not neo_clone:
            return None
        
        # Prepare content summary
        content_summary = "\n\n".join([
            f"URL: {url}\nContent: {content[:500]}..."
            for url, content in list(scraped_content.items())[:10]
        ])
        
        artifacts_text = "\n".join([
            f"- {a.type}: {a.value} (confidence: {a.confidence:.2f})"
            for a in artifacts[:20]
        ])
        
        prompt = f"""
        As an expert cybercrime threat intelligence analyst, analyze this OSINT investigation data:
        
        Investigation Query: {original_query}
        Refined Query: {refined_query}
        
        Key Findings from Content:
        {content_summary}
        
        Identified Threat Artifacts:
        {artifacts_text}
        
        Provide a comprehensive analysis including:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (3-5 bullet points)
        3. Detailed Analysis (technical insights)
        4. Threat Level Assessment (low/medium/high/critical)
        5. Recommended Next Steps (3-5 actionable items)
        
        Format your response as JSON with these keys:
        {{
            "executive_summary": "...",
            "key_findings": ["...", "..."],
            "detailed_analysis": "...",
            "threat_level": "...",
            "next_steps": ["...", "..."]
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                neo_clone,
                message=prompt,
                mode="tool",
                timeout=60000
            )
            
            if response and isinstance(response, str):
                # Try to parse JSON response
                try:
                    analysis_data = json.loads(response)
                    analysis_data["artifacts"] = [a.__dict__ for a in artifacts]
                    analysis_data["confidence_score"] = 0.85
                    analysis_data["models_used"] = ["neo-clone"]
                    return analysis_data
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return {
                        "executive_summary": "Analysis completed using Neo-Clone AI brain",
                        "key_findings": ["Advanced threat intelligence analysis performed"],
                        "detailed_analysis": response,
                        "threat_level": "medium",
                        "next_steps": ["Review detailed analysis results"],
                        "artifacts": [a.__dict__ for a in artifacts],
                        "confidence_score": 0.8,
                        "models_used": ["neo-clone"]
                    }
            
            return None
        except Exception as e:
            self.logger.error(f"Neo-Clone analysis error: {e}")
            return None
    
    async def _traditional_analyze_content(
        self,
        original_query: str,
        refined_query: str,
        filtered_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        artifacts: List[ThreatArtifact]
    ) -> Dict[str, Any]:
        """Traditional content analysis without Neo-Clone"""
        
        # Generate executive summary
        total_artifacts = len(artifacts)
        high_confidence_artifacts = len([a for a in artifacts if a.confidence > 0.7])
        
        executive_summary = f"Investigation of '{original_query}' yielded {len(filtered_results)} relevant results with {total_artifacts} identified artifacts, {high_confidence_artifacts} of which are high confidence."
        
        # Generate key findings
        key_findings = [
            f"Found {total_artifacts} potential threat intelligence artifacts",
            f"{len(filtered_results)} relevant sources identified and analyzed",
            f"High-confidence indicators: {high_confidence_artifacts}"
        ]
        
        # Add artifact type breakdown
        artifact_types = {}
        for artifact in artifacts:
            artifact_types[artifact.type] = artifact_types.get(artifact.type, 0) + 1
        
        if artifact_types:
            key_findings.append(f"Primary artifact types: {', '.join(f'{k} ({v})' for k, v in sorted(artifact_types.items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        # Detailed analysis
        detailed_analysis = f"""
        Investigation Analysis for: {original_query}
        
        Search Results Analysis:
        - Total results processed: {len(filtered_results)}
        - Content successfully scraped: {len(scraped_content)}
        - Primary sources: Onion sites and dark web forums
        
        Threat Intelligence Artifacts:
        - Total artifacts identified: {total_artifacts}
        - High confidence artifacts: {high_confidence_artifacts}
        - Artifact categories: {', '.join(artifact_types.keys())}
        
        Content Analysis:
        The investigation revealed various indicators that may be relevant to the query. 
        The extracted artifacts provide potential leads for further investigation.
        """
        
        # Determine threat level
        threat_level = "low"
        if high_confidence_artifacts > 10:
            threat_level = "critical"
        elif high_confidence_artifacts > 5:
            threat_level = "high"
        elif high_confidence_artifacts > 2:
            threat_level = "medium"
        
        # Next steps
        next_steps = [
            "Investigate high-confidence artifacts for additional context",
            "Cross-reference artifacts with known threat intelligence databases",
            "Monitor identified sources for related activity",
            "Consider additional searches based on key findings"
        ]
        
        return {
            "executive_summary": executive_summary,
            "key_findings": key_findings,
            "detailed_analysis": detailed_analysis,
            "threat_level": threat_level,
            "next_steps": next_steps,
            "artifacts": [a.__dict__ for a in artifacts],
            "confidence_score": min(0.7 + (high_confidence_artifacts * 0.05), 0.95),
            "models_used": self.models_used.copy()
        }
    
    def get_models_used(self) -> List[str]:
        """Get list of AI models used in analysis"""
        return self.models_used.copy()
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.models_used.clear()