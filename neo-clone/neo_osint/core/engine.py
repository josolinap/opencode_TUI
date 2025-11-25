"""
Core Neo-OSINT Engine - Main orchestration system
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import uuid
from datetime import datetime

from .config import NeoOSINTConfig, SearchEngineConfig, AIModelConfig
from ..search.discovery import SearchEngineDiscovery
from ..ai.analyzer import AIAnalyzer
from ..evidence.collector import EvidenceCollector
from ..security.anonymizer import Anonymizer
from ..plugins.manager import PluginManager


@dataclass
class InvestigationResult:
    """Results from an OSINT investigation"""
    investigation_id: str
    query: str
    refined_query: str
    search_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    scraped_content: Dict[str, str]
    analysis: Dict[str, Any]
    evidence_files: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    execution_time: float


class NeoOSINTEngine:
    """Main Neo-OSINT investigation engine"""
    
    def __init__(self, config: Optional[NeoOSINTConfig] = None):
        self.config = config or NeoOSINTConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.search_discovery = SearchEngineDiscovery(self.config)
        self.ai_analyzer = AIAnalyzer(self.config)
        self.evidence_collector = EvidenceCollector(self.config)
        self.anonymizer = Anonymizer(self.config)
        self.plugin_manager = PluginManager(self.config)
        
        # Initialize workspace
        self._initialize_workspace()
        
        # Performance tracking
        self.metrics = {
            "total_investigations": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_requests": 0,
            "average_response_time": 0.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("neo_osint")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_workspace(self) -> None:
        """Initialize workspace directories"""
        workspace = Path(self.config.workspace_dir)
        workspace.mkdir(exist_ok=True)
        
        # Create subdirectories
        (workspace / "evidence").mkdir(exist_ok=True)
        (workspace / "reports").mkdir(exist_ok=True)
        (workspace / "logs").mkdir(exist_ok=True)
        (workspace / "cache").mkdir(exist_ok=True)
        (workspace / "plugins").mkdir(exist_ok=True)
    
    async def investigate(
        self,
        query: str,
        max_results: int = 50,
        include_clear_web: bool = False,
        save_evidence: bool = True,
        use_plugins: bool = True
    ) -> InvestigationResult:
        """
        Conduct a comprehensive OSINT investigation
        
        Args:
            query: The search query
            max_results: Maximum number of results to process
            include_clear_web: Whether to include clear web search
            save_evidence: Whether to save evidence files
            use_plugins: Whether to use plugins for enhanced analysis
        
        Returns:
            InvestigationResult with all findings
        """
        investigation_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting investigation {investigation_id} for query: {query}")
        
        try:
            # Step 1: Query refinement using AI
            self.logger.info("Refining query with AI...")
            refined_query = await self.ai_analyzer.refine_query(query)
            
            # Step 2: Search engine discovery and execution
            self.logger.info("Executing search across multiple engines...")
            search_results = await self.search_discovery.search(
                refined_query,
                max_results=max_results,
                include_clear_web=include_clear_web
            )
            
            # Step 3: AI-powered result filtering
            self.logger.info("Filtering results with AI analysis...")
            filtered_results = await self.ai_analyzer.filter_results(
                refined_query,
                search_results
            )
            
            # Step 4: Content scraping and extraction
            self.logger.info("Scraping content from filtered results...")
            scraped_content = await self.search_discovery.scrape_content(
                filtered_results,
                max_workers=self.config.max_concurrent_requests
            )
            
            # Step 5: Advanced AI analysis
            self.logger.info("Performing advanced threat intelligence analysis...")
            analysis = await self.ai_analyzer.analyze_content(
                query,
                refined_query,
                filtered_results,
                scraped_content
            )
            
            # Step 6: Plugin-based enhancement
            if use_plugins:
                self.logger.info("Running plugin enhancements...")
                plugin_results = await self.plugin_manager.run_plugins(
                    query,
                    filtered_results,
                    scraped_content,
                    analysis
                )
                analysis["plugin_results"] = plugin_results
            
            # Step 7: Evidence collection
            evidence_files = []
            if save_evidence:
                self.logger.info("Collecting and preserving evidence...")
                evidence_files = await self.evidence_collector.collect_evidence(
                    investigation_id,
                    query,
                    filtered_results,
                    scraped_content,
                    analysis
                )
            
            # Step 8: Generate metadata
            metadata = {
                "investigation_id": investigation_id,
                "query": query,
                "refined_query": refined_query,
                "total_search_results": len(search_results),
                "filtered_results": len(filtered_results),
                "scraped_content": len(scraped_content),
                "evidence_files": len(evidence_files),
                "plugins_used": list(self.plugin_manager.active_plugins.keys()) if use_plugins else [],
                "search_engines_used": list(set(r.get("engine") for r in search_results if r.get("engine"))),
                "ai_models_used": self.ai_analyzer.get_models_used(),
                "config_hash": self._get_config_hash()
            }
            
            execution_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(len(search_results), execution_time, success=True)
            
            result = InvestigationResult(
                investigation_id=investigation_id,
                query=query,
                refined_query=refined_query,
                search_results=search_results,
                filtered_results=filtered_results,
                scraped_content=scraped_content,
                analysis=analysis,
                evidence_files=evidence_files,
                metadata=metadata,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            
            self.logger.info(f"Investigation {investigation_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Investigation {investigation_id} failed: {str(e)}")
            self._update_metrics(0, time.time() - start_time, success=False)
            raise
    
    async def generate_report(
        self,
        result: InvestigationResult,
        format: str = "markdown",
        include_raw_data: bool = False
    ) -> str:
        """Generate investigation report"""
        self.logger.info(f"Generating {format} report for investigation {result.investigation_id}")
        
        if format.lower() == "markdown":
            return await self._generate_markdown_report(result, include_raw_data)
        elif format.lower() == "json":
            return await self._generate_json_report(result, include_raw_data)
        elif format.lower() == "html":
            return await self._generate_html_report(result, include_raw_data)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    async def _generate_markdown_report(self, result: InvestigationResult, include_raw: bool) -> str:
        """Generate markdown report"""
        report = f"""# Neo-OSINT Investigation Report

## Investigation Details
- **Investigation ID:** {result.investigation_id}
- **Original Query:** {result.query}
- **Refined Query:** {result.refined_query}
- **Timestamp:** {result.timestamp.isoformat()}
- **Execution Time:** {result.execution_time:.2f} seconds

## Executive Summary
{result.analysis.get('executive_summary', 'No executive summary available.')}

## Key Findings
{result.analysis.get('key_findings', 'No key findings available.')}

## Threat Intelligence Artifacts
{self._format_artifacts(result.analysis.get('artifacts', []))}

## Detailed Analysis
{result.analysis.get('detailed_analysis', 'No detailed analysis available.')}

## Investigation Statistics
- Total Search Results: {len(result.search_results)}
- Filtered Results: {len(result.filtered_results)}
- Scraped Content: {len(result.scraped_content)}
- Evidence Files: {len(result.evidence_files)}

## Search Engines Used
{', '.join(result.metadata.get('search_engines_used', []))}

## Next Steps
{result.analysis.get('next_steps', 'No next steps identified.')}

## Evidence Files
{chr(10).join(f"- {file}" for file in result.evidence_files)}

---
*Report generated by Neo-OSINT on {datetime.now().isoformat()}*
"""
        
        if include_raw:
            report += f"""

## Raw Data

### Search Results
```json
{json.dumps(result.search_results, indent=2)}
```

### Scraped Content
```json
{json.dumps(result.scraped_content, indent=2)}
```
"""
        
        return report
    
    async def _generate_json_report(self, result: InvestigationResult, include_raw: bool) -> str:
        """Generate JSON report"""
        report_data = {
            "investigation_id": result.investigation_id,
            "query": result.query,
            "refined_query": result.refined_query,
            "timestamp": result.timestamp.isoformat(),
            "execution_time": result.execution_time,
            "analysis": result.analysis,
            "metadata": result.metadata,
            "evidence_files": result.evidence_files
        }
        
        if include_raw:
            report_data["search_results"] = result.search_results
            report_data["scraped_content"] = result.scraped_content
        
        return json.dumps(report_data, indent=2)
    
    async def _generate_html_report(self, result: InvestigationResult, include_raw: bool) -> str:
        """Generate HTML report"""
        # Basic HTML template - can be enhanced
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neo-OSINT Investigation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .artifact {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 3px solid #007cba; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Neo-OSINT Investigation Report</h1>
        <p><strong>Investigation ID:</strong> {result.investigation_id}</p>
        <p><strong>Query:</strong> {result.query}</p>
        <p><strong>Timestamp:</strong> {result.timestamp.isoformat()}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{result.analysis.get('executive_summary', 'No executive summary available.')}</p>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        <p>{result.analysis.get('key_findings', 'No key findings available.')}</p>
    </div>
    
    <div class="section">
        <h2>Threat Intelligence Artifacts</h2>
        {self._format_html_artifacts(result.analysis.get('artifacts', []))}
    </div>
    
    <div class="section">
        <h2>Next Steps</h2>
        <p>{result.analysis.get('next_steps', 'No next steps identified.')}</p>
    </div>
</body>
</html>
"""
        return html
    
    def _format_artifacts(self, artifacts: List[Dict]) -> str:
        """Format artifacts for markdown"""
        if not artifacts:
            return "No artifacts identified."
        
        formatted = []
        for artifact in artifacts:
            formatted.append(f"- **{artifact.get('type', 'Unknown')}:** {artifact.get('value', 'N/A')}")
        
        return "\n".join(formatted)
    
    def _format_html_artifacts(self, artifacts: List[Dict]) -> str:
        """Format artifacts for HTML"""
        if not artifacts:
            return "<p>No artifacts identified.</p>"
        
        html = ""
        for artifact in artifacts:
            html += f'<div class="artifact"><strong>{artifact.get("type", "Unknown")}:</strong> {artifact.get("value", "N/A")}</div>'
        
        return html
    
    def _get_config_hash(self) -> str:
        """Get hash of current configuration for tracking"""
        config_str = json.dumps(self.config.__dict__, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _update_metrics(self, requests_count: int, execution_time: float, success: bool) -> None:
        """Update performance metrics"""
        self.metrics["total_investigations"] += 1
        self.metrics["total_requests"] += requests_count
        
        if success:
            self.metrics["successful_searches"] += 1
        else:
            self.metrics["failed_searches"] += 1
        
        # Update average response time
        total_time = self.metrics["average_response_time"] * (self.metrics["total_investigations"] - 1)
        self.metrics["average_response_time"] = (total_time + execution_time) / self.metrics["total_investigations"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.search_discovery.cleanup()
        await self.ai_analyzer.cleanup()
        await self.evidence_collector.cleanup()
        await self.plugin_manager.cleanup()