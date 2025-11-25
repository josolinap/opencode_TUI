"""
Validation and testing script for Neo-OSINT
"""

import asyncio
import sys
import os
import json
from pathlib import Path
import logging

# Add the neo_osint package to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.config import NeoOSINTConfig
    from core.engine import NeoOSINTEngine
    from search.discovery import SearchEngineDiscovery
    from ai.analyzer import AIAnalyzer
    from evidence.collector import EvidenceCollector
    from security.anonymizer import Anonymizer
    from plugins.manager import PluginManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class NeoOSINTValidator:
    """Validation system for Neo-OSINT components"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {
            "config": False,
            "search": False,
            "ai": False,
            "evidence": False,
            "security": False,
            "plugins": False,
            "integration": False
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger("neo_osint_validator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def validate_all(self) -> bool:
        """Run all validation tests"""
        print("🔍 Neo-OSINT Validation Suite")
        print("=" * 50)
        
        # Test configuration
        await self._validate_config()
        
        # Test search discovery
        await self._validate_search()
        
        # Test AI analyzer
        await self._validate_ai()
        
        # Test evidence collector
        await self._validate_evidence()
        
        # Test security/anonymizer
        await self._validate_security()
        
        # Test plugin system
        await self._validate_plugins()
        
        # Test integration
        await self._validate_integration()
        
        # Summary
        self._print_summary()
        
        return all(self.test_results.values())
    
    async def _validate_config(self) -> None:
        """Validate configuration system"""
        print("\n📋 Testing Configuration System...")
        
        try:
            # Test default config creation
            config = NeoOSINTConfig()
            assert config.workspace_dir == "neo_osint_workspace"
            
            # Test default search engines
            engines = config.get_default_search_engines()
            assert len(engines) > 0
            
            # Test default AI models
            models = config.get_default_ai_models()
            assert len(models) > 0
            
            # Test config serialization
            test_config_file = "test_config.json"
            config.to_file(test_config_file)
            
            # Test config loading
            loaded_config = NeoOSINTConfig.from_file(test_config_file)
            assert loaded_config.workspace_dir == config.workspace_dir
            
            # Cleanup
            os.remove(test_config_file)
            
            print("✅ Configuration system working correctly")
            self.test_results["config"] = True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
    
    async def _validate_search(self) -> None:
        """Validate search discovery system"""
        print("\n🔍 Testing Search Discovery System...")
        
        try:
            config = NeoOSINTConfig()
            search = SearchEngineDiscovery(config)
            
            # Test search engine initialization
            assert len(search.search_engines) > 0
            
            # Test proxy configuration
            proxies = search._get_tor_proxies()
            assert isinstance(proxies, dict)
            
            # Test header generation
            headers = search._get_headers()
            assert "User-Agent" in headers
            
            # Test result parsing (mock HTML)
            mock_html = '''
            <html>
                <a href="http://example.onion/page">Example Site</a>
                <a href="https://clearweb.com">Clear Web</a>
            </html>
            '''
            results = search._parse_search_results(mock_html, "test_engine")
            assert len(results) >= 1
            assert results[0].url.endswith(".onion")
            
            print("✅ Search discovery system working correctly")
            self.test_results["search"] = True
            
        except Exception as e:
            print(f"❌ Search validation failed: {e}")
    
    async def _validate_ai(self) -> None:
        """Validate AI analyzer system"""
        print("\n🤖 Testing AI Analyzer System...")
        
        try:
            config = NeoOSINTConfig()
            analyzer = AIAnalyzer(config)
            
            # Test artifact patterns
            assert len(analyzer.artifact_patterns) > 0
            
            # Test email extraction
            email_pattern = analyzer.artifact_patterns["email"]
            test_text = "Contact us at admin@example.com for more info"
            emails = email_pattern.findall(test_text)
            assert "admin@example.com" in emails
            
            # Test Bitcoin address extraction
            btc_pattern = analyzer.artifact_patterns["bitcoin"]
            test_btc = "Send to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
            btc_addresses = btc_pattern.findall(test_btc)
            assert "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" in btc_addresses
            
            # Test query refinement (basic)
            refined = await analyzer._refine_query_with_model("test query", config.ai_models[0])
            assert isinstance(refined, str)
            
            # Test artifact extraction
            test_content = {
                "http://test.onion": "Contact admin@test.com or send to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
            }
            artifacts = await analyzer._extract_artifacts(test_content)
            assert len(artifacts) >= 2  # email + bitcoin
            
            print("✅ AI analyzer system working correctly")
            self.test_results["ai"] = True
            
        except Exception as e:
            print(f"❌ AI validation failed: {e}")
    
    async def _validate_evidence(self) -> None:
        """Validate evidence collector system"""
        print("\n📁 Testing Evidence Collector System...")
        
        try:
            config = NeoOSINTConfig()
            config.workspace_dir = "test_workspace"
            collector = EvidenceCollector(config)
            
            # Test directory creation
            assert collector.evidence_dir.exists()
            
            # Test hash calculation
            test_content = "test content for hashing"
            content_hash = collector._calculate_hash(test_content)
            assert len(content_hash) == 64  # SHA256 length
            assert isinstance(content_hash, str)
            
            # Test metadata generation
            test_results = [{"title": "Test", "url": "http://test.onion"}]
            test_analysis = {"artifacts": [], "threat_level": "low"}
            
            metadata_file = await collector._save_metadata(
                collector.evidence_dir / "test",
                "test-id",
                "test query",
                test_results,
                test_analysis
            )
            
            assert os.path.exists(metadata_file)
            
            # Load and verify metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert metadata["investigation_id"] == "test-id"
            assert metadata["query"] == "test query"
            
            # Cleanup
            import shutil
            shutil.rmtree(config.workspace_dir)
            
            print("✅ Evidence collector system working correctly")
            self.test_results["evidence"] = True
            
        except Exception as e:
            print(f"❌ Evidence validation failed: {e}")
    
    async def _validate_security(self) -> None:
        """Validate security/anonymizer system"""
        print("\n🔒 Testing Security & Anonymizer System...")
        
        try:
            config = NeoOSINTConfig()
            config.security.use_tor = False  # Disable for testing
            anonymizer = Anonymizer(config)
            
            # Test header generation
            headers = await anonymizer.get_anonymized_headers()
            assert "User-Agent" in headers
            assert "Accept" in headers
            
            # Test request counting
            anonymizer.increment_request_count()
            assert anonymizer.request_count >= 1
            
            # Test Tor connection check (should return False when disabled)
            tor_status = await anonymizer.check_tor_connection()
            assert isinstance(tor_status, bool)
            
            print("✅ Security & anonymizer system working correctly")
            self.test_results["security"] = True
            
        except Exception as e:
            print(f"❌ Security validation failed: {e}")
    
    async def _validate_plugins(self) -> None:
        """Validate plugin system"""
        print("\n🔌 Testing Plugin System...")
        
        try:
            config = NeoOSINTConfig()
            manager = PluginManager(config)
            
            # Test plugin directory creation
            for plugin_dir in manager.plugin_dirs:
                assert plugin_dir.exists()
            
            # Test built-in plugins
            from plugins.manager import VirusTotalPlugin, ShodanPlugin, IOCExtractorPlugin
            
            # Test plugin instantiation
            vt_plugin = VirusTotalPlugin(config)
            assert vt_plugin.name == "virustotal"
            assert vt_plugin.version == "1.0.0"
            
            shodan_plugin = ShodanPlugin(config)
            assert shodan_plugin.name == "shodan"
            
            ioc_plugin = IOCExtractorPlugin(config)
            assert ioc_plugin.name == "ioc_extractor"
            
            # Test plugin execution
            test_query = "test"
            test_results = []
            test_content = {}
            test_analysis = {}
            
            ioc_result = await ioc_plugin.execute(test_query, test_results, test_content, test_analysis)
            assert "extracted_iocs" in ioc_result
            assert "total_iocs" in ioc_result
            
            print("✅ Plugin system working correctly")
            self.test_results["plugins"] = True
            
        except Exception as e:
            print(f"❌ Plugin validation failed: {e}")
    
    async def _validate_integration(self) -> None:
        """Validate full system integration"""
        print("\n🔗 Testing System Integration...")
        
        try:
            # Create test configuration
            config = NeoOSINTConfig()
            config.workspace_dir = "test_integration_workspace"
            config.security.use_tor = False  # Disable for testing
            config.use_neo_clone_skills = False  # Disable for testing
            
            # Initialize engine
            engine = NeoOSINTEngine(config)
            
            # Test workspace creation
            workspace = Path(config.workspace_dir)
            assert workspace.exists()
            
            # Test metrics
            metrics = engine.get_metrics()
            assert "total_investigations" in metrics
            assert metrics["total_investigations"] == 0
            
            # Test report generation (without actual investigation)
            from core.engine import InvestigationResult
            from datetime import datetime
            
            mock_result = InvestigationResult(
                investigation_id="test-id",
                query="test query",
                refined_query="refined query",
                search_results=[],
                filtered_results=[],
                scraped_content={},
                analysis={
                    "executive_summary": "Test summary",
                    "key_findings": ["Test finding"],
                    "artifacts": [],
                    "detailed_analysis": "Test analysis",
                    "threat_level": "low",
                    "next_steps": ["Test step"]
                },
                evidence_files=[],
                metadata={},
                timestamp=datetime.now(),
                execution_time=1.0
            )
            
            # Test markdown report generation
            markdown_report = await engine.generate_report(mock_result, "markdown")
            assert "Test summary" in markdown_report
            assert "Test finding" in markdown_report
            
            # Test JSON report generation
            json_report = await engine.generate_report(mock_result, "json")
            report_data = json.loads(json_report)
            assert report_data["executive_summary"] == "Test summary"
            
            # Cleanup
            await engine.cleanup()
            
            # Remove test workspace
            import shutil
            shutil.rmtree(config.workspace_dir)
            
            print("✅ System integration working correctly")
            self.test_results["integration"] = True
            
        except Exception as e:
            print(f"❌ Integration validation failed: {e}")
    
    def _print_summary(self) -> None:
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        for component, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{component.title():<15} {status}")
        
        total_passed = sum(self.test_results.values())
        total_tests = len(self.test_results)
        
        print(f"\nOverall: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("🎉 All systems validated successfully!")
        else:
            print("⚠️  Some systems need attention before deployment.")


async def main():
    """Main validation function"""
    validator = NeoOSINTValidator()
    success = await validator.validate_all()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())