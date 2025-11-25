"""
Plugin management system for Neo-OSINT
"""

import asyncio
import logging
import importlib
import os
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod

from ..core.config import NeoOSINTConfig


class OSINTPlugin(ABC):
    """Base class for OSINT plugins"""
    
    def __init__(self, config: NeoOSINTConfig):
        self.config = config
        self.logger = logging.getLogger(f"neo_osint.plugin.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass
    
    @abstractmethod
    async def execute(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute plugin logic"""
        pass
    
    async def initialize(self) -> None:
        """Initialize plugin (optional)"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources (optional)"""
        pass


class PluginManager:
    """Plugin management system"""
    
    def __init__(self, config: NeoOSINTConfig):
        self.config = config
        self.logger = logging.getLogger("neo_osint.plugins")
        
        # Plugin storage
        self.active_plugins: Dict[str, OSINTPlugin] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {}
        
        # Plugin directories
        self.plugin_dirs = [
            Path(__file__).parent / "builtin",
            Path(config.workspace_dir) / "plugins"
        ]
        
        # Ensure plugin directories exist
        for plugin_dir in self.plugin_dirs:
            plugin_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_plugins(self) -> None:
        """Load all available plugins"""
        self.logger.info("Loading OSINT plugins...")
        
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                await self._load_plugins_from_directory(plugin_dir)
        
        self.logger.info(f"Loaded {len(self.active_plugins)} plugins")
    
    async def _load_plugins_from_directory(self, plugin_dir: Path) -> None:
        """Load plugins from a specific directory"""
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                await self._load_plugin_from_file(plugin_file)
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file}: {e}")
    
    async def _load_plugin_from_file(self, plugin_file: Path) -> None:
        """Load a single plugin from file"""
        module_name = plugin_file.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {plugin_file}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for plugin classes
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, OSINTPlugin) and 
                attr != OSINTPlugin):
                
                plugin_instance = attr(self.config)
                await plugin_instance.initialize()
                
                self.active_plugins[plugin_instance.name] = plugin_instance
                self.logger.info(f"Loaded plugin: {plugin_instance.name} v{plugin_instance.version}")
    
    async def run_plugins(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all active plugins"""
        if not self.active_plugins:
            return {}
        
        self.logger.info(f"Running {len(self.active_plugins)} plugins")
        
        plugin_results = {}
        
        # Run plugins concurrently
        tasks = []
        for plugin_name, plugin in self.active_plugins.items():
            task = asyncio.create_task(
                self._run_single_plugin(plugin, query, search_results, scraped_content, analysis)
            )
            tasks.append((plugin_name, task))
        
        # Wait for all plugins to complete
        for plugin_name, task in tasks:
            try:
                result = await task
                plugin_results[plugin_name] = result
                self.logger.info(f"Plugin {plugin_name} completed successfully")
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} failed: {e}")
                plugin_results[plugin_name] = {"error": str(e)}
        
        return plugin_results
    
    async def _run_single_plugin(
        self,
        plugin: OSINTPlugin,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single plugin"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await plugin.execute(query, search_results, scraped_content, analysis)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            result["execution_time"] = execution_time
            result["plugin_name"] = plugin.name
            result["plugin_version"] = plugin.version
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return {
                "error": str(e),
                "execution_time": execution_time,
                "plugin_name": plugin.name,
                "plugin_version": plugin.version
            }
    
    def get_plugin_info(self) -> List[Dict[str, str]]:
        """Get information about loaded plugins"""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description
            }
            for plugin in self.active_plugins.values()
        ]
    
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook for specific events"""
        if event not in self.plugin_hooks:
            self.plugin_hooks[event] = []
        self.plugin_hooks[event].append(callback)
    
    async def trigger_hooks(self, event: str, data: Any) -> None:
        """Trigger hooks for a specific event"""
        if event in self.plugin_hooks:
            for hook in self.plugin_hooks[event]:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(data)
                    else:
                        hook(data)
                except Exception as e:
                    self.logger.error(f"Hook failed for event {event}: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all plugins"""
        for plugin in self.active_plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up plugin {plugin.name}: {e}")
        
        self.active_plugins.clear()
        self.plugin_hooks.clear()


# Built-in plugins
class VirusTotalPlugin(OSINTPlugin):
    """VirusTotal integration plugin"""
    
    @property
    def name(self) -> str:
        return "virustotal"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Check indicators against VirusTotal"
    
    async def execute(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute VirusTotal checks"""
        # This would integrate with VirusTotal API
        # For now, return placeholder
        return {
            "indicators_checked": 0,
            "positives": 0,
            "details": "VirusTotal integration not configured"
        }


class ShodanPlugin(OSINTPlugin):
    """Shodan integration plugin"""
    
    @property
    def name(self) -> str:
        return "shodan"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Search Shodan for infrastructure information"
    
    async def execute(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Shodan searches"""
        # This would integrate with Shodan API
        return {
            "hosts_found": 0,
            "services": [],
            "details": "Shodan integration not configured"
        }


class IOCExtractorPlugin(OSINTPlugin):
    """Enhanced IOC extraction plugin"""
    
    @property
    def name(self) -> str:
        return "ioc_extractor"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Enhanced indicator of compromise extraction"
    
    async def execute(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        scraped_content: Dict[str, str],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract additional IOCs"""
        import re
        
        iocs = {
            "cve": [],
            "malware": [],
            "actor": [],
            "ttp": []
        }
        
        # Extract CVEs
        cve_pattern = re.compile(r'CVE-\d{4}-\d{4,7}')
        for content in scraped_content.values():
            iocs["cve"].extend(cve_pattern.findall(content))
        
        # Extract malware names (basic pattern)
        malware_keywords = ["trojan", "backdoor", "ransomware", "botnet", "malware"]
        for content in scraped_content.values():
            for keyword in malware_keywords:
                if keyword in content.lower():
                    iocs["malware"].append(keyword)
        
        # Remove duplicates
        for key in iocs:
            iocs[key] = list(set(iocs[key]))
        
        return {
            "extracted_iocs": iocs,
            "total_iocs": sum(len(v) for v in iocs.values())
        }