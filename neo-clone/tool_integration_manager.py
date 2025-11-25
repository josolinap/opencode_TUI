"""
Tool Integration Manager

Manages the integration of extended MCP tools into the Neo-Clone system.
Provides automatic tool discovery, registration, and management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import MCP components
try:
    from mcp_protocol import MCPClient, MCPConfig, MCPTool
    from extended_mcp_tools import ExtendedMCPTools, ExtendedToolExecutor
    MCP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MCP components not available: {e}")
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class ToolIntegrationManager:
    """Manages integration of extended MCP tools"""
    
    def __init__(self):
        self.mcp_client: Optional[MCPClient] = None
        self.extended_executor = ExtendedToolExecutor()
        self.registered_tools: Dict[str, MCPTool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the tool integration manager"""
        if self._initialized:
            return True
            
        if not MCP_AVAILABLE:
            logger.warning("MCP not available - cannot initialize tool integration")
            return False
            
        try:
            # Initialize MCP client
            config = MCPConfig(
                enable_caching=True,
                enable_discovery=True,
                auto_register_tools=True
            )
            
            self.mcp_client = MCPClient(config)
            await self.mcp_client.start()
            
            # Register extended tools
            await self._register_extended_tools()
            
            self._initialized = True
            logger.info("Tool Integration Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Tool Integration Manager: {e}")
            return False
    
    async def _register_extended_tools(self) -> None:
        """Register all extended MCP tools"""
        try:
            # Get all extended tools
            extended_tools = ExtendedMCPTools.get_all_extended_tools()
            
            # Register each tool
            for tool in extended_tools:
                await self._register_single_tool(tool)
            
            # Organize tools by category
            self._organize_tools_by_category()
            
            logger.info(f"Registered {len(extended_tools)} extended MCP tools")
            
        except Exception as e:
            logger.error(f"Failed to register extended tools: {e}")
    
    async def _register_single_tool(self, tool: MCPTool) -> None:
        """Register a single MCP tool"""
        try:
            if self.mcp_client and hasattr(self.mcp_client, 'registry'):
                self.mcp_client.registry.register_tool(tool)
                self.registered_tools[tool.id] = tool
                logger.debug(f"Registered tool: {tool.name} ({tool.id})")
            else:
                # Fallback registration
                self.registered_tools[tool.id] = tool
                logger.debug(f"Locally registered tool: {tool.name} ({tool.id})")
                
        except Exception as e:
            logger.error(f"Failed to register tool {tool.id}: {e}")
    
    def _organize_tools_by_category(self) -> None:
        """Organize registered tools by category"""
        self.tool_categories.clear()
        
        for tool in self.registered_tools.values():
            category = tool.category
            if category not in self.tool_categories:
                self.tool_categories[category] = []
            self.tool_categories[category].append(tool.id)
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool using the appropriate executor"""
        try:
            # First try MCP client
            if self.mcp_client:
                execution = await self.mcp_client.execute_tool(tool_id, parameters)
                if execution and hasattr(execution, 'result'):
                    return {
                        "success": True,
                        "result": execution.result,
                        "execution_time": execution.execution_time or 0.0
                    }
            
            # Fallback to extended executor
            return await self.extended_executor.execute_tool(tool_id, parameters)
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_tools(self, category: Optional[str] = None) -> List[MCPTool]:
        """Get available tools, optionally filtered by category"""
        if category:
            return [tool for tool in self.registered_tools.values() if tool.category == category]
        return list(self.registered_tools.values())
    
    def get_tool_categories(self) -> List[str]:
        """Get all available tool categories"""
        return list(self.tool_categories.keys())
    
    def get_tools_by_category(self) -> Dict[str, List[str]]:
        """Get tools organized by category"""
        return self.tool_categories.copy()
    
    def search_tools(self, query: str) -> List[MCPTool]:
        """Search tools by name, description, or category"""
        query_lower = query.lower()
        results = []
        
        for tool in self.registered_tools.values():
            # Search in name
            if query_lower in tool.name.lower():
                results.append(tool)
                continue
            
            # Search in description
            if query_lower in tool.description.lower():
                results.append(tool)
                continue
            
            # Search in category
            if query_lower in tool.category.lower():
                results.append(tool)
                continue
        
        return results
    
    def get_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool"""
        if tool_id not in self.registered_tools:
            return None
        
        tool = self.registered_tools[tool_id]
        return {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "security_level": tool.security_level.value if hasattr(tool.security_level, 'value') else str(tool.security_level),
            "parameters": [
                {
                    "name": param.name,
                    "type": param.param_type,
                    "required": param.required,
                    "default": param.default,
                    "description": param.description
                }
                for param in tool.parameters
            ],
            "execution_requirements": tool.execution_requirements,
            "tags": tool.tags
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered tools"""
        return {
            "total_tools": len(self.registered_tools),
            "categories": len(self.tool_categories),
            "tools_by_category": {
                category: len(tools) for category, tools in self.tool_categories.items()
            },
            "security_levels": self._get_security_level_stats(),
            "initialized": self._initialized
        }
    
    def _get_security_level_stats(self) -> Dict[str, int]:
        """Get statistics by security level"""
        stats = {}
        for tool in self.registered_tools.values():
            level = str(tool.security_level)
            stats[level] = stats.get(level, 0) + 1
        return stats
    
    async def add_custom_tool(self, tool: MCPTool) -> bool:
        """Add a custom tool to the registry"""
        try:
            await self._register_single_tool(tool)
            self._organize_tools_by_category()
            logger.info(f"Added custom tool: {tool.name} ({tool.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom tool: {e}")
            return False
    
    async def remove_tool(self, tool_id: str) -> bool:
        """Remove a tool from the registry"""
        try:
            if tool_id in self.registered_tools:
                del self.registered_tools[tool_id]
                self._organize_tools_by_category()
                
                # Also remove from MCP client registry if available
                if self.mcp_client and hasattr(self.mcp_client, 'registry'):
                    self.mcp_client.registry.unregister_tool(tool_id)
                
                logger.info(f"Removed tool: {tool_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove tool: {e}")
            return False
    
    async def refresh_tools(self) -> None:
        """Refresh the tool registry"""
        try:
            # Clear current registry
            self.registered_tools.clear()
            self.tool_categories.clear()
            
            # Re-register extended tools
            await self._register_extended_tools()
            
            logger.info("Tool registry refreshed")
            
        except Exception as e:
            logger.error(f"Failed to refresh tools: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the tool integration manager"""
        try:
            if self.mcp_client:
                await self.mcp_client.stop()
                self.mcp_client = None
            
            self.registered_tools.clear()
            self.tool_categories.clear()
            self._initialized = False
            
            logger.info("Tool Integration Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global instance
tool_integration_manager = ToolIntegrationManager()


async def get_tool_integration_manager() -> ToolIntegrationManager:
    """Get the global tool integration manager instance"""
    if not tool_integration_manager._initialized:
        await tool_integration_manager.initialize()
    return tool_integration_manager


# Convenience functions
async def execute_extended_tool(tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an extended tool"""
    manager = await get_tool_integration_manager()
    return await manager.execute_tool(tool_id, parameters)


async def get_all_extended_tools() -> List[MCPTool]:
    """Get all extended tools"""
    manager = await get_tool_integration_manager()
    return manager.get_available_tools()


async def get_tools_by_category(category: str) -> List[MCPTool]:
    """Get tools by category"""
    manager = await get_tool_integration_manager()
    return manager.get_available_tools(category)


async def search_extended_tools(query: str) -> List[MCPTool]:
    """Search extended tools"""
    manager = await get_tool_integration_manager()
    return manager.search_tools(query)