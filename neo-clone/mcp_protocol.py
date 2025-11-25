"""
MCP Protocol Implementation for Neo-Clone
Model Context Protocol - Enhanced Tool Integration

This module provides MCP protocol support for seamless integration with
external tools, APIs, and services while maintaining security and performance.

Author: Neo-Clone Enhanced
Version: 5.1 (Phase 5 - Enhanced Tool Integration)
"""

import asyncio
import json
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import tempfile
import subprocess
import sys

# Configure logging
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Tool security levels"""
    SAFE = "safe"                    # Read-only operations
    RESTRICTED = "restricted"        # Limited write operations  
    DANGEROUS = "dangerous"          # System-level operations
    CUSTOM = "custom"                 # Custom security rules


class ToolStatus(Enum):
    """Tool execution status"""
    IDLE = "idle"
    DISCOVERING = "discovering"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


@dataclass
class ToolParameter:
    """MCP tool parameter definition"""
    name: str
    param_type: str  # 'string', 'integer', 'boolean', 'list', 'dict'
    required: bool = False
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # Regex pattern for validation


@dataclass
class MCPTool:
    """MCP tool definition"""
    id: str
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    parameters: List[ToolParameter] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.SAFE
    execution_requirements: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    author: str = "MCP Community"
    license: str = "MIT"
    repository: Optional[str] = None
    documentation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'version': self.version,
            'parameters': [
                {
                    'name': p.name,
                    'type': p.param_type,
                    'required': p.required,
                    'default': p.default,
                    'description': p.description,
                    'choices': p.choices,
                    'min_value': p.min_value,
                    'max_value': p.max_value,
                    'pattern': p.pattern
                } for p in self.parameters
            ],
            'security_level': self.security_level.value,
            'execution_requirements': self.execution_requirements,
            'tags': self.tags,
            'author': self.author,
            'license': self.license,
            'repository': self.repository,
            'documentation': self.documentation
        }


@dataclass
class ToolExecution:
    """Tool execution record"""
    execution_id: str
    tool_id: str
    parameters: Dict[str, Any]
    status: ToolStatus = ToolStatus.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    security_context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class SecurityConfig:
    """Security configuration for tool execution"""
    max_execution_time: int = 60  # seconds
    max_memory_usage: int = 512   # MB
    allowed_network_hosts: List[str] = field(default_factory=list)
    blocked_file_patterns: List[str] = field(default_factory=list)
    require_approval_for: List[str] = field(default_factory=list)
    audit_log_enabled: bool = True
    sandbox_enabled: bool = True
    allow_file_system: bool = False
    allow_network_access: bool = False
    allow_system_commands: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """Sandbox configuration for tool execution"""
    allow_network: bool = False
    allow_file_system: bool = False
    allow_system_commands: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    blocked_commands: List[str] = field(default_factory=list)
    allowed_paths: List[str] = field(default_factory=list)


class SecurityManager:
    """Security manager for MCP tool execution"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.audit_log: List[Dict[str, Any]] = []
        
    def validate_execution(self, tool: MCPTool, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tool execution permissions"""
        try:
            # Check security level
            if tool.security_level == SecurityLevel.DANGEROUS:
                if tool.id not in self.config.require_approval_for:
                    return False, f"Dangerous tool '{tool.id}' requires explicit approval"
            
            # Validate parameters
            for param in tool.parameters:
                if param.required and param.name not in parameters:
                    return False, f"Required parameter '{param.name}' is missing"
                
                if param.name in parameters:
                    value = parameters[param.name]
                    
                    # Type validation
                    if not self._validate_parameter_type(value, param.param_type):
                        return False, f"Parameter '{param.name}' has invalid type"
                    
                    # Range validation
                    if param.min_value is not None and value < param.min_value:
                        return False, f"Parameter '{param.name}' below minimum value"
                    
                    if param.max_value is not None and value > param.max_value:
                        return False, f"Parameter '{param.name}' above maximum value"
                    
                    # Pattern validation
                    if param.pattern and not self._validate_pattern(value, param.pattern):
                        return False, f"Parameter '{param.name}' does not match required pattern"
            
            # Check execution requirements
            if tool.execution_requirements.get('network', False) and not self.config.allow_network_access:
                return False, "Tool requires network access but it's disabled"
            
            if tool.execution_requirements.get('file_system', False) and not self.config.allow_file_system:
                return False, "Tool requires file system access but it's disabled"
            
            if tool.execution_requirements.get('system_commands', False) and not self.config.allow_system_commands:
                return False, "Tool requires system commands but they're disabled"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return False, f"Security validation failed: {str(e)}"
    
    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Validate parameter type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow
        
        return isinstance(value, expected_python_type)
    
    def _validate_pattern(self, value: Any, pattern: str) -> bool:
        """Validate parameter against regex pattern"""
        try:
            import re
            return bool(re.match(pattern, str(value)))
        except Exception:
            return True  # Pattern error, allow
    
    def create_sandbox(self, tool: MCPTool) -> SandboxConfig:
        """Create sandbox configuration for tool"""
        return SandboxConfig(
            allow_network=tool.execution_requirements.get('network', False) and self.config.allow_network_access,
            allow_file_system=tool.execution_requirements.get('file_system', False) and self.config.allow_file_system,
            allow_system_commands=tool.execution_requirements.get('system_commands', False) and self.config.allow_system_commands,
            resource_limits=self.config.resource_limits,
            timeout_seconds=min(tool.execution_requirements.get('timeout', 30), self.config.max_execution_time),
            working_directory=tempfile.mkdtemp(prefix=f"mcp_sandbox_{tool.id}_"),
            environment_variables=self._get_safe_environment(),
            blocked_commands=self.config.blocked_file_patterns,
            allowed_paths=self._get_allowed_paths(tool)
        )
    
    def _get_safe_environment(self) -> Dict[str, str]:
        """Get safe environment variables for sandbox"""
        # Only include safe environment variables
        safe_vars = ['PATH', 'HOME', 'USER', 'LANG', 'LC_ALL']
        return {k: v for k, v in os.environ.items() if k in safe_vars}
    
    def _get_allowed_paths(self, tool: MCPTool) -> List[str]:
        """Get allowed paths for tool execution"""
        allowed = []
        
        # Add temp directory
        allowed.append(tempfile.gettempdir())
        
        # Add tool-specific paths from requirements
        tool_paths = tool.execution_requirements.get('allowed_paths', [])
        allowed.extend(tool_paths)
        
        # Filter blocked patterns
        filtered = []
        for path in allowed:
            if not any(pattern in path for pattern in self.config.blocked_file_patterns):
                filtered.append(path)
        
        return filtered
    
    def log_execution(self, execution: ToolExecution) -> None:
        """Log tool execution for audit"""
        if self.config.audit_log_enabled:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'execution_id': execution.execution_id,
                'tool_id': execution.tool_id,
                'parameters': execution.parameters,
                'status': execution.status.value,
                'start_time': execution.start_time.isoformat() if execution.start_time else None,
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'execution_time': execution.execution_time,
                'error': execution.error
            }
            self.audit_log.append(log_entry)
            
            # Keep audit log size manageable
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-5000:]


class ToolRegistry:
    """Registry for MCP tools"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.categories: Dict[str, List[str]] = {}
        self.last_update: Optional[datetime] = None
        
    def register_tool(self, tool: MCPTool) -> None:
        """Register a new tool"""
        self.tools[tool.id] = tool
        
        # Update category mapping
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        if tool.id not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.id)
        
        self.last_update = datetime.now()
        logger.info(f"Registered MCP tool: {tool.name} ({tool.id})")
    
    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool"""
        if tool_id not in self.tools:
            return False
        
        tool = self.tools[tool_id]
        
        # Remove from category mapping
        if tool.category in self.categories and tool_id in self.categories[tool.category]:
            self.categories[tool.category].remove(tool_id)
        
        # Remove from registry
        del self.tools[tool_id]
        self.last_update = datetime.now()
        
        logger.info(f"Unregistered MCP tool: {tool.name} ({tool_id})")
        return True
    
    def get_tool(self, tool_id: str) -> Optional[MCPTool]:
        """Get tool by ID"""
        return self.tools.get(tool_id)
    
    def list_tools(self, category: Optional[str] = None, security_level: Optional[SecurityLevel] = None) -> List[MCPTool]:
        """List tools with optional filtering"""
        tools = list(self.tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if security_level:
            tools = [t for t in tools if t.security_level == security_level]
        
        return tools
    
    def search_tools(self, query: str) -> List[MCPTool]:
        """Search tools by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for tool in self.tools.values():
            # Search in name
            if query_lower in tool.name.lower():
                results.append(tool)
                continue
            
            # Search in description
            if query_lower in tool.description.lower():
                results.append(tool)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in tool.tags):
                results.append(tool)
        
        return results
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_tools': len(self.tools),
            'categories': len(self.categories),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'tools_by_category': {cat: len(tools) for cat, tools in self.categories.items()},
            'tools_by_security': {
                level.value: len([t for t in self.tools.values() if t.security_level == level])
                for level in SecurityLevel
            }
        }


# Import os for environment variables
import os


@dataclass
class MCPMessage:
    """MCP protocol message"""
    message_id: str
    message_type: str  # 'tool_call', 'tool_response', 'error', 'discovery', 'register'
    tool_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type,
            'tool_id': self.tool_id,
            'parameters': self.parameters,
            'result': self.result,
            'error': self.error,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create from dictionary"""
        return cls(
            message_id=data['message_id'],
            message_type=data['message_type'],
            tool_id=data.get('tool_id'),
            parameters=data.get('parameters'),
            result=data.get('result'),
            error=data.get('error'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


@dataclass
class MCPConfig:
    """Configuration for MCP client"""
    registry_url: str = "https://registry.mcp.dev"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_discovery: bool = True
    auto_register_tools: bool = True
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    parallel_execution: bool = True
    max_concurrent_executions: int = 5


class MCPClient:
    """MCP protocol client for Neo-Clone"""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.registry = ToolRegistry()
        self.security = SecurityManager(config.security_config)
        self.executor = None  # Will be initialized later
        self.message_handlers: Dict[str, callable] = {}
        self.execution_history: List[ToolExecution] = []
        self.cache: Dict[str, Any] = {}
        self._running = False
        self._execution_semaphore = None
        
        # Register default message handlers
        self._register_default_handlers()
        
        logger.info("MCP Client initialized")
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers"""
        self.message_handlers.update({
            'tool_call': self._handle_tool_call,
            'tool_response': self._handle_tool_response,
            'error': self._handle_error,
            'discovery': self._handle_discovery,
            'register': self._handle_register
        })
    
    async def start(self) -> None:
        """Start the MCP client"""
        if self._running:
            logger.warning("MCP Client is already running")
            return
        
        self._running = True
        self._execution_semaphore = asyncio.Semaphore(self.config.max_concurrent_executions)
        
        # Initialize executor
        self.executor = SecureExecutor(self.security)
        
        # Start background tasks
        asyncio.create_task(self._discovery_loop())
        asyncio.create_task(self._cache_cleanup_loop())
        
        logger.info("MCP Client started")
    
    async def stop(self) -> None:
        """Stop the MCP client"""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for all executions to complete
        if self._execution_semaphore:
            # Release all semaphore slots
            for _ in range(self.config.max_concurrent_executions):
                self._execution_semaphore.release()
        
        logger.info("MCP Client stopped")
    
    async def discover_tools(self) -> List[MCPTool]:
        """Discover available MCP tools"""
        try:
            # Check cache first
            cache_key = "discovered_tools"
            if self.config.enable_caching and cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.config.cache_ttl_seconds:
                    logger.debug("Returning cached tools")
                    return cached_data['tools']
            
            # Perform discovery
            tools = await self._perform_discovery()
            
            # Cache results
            if self.config.enable_caching:
                self.cache[cache_key] = {
                    'tools': tools,
                    'timestamp': time.time()
                }
            
            # Auto-register tools if enabled
            if self.config.auto_register_tools:
                for tool in tools:
                    self.registry.register_tool(tool)
            
            logger.info(f"Discovered {len(tools)} MCP tools")
            return tools
            
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return []
    
    async def _perform_discovery(self) -> List[MCPTool]:
        """Perform actual tool discovery"""
        tools = []
        
        try:
            # For now, create some example tools
            # In a real implementation, this would query the MCP registry
            example_tools = [
                MCPTool(
                    id="mcp_file_reader",
                    name="File Reader",
                    description="Read file contents securely",
                    category="file_system",
                    parameters=[
                        ToolParameter("file_path", "string", True, description="Path to file to read"),
                        ToolParameter("encoding", "string", False, "utf-8", description="File encoding")
                    ],
                    security_level=SecurityLevel.SAFE,
                    execution_requirements={"file_system": True}
                ),
                MCPTool(
                    id="mcp_web_fetch",
                    name="Web Fetch",
                    description="Fetch content from web URLs",
                    category="web",
                    parameters=[
                        ToolParameter("url", "string", True, description="URL to fetch"),
                        ToolParameter("timeout", "integer", False, 30, description="Request timeout")
                    ],
                    security_level=SecurityLevel.RESTRICTED,
                    execution_requirements={"network": True}
                ),
                MCPTool(
                    id="mcp_data_analyzer",
                    name="Data Analyzer",
                    description="Analyze structured data",
                    category="data_processing",
                    parameters=[
                        ToolParameter("data", "dict", True, description="Data to analyze"),
                        ToolParameter("analysis_type", "string", False, "summary", description="Type of analysis")
                    ],
                    security_level=SecurityLevel.SAFE
                )
            ]
            
            tools.extend(example_tools)
            
        except Exception as e:
            logger.error(f"Discovery error: {e}")
        
        return tools
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> ToolExecution:
        """Execute tool with security controls"""
        execution_id = str(uuid.uuid4())
        execution = ToolExecution(
            execution_id=execution_id,
            tool_id=tool_id,
            parameters=parameters
        )
        
        try:
            # Acquire semaphore for concurrent execution control
            async with self._execution_semaphore:
                execution.status = ToolStatus.EXECUTING
                execution.start_time = datetime.now()
                
                # Get tool
                tool = self.registry.get_tool(tool_id)
                if not tool:
                    raise ValueError(f"Tool '{tool_id}' not found")
                
                # Security validation
                is_valid, error_msg = self.security.validate_execution(tool, parameters)
                if not is_valid:
                    raise PermissionError(f"Security validation failed: {error_msg}")
                
                # Create sandbox
                sandbox_config = self.security.create_sandbox(tool)
                
                # Execute tool
                result = await self.executor.execute_tool(tool, parameters, sandbox_config)
                
                # Update execution record
                execution.result = result
                execution.status = ToolStatus.COMPLETED
                execution.end_time = datetime.now()
                execution.execution_time = execution.duration
                
                # Log execution
                self.security.log_execution(execution)
                self.execution_history.append(execution)
                
                # Keep history manageable
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-500:]
                
                logger.info(f"Tool '{tool_id}' executed successfully in {execution.execution_time:.2f}s")
                
        except Exception as e:
            execution.status = ToolStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            execution.execution_time = execution.duration
            
            self.security.log_execution(execution)
            self.execution_history.append(execution)
            
            logger.error(f"Tool execution failed: {e}")
        
        return execution
    
    async def send_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Send message and handle response"""
        try:
            # Handle message locally
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Message handling failed: {e}")
            return MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type="error",
                error=str(e)
            )
    
    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle tool call message"""
        try:
            execution = await self.execute_tool(message.tool_id, message.parameters or {})
            
            return MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type="tool_response",
                tool_id=message.tool_id,
                result=execution.result,
                metadata={
                    'execution_id': execution.execution_id,
                    'execution_time': execution.execution_time,
                    'status': execution.status.value
                }
            )
            
        except Exception as e:
            return MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type="error",
                tool_id=message.tool_id,
                error=str(e)
            )
    
    async def _handle_tool_response(self, message: MCPMessage) -> MCPMessage:
        """Handle tool response message"""
        # Log response
        logger.debug(f"Received tool response for {message.tool_id}")
        return message
    
    async def _handle_error(self, message: MCPMessage) -> MCPMessage:
        """Handle error message"""
        logger.error(f"MCP Error: {message.error}")
        return message
    
    async def _handle_discovery(self, message: MCPMessage) -> MCPMessage:
        """Handle discovery message"""
        tools = await self.discover_tools()
        
        return MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type="tool_response",
            result={
                'tools': [tool.to_dict() for tool in tools],
                'count': len(tools)
            }
        )
    
    async def _handle_register(self, message: MCPMessage) -> MCPMessage:
        """Handle tool registration message"""
        try:
            tool_data = message.parameters.get('tool')
            if tool_data:
                # Convert dict to MCPTool
                tool = MCPTool(
                    id=tool_data['id'],
                    name=tool_data['name'],
                    description=tool_data['description'],
                    category=tool_data['category'],
                    security_level=SecurityLevel(tool_data.get('security_level', 'safe'))
                )
                
                self.registry.register_tool(tool)
                
                return MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type="tool_response",
                    result={'registered': True, 'tool_id': tool.id}
                )
            else:
                raise ValueError("No tool data provided")
                
        except Exception as e:
            return MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type="error",
                error=str(e)
            )
    
    async def _discovery_loop(self) -> None:
        """Background discovery loop"""
        while self._running:
            try:
                if self.config.enable_discovery:
                    await self.discover_tools()
                
                # Sleep for a reasonable interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(60)  # 1 minute on error
    
    async def _cache_cleanup_loop(self) -> None:
        """Background cache cleanup loop"""
        while self._running:
            try:
                if self.config.enable_caching:
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, data in self.cache.items():
                        if isinstance(data, dict) and 'timestamp' in data:
                            if current_time - data['timestamp'] > self.config.cache_ttl_seconds:
                                expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        logger.debug(f"Expired cache entry: {key}")
                
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        return {
            'running': self._running,
            'connected': True,  # For now, always true
            'tools_registered': len(self.registry.tools),
            'last_update': self.registry.last_update.isoformat() if self.registry.last_update else None
        }
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [tool.to_dict() for tool in self.registry.tools.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'running': self._running,
            'registry_stats': self.registry.get_stats(),
            'execution_history_size': len(self.execution_history),
            'cache_size': len(self.cache),
            'config': {
                'max_concurrent_executions': self.config.max_concurrent_executions,
                'enable_caching': self.config.enable_caching,
                'enable_discovery': self.config.enable_discovery
            }
        }


class SecureExecutor:
    """Secure tool execution environment"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        
    async def execute_tool(self, tool: MCPTool, parameters: Dict[str, Any], 
                          sandbox_config: SandboxConfig) -> Any:
        """Execute tool in secure environment"""
        try:
            # Route to appropriate executor based on tool category
            if tool.category == "file_system":
                return await self._execute_file_tool(tool, parameters, sandbox_config)
            elif tool.category == "web":
                return await self._execute_web_tool(tool, parameters, sandbox_config)
            elif tool.category == "data_processing":
                return await self._execute_data_tool(tool, parameters, sandbox_config)
            else:
                return await self._execute_generic_tool(tool, parameters, sandbox_config)
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise
    
    async def _execute_file_tool(self, tool: MCPTool, parameters: Dict[str, Any], 
                               sandbox_config: SandboxConfig) -> Any:
        """Execute file system tool"""
        file_path = parameters.get('file_path')
        
        if not file_path:
            raise Exception("file_path parameter is required")
        
        if tool.id == "mcp_file_reader":
            try:
                with open(file_path, 'r', encoding=parameters.get('encoding', 'utf-8')) as f:
                    content = f.read()
                return {'content': content, 'size': len(content)}
            except Exception as e:
                raise Exception(f"Failed to read file: {e}")
        
        raise NotImplementedError(f"File tool '{tool.id}' not implemented")
    
    async def _execute_web_tool(self, tool: MCPTool, parameters: Dict[str, Any], 
                              sandbox_config: SandboxConfig) -> Any:
        """Execute web tool"""
        if tool.id == "mcp_web_fetch":
            url = parameters.get('url')
            timeout = parameters.get('timeout', 30)
            
            try:
                # Try to use aiohttp first, fallback to urllib
                try:
                    import aiohttp
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                        async with session.get(url) as response:
                            content = await response.text()
                            return {
                                'content': content,
                                'status_code': response.status,
                                'headers': dict(response.headers)
                            }
                except ImportError:
                    # Fallback to urllib for synchronous operation
                    import urllib.request
                    import urllib.error
                    
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        content = response.read().decode('utf-8')
                        return {
                            'content': content,
                            'status_code': response.getcode(),
                            'headers': dict(response.headers) if hasattr(response, 'headers') else {}
                        }
            except Exception as e:
                raise Exception(f"Failed to fetch URL: {e}")
        
        raise NotImplementedError(f"Web tool '{tool.id}' not implemented")
    
    async def _execute_data_tool(self, tool: MCPTool, parameters: Dict[str, Any], 
                               sandbox_config: SandboxConfig) -> Any:
        """Execute data processing tool"""
        data = parameters.get('data')
        analysis_type = parameters.get('analysis_type', 'summary')
        
        if tool.id == "mcp_data_analyzer":
            try:
                if analysis_type == "summary":
                    if isinstance(data, dict):
                        return {
                            'type': 'dictionary',
                            'keys': list(data.keys()),
                            'size': len(data)
                        }
                    elif isinstance(data, list):
                        return {
                            'type': 'list',
                            'length': len(data),
                            'sample': data[:5] if len(data) > 0 else []
                        }
                    else:
                        return {
                            'type': type(data).__name__,
                            'value': str(data)[:100]
                        }
                else:
                    return {'message': f'Analysis type "{analysis_type}" not implemented'}
            except Exception as e:
                raise Exception(f"Data analysis failed: {e}")
        
        raise NotImplementedError(f"Data tool '{tool.id}' not implemented")
    
    async def _execute_generic_tool(self, tool: MCPTool, parameters: Dict[str, Any], 
                                  sandbox_config: SandboxConfig) -> Any:
        """Execute generic tool"""
        # For now, return a placeholder result
        return {
            'tool_id': tool.id,
            'parameters': parameters,
            'message': 'Generic tool execution (placeholder)'
        }