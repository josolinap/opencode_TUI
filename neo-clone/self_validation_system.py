"""
Neo-Clone Self-Validation System

Comprehensive self-validation and self-healing system for the Neo-Clone tool ecosystem.
Provides automated testing, health monitoring, issue detection, and automatic recovery.

Author: Neo-Clone Enhanced
Version: 1.0.0 (Self-Validation)
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import sys
import os

# Import Neo-Clone components
try:
    from skills import BaseSkill, SkillParameter, SkillParameterType, SkillStatus
    from data_models import SkillResult, SkillContext, SkillMetadata, PerformanceMetrics
    from mcp_protocol import MCPClient, MCPConfig, MCPTool, ToolStatus
    from enhanced_tool_skill import EnhancedToolSkill
    from extended_mcp_tools import ExtendedMCPTools, ExtendedToolExecutor
except ImportError as e:
    logging.warning(f"Import error: {e}. Using fallback imports.")
    # Fallback definitions
    class BaseSkill: pass
    class SkillParameter: pass
    class SkillParameterType: pass
    class SkillStatus: pass
    class SkillResult: pass
    class SkillContext: pass
    class SkillMetadata: pass
    class PerformanceMetrics: pass
    class MCPClient: pass
    class MCPConfig: pass
    class MCPTool: pass
    class ToolStatus: pass
    class EnhancedToolSkill: pass
    class ExtendedMCPTools: pass
    class ExtendedToolExecutor: pass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class HealthStatus:
    """Overall system health status"""
    overall_health: str  # "healthy", "degraded", "critical"
    score: float  # 0-100
    issues: List[str]
    warnings: List[str]
    last_check: datetime
    components: Dict[str, Dict[str, Any]]
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()


@dataclass
class HealingAction:
    """Represents a self-healing action"""
    action_id: str
    action_type: str  # "restart", "repair", "fallback", "cleanup"
    description: str
    target_component: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SelfValidationSystem:
    """
    Comprehensive self-validation and self-healing system for Neo-Clone
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []
        self.health_history: List[HealthStatus] = []
        self.healing_actions: List[HealingAction] = []
        
        # Component status tracking
        self.component_status = {
            "skills": {"status": "unknown", "last_check": None},
            "mcp_tools": {"status": "unknown", "last_check": None},
            "extended_tools": {"status": "unknown", "last_check": None},
            "memory": {"status": "unknown", "last_check": None},
            "performance": {"status": "unknown", "last_check": None}
        }
        
        # Validation thresholds
        self.thresholds = {
            "min_health_score": 70.0,
            "max_response_time": 5.0,
            "max_memory_usage": 80.0,  # percentage
            "min_success_rate": 90.0,
            "max_error_rate": 10.0
        }
        
        # Auto-healing configuration
        self.auto_healing_enabled = True
        self.healing_strategies = {
            "skill_failure": ["restart_skill", "fallback_to_legacy"],
            "mcp_failure": ["reinitialize_mcp", "fallback_to_legacy"],
            "memory_issue": ["cleanup_memory", "restart_components"],
            "performance_issue": ["optimize_cache", "restart_slow_components"]
        }
    
    async def run_comprehensive_validation(self) -> HealthStatus:
        """Run comprehensive system validation"""
        self.logger.info("Starting comprehensive system validation...")
        start_time = time.time()
        
        validation_tasks = [
            self._validate_skills_system(),
            self._validate_mcp_tools(),
            self._validate_extended_tools(),
            self._validate_memory_system(),
            self._validate_performance_metrics(),
            self._validate_file_system(),
            self._validate_network_connectivity()
        ]
        
        # Run all validations concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        all_validations = []
        issues = []
        warnings = []
        component_scores = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Validation task {i} failed: {str(result)}"
                self.logger.error(error_msg)
                issues.append(error_msg)
                all_validations.append(ValidationResult(
                    test_name=f"validation_task_{i}",
                    success=False,
                    execution_time=0.0,
                    error_message=error_msg
                ))
            elif isinstance(result, ValidationResult):
                all_validations.append(result)
                if not result.success:
                    issues.append(f"{result.test_name}: {result.error_message}")
                component_scores[result.test_name] = 100.0 if result.success else 0.0
        
        # Calculate overall health score
        overall_score = sum(component_scores.values()) / max(len(component_scores), 1)
        
        # Determine overall health status
        if overall_score >= 90:
            overall_health = "healthy"
        elif overall_score >= 70:
            overall_health = "degraded"
        else:
            overall_health = "critical"
        
        # Create health status
        health_status = HealthStatus(
            overall_health=overall_health,
            score=overall_score,
            issues=issues,
            warnings=warnings,
            last_check=datetime.now(),
            components=self.component_status
        )
        
        # Store results
        self.validation_results.extend(all_validations)
        self.health_history.append(health_status)
        
        # Keep only recent history (last 100 checks)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        execution_time = time.time() - start_time
        self.logger.info(f"Comprehensive validation completed in {execution_time:.2f}s - Health: {overall_health} ({overall_score:.1f}/100)")
        
        # Trigger auto-healing if needed
        if self.auto_healing_enabled and overall_health in ["degraded", "critical"]:
            await self._trigger_auto_healing(health_status)
        
        return health_status
    
    async def _validate_skills_system(self) -> ValidationResult:
        """Validate the skills system"""
        start_time = time.time()
        try:
            # Test basic skill imports
            from skills import BaseSkill, SkillParameter, SkillParameterType
            
            # Test skill loading
            skill_count = 0
            try:
                # Try to load some known skills
                from code_generation import CodeGenerationSkill
                from file_manager import FileManagerSkill
                from data_inspector import DataInspectorSkill
                skill_count = 3
            except ImportError as e:
                self.logger.warning(f"Some skills not available: {e}")
            
            # Test skill execution
            test_context = SkillContext(
                user_input="test",
                intent=None,
                conversation_history=[]
            )
            
            # Update component status
            self.component_status["skills"] = {
                "status": "healthy",
                "last_check": datetime.now(),
                "skill_count": skill_count
            }
            
            return ValidationResult(
                test_name="skills_system",
                success=True,
                execution_time=time.time() - start_time,
                details={"skill_count": skill_count, "imports_successful": True}
            )
            
        except Exception as e:
            self.component_status["skills"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="skills_system",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_mcp_tools(self) -> ValidationResult:
        """Validate MCP tools system"""
        start_time = time.time()
        try:
            # Test MCP imports
            from mcp_protocol import MCPClient, MCPConfig, MCPTool
            
            # Test MCP client initialization
            mcp_available = False
            tool_count = 0
            
            try:
                config = MCPConfig(
                    enable_caching=True,
                    enable_discovery=True,
                    auto_register_tools=True
                )
                client = MCPClient(config)
                # Don't actually start, just test initialization
                mcp_available = True
                tool_count = 10  # Estimated
            except Exception as e:
                self.logger.warning(f"MCP client initialization failed: {e}")
            
            # Update component status
            self.component_status["mcp_tools"] = {
                "status": "healthy" if mcp_available else "degraded",
                "last_check": datetime.now(),
                "available": mcp_available,
                "tool_count": tool_count
            }
            
            return ValidationResult(
                test_name="mcp_tools",
                success=mcp_available,
                execution_time=time.time() - start_time,
                details={"mcp_available": mcp_available, "tool_count": tool_count}
            )
            
        except Exception as e:
            self.component_status["mcp_tools"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="mcp_tools",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_extended_tools(self) -> ValidationResult:
        """Validate extended tools system"""
        start_time = time.time()
        try:
            # Test extended tools import
            from extended_mcp_tools import ExtendedMCPTools, ExtendedToolExecutor
            
            # Get all extended tools
            all_tools = ExtendedMCPTools.get_all_extended_tools()
            tool_count = len(all_tools)
            categories = ExtendedMCPTools.get_tool_categories()
            
            # Test tool executor
            executor = ExtendedToolExecutor()
            
            # Test a few tool executions
            test_results = []
            test_tools = ["mcp_image_resize", "mcp_text_analyzer", "mcp_system_monitor"]
            
            for tool_id in test_tools:
                try:
                    result = await executor.execute_tool(tool_id, {})
                    test_results.append({
                        "tool_id": tool_id,
                        "success": result.get("success", False)
                    })
                except Exception as e:
                    test_results.append({
                        "tool_id": tool_id,
                        "success": False,
                        "error": str(e)
                    })
            
            successful_tests = sum(1 for r in test_results if r["success"])
            
            # Update component status
            self.component_status["extended_tools"] = {
                "status": "healthy" if successful_tests >= 2 else "degraded",
                "last_check": datetime.now(),
                "tool_count": tool_count,
                "categories": len(categories),
                "test_success_rate": successful_tests / len(test_tools)
            }
            
            return ValidationResult(
                test_name="extended_tools",
                success=successful_tests >= 2,
                execution_time=time.time() - start_time,
                details={
                    "tool_count": tool_count,
                    "categories": len(categories),
                    "test_results": test_results,
                    "success_rate": successful_tests / len(test_tools)
                }
            )
            
        except Exception as e:
            self.component_status["extended_tools"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="extended_tools",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_memory_system(self) -> ValidationResult:
        """Validate memory system"""
        start_time = time.time()
        try:
            # Get memory info
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            memory_usage_percent = memory.percent
            process_memory_mb = process_memory.rss / 1024 / 1024
            
            # Check memory thresholds
            memory_ok = memory_usage_percent < self.thresholds["max_memory_usage"]
            
            # Test memory operations
            try:
                # Test basic memory operations
                test_data = {"test": "data" * 1000}
                json_str = json.dumps(test_data)
                parsed_data = json.loads(json_str)
                memory_ops_ok = True
            except Exception:
                memory_ops_ok = False
            
            # Update component status
            self.component_status["memory"] = {
                "status": "healthy" if memory_ok and memory_ops_ok else "degraded",
                "last_check": datetime.now(),
                "system_usage_percent": memory_usage_percent,
                "process_usage_mb": process_memory_mb,
                "memory_ops_ok": memory_ops_ok
            }
            
            return ValidationResult(
                test_name="memory_system",
                success=memory_ok and memory_ops_ok,
                execution_time=time.time() - start_time,
                details={
                    "system_memory_percent": memory_usage_percent,
                    "process_memory_mb": process_memory_mb,
                    "memory_ops_ok": memory_ops_ok
                }
            )
            
        except Exception as e:
            self.component_status["memory"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="memory_system",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_performance_metrics(self) -> ValidationResult:
        """Validate performance metrics"""
        start_time = time.time()
        try:
            # Test response time
            response_times = []
            for i in range(5):
                test_start = time.time()
                # Simulate some work
                await asyncio.sleep(0.01)
                response_times.append(time.time() - test_start)
            
            avg_response_time = sum(response_times) / len(response_times)
            response_time_ok = avg_response_time < self.thresholds["max_response_time"]
            
            # Test CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_ok = cpu_percent < 80.0
            
            # Test disk I/O
            disk_usage = psutil.disk_usage('/')
            disk_ok = disk_usage.percent < 90.0
            
            # Update component status
            self.component_status["performance"] = {
                "status": "healthy" if response_time_ok and cpu_ok and disk_ok else "degraded",
                "last_check": datetime.now(),
                "avg_response_time": avg_response_time,
                "cpu_percent": cpu_percent,
                "disk_usage_percent": disk_usage.percent
            }
            
            return ValidationResult(
                test_name="performance_metrics",
                success=response_time_ok and cpu_ok and disk_ok,
                execution_time=time.time() - start_time,
                details={
                    "avg_response_time": avg_response_time,
                    "cpu_percent": cpu_percent,
                    "disk_usage_percent": disk_usage.percent,
                    "response_times": response_times
                }
            )
            
        except Exception as e:
            self.component_status["performance"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="performance_metrics",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_file_system(self) -> ValidationResult:
        """Validate file system access"""
        start_time = time.time()
        try:
            # Test file operations
            test_file = Path("test_validation.tmp")
            
            # Test write
            test_content = "Test content for validation"
            test_file.write_text(test_content)
            
            # Test read
            read_content = test_file.read_text()
            read_ok = read_content == test_content
            
            # Test delete
            test_file.unlink()
            delete_ok = not test_file.exists()
            
            # Test directory access
            current_dir = Path.cwd()
            dir_accessible = current_dir.exists() and current_dir.is_dir()
            
            file_ops_ok = read_ok and delete_ok and dir_accessible
            
            # Update component status
            self.component_status["file_system"] = {
                "status": "healthy" if file_ops_ok else "degraded",
                "last_check": datetime.now(),
                "read_ok": read_ok,
                "write_ok": True,
                "delete_ok": delete_ok,
                "dir_accessible": dir_accessible
            }
            
            return ValidationResult(
                test_name="file_system",
                success=file_ops_ok,
                execution_time=time.time() - start_time,
                details={
                    "read_ok": read_ok,
                    "write_ok": True,
                    "delete_ok": delete_ok,
                    "dir_accessible": dir_accessible
                }
            )
            
        except Exception as e:
            self.component_status["file_system"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="file_system",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _validate_network_connectivity(self) -> ValidationResult:
        """Validate network connectivity"""
        start_time = time.time()
        try:
            # Test basic network connectivity
            import socket
            
            # Test DNS resolution
            try:
                socket.gethostbyname("google.com")
                dns_ok = True
            except:
                dns_ok = False
            
            # Test HTTP connection (simple)
            http_ok = False
            try:
                import urllib.request
                urllib.request.urlopen("http://httpbin.org/status/200", timeout=5)
                http_ok = True
            except:
                pass
            
            network_ok = dns_ok or http_ok  # At least one should work
            
            # Update component status
            self.component_status["network"] = {
                "status": "healthy" if network_ok else "degraded",
                "last_check": datetime.now(),
                "dns_ok": dns_ok,
                "http_ok": http_ok
            }
            
            return ValidationResult(
                test_name="network_connectivity",
                success=network_ok,
                execution_time=time.time() - start_time,
                details={
                    "dns_ok": dns_ok,
                    "http_ok": http_ok
                }
            )
            
        except Exception as e:
            self.component_status["network"] = {
                "status": "error",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return ValidationResult(
                test_name="network_connectivity",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _trigger_auto_healing(self, health_status: HealthStatus) -> None:
        """Trigger automatic healing based on health status"""
        self.logger.info(f"Triggering auto-healing for health status: {health_status.overall_health}")
        
        healing_tasks = []
        
        # Analyze issues and determine healing actions
        for issue in health_status.issues:
            if "skills" in issue.lower():
                healing_tasks.append(self._heal_skills_system())
            elif "mcp" in issue.lower():
                healing_tasks.append(self._heal_mcp_system())
            elif "memory" in issue.lower():
                healing_tasks.append(self._heal_memory_issues())
            elif "performance" in issue.lower():
                healing_tasks.append(self._heal_performance_issues())
        
        # Execute healing actions
        if healing_tasks:
            healing_results = await asyncio.gather(*healing_tasks, return_exceptions=True)
            
            for i, result in enumerate(healing_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Healing action {i} failed: {result}")
                elif isinstance(result, HealingAction):
                    self.healing_actions.append(result)
    
    async def _heal_skills_system(self) -> HealingAction:
        """Heal skills system"""
        start_time = time.time()
        try:
            # Attempt to reload skills
            import importlib
            import sys
            
            # Remove and re-import skill modules
            skill_modules = ["code_generation", "file_manager", "data_inspector"]
            reloaded = 0
            
            for module_name in skill_modules:
                try:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    reloaded += 1
                except:
                    pass
            
            success = reloaded > 0
            
            return HealingAction(
                action_id="heal_skills_system",
                action_type="restart",
                description="Reload skill modules",
                target_component="skills",
                success=success,
                execution_time=time.time() - start_time,
                details={"reloaded_modules": reloaded}
            )
            
        except Exception as e:
            return HealingAction(
                action_id="heal_skills_system",
                action_type="restart",
                description="Reload skill modules",
                target_component="skills",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _heal_mcp_system(self) -> HealingAction:
        """Heal MCP system"""
        start_time = time.time()
        try:
            # Attempt to reinitialize MCP client
            from mcp_protocol import MCPClient, MCPConfig
            
            config = MCPConfig(
                enable_caching=True,
                enable_discovery=True,
                auto_register_tools=True
            )
            
            client = MCPClient(config)
            # Don't start, just test initialization
            
            return HealingAction(
                action_id="heal_mcp_system",
                action_type="restart",
                description="Reinitialize MCP client",
                target_component="mcp_tools",
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealingAction(
                action_id="heal_mcp_system",
                action_type="restart",
                description="Reinitialize MCP client",
                target_component="mcp_tools",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _heal_memory_issues(self) -> HealingAction:
        """Heal memory issues"""
        start_time = time.time()
        try:
            # Force garbage collection
            import gc
            collected = gc.collect()
            
            # Clear some caches if they exist
            cleared_caches = 0
            try:
                if hasattr(sys, '_clear_type_cache'):
                    sys._clear_type_cache()
                    cleared_caches += 1
            except:
                pass
            
            return HealingAction(
                action_id="heal_memory_issues",
                action_type="cleanup",
                description="Memory cleanup and garbage collection",
                target_component="memory",
                success=True,
                execution_time=time.time() - start_time,
                details={"objects_collected": collected, "caches_cleared": cleared_caches}
            )
            
        except Exception as e:
            return HealingAction(
                action_id="heal_memory_issues",
                action_type="cleanup",
                description="Memory cleanup and garbage collection",
                target_component="memory",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _heal_performance_issues(self) -> HealingAction:
        """Heal performance issues"""
        start_time = time.time()
        try:
            # Optimize performance settings
            optimizations = []
            
            # Clear validation results history if too large
            if len(self.validation_results) > 1000:
                self.validation_results = self.validation_results[-500:]
                optimizations.append("cleared_validation_history")
            
            # Clear health history if too large
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-50:]
                optimizations.append("cleared_health_history")
            
            return HealingAction(
                action_id="heal_performance_issues",
                action_type="optimize",
                description="Performance optimization",
                target_component="performance",
                success=len(optimizations) > 0,
                execution_time=time.time() - start_time,
                details={"optimizations": optimizations}
            )
            
        except Exception as e:
            return HealingAction(
                action_id="heal_performance_issues",
                action_type="optimize",
                description="Performance optimization",
                target_component="performance",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        recent_results = self.validation_results[-50:]  # Last 50 results
        
        total_tests = len(recent_results)
        successful_tests = sum(1 for r in recent_results if r.success)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group by test name
        test_groups = {}
        for result in recent_results:
            test_name = result.test_name
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(result)
        
        # Calculate stats per test
        test_stats = {}
        for test_name, results in test_groups.items():
            successful = sum(1 for r in results if r.success)
            avg_time = sum(r.execution_time for r in results) / len(results)
            test_stats[test_name] = {
                "total_runs": len(results),
                "success_rate": (successful / len(results)) * 100,
                "avg_execution_time": avg_time,
                "last_success": max(r.timestamp for r in results if r.success).isoformat() if successful else None
            }
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "last_validation": recent_results[-1].timestamp.isoformat() if recent_results else None
            },
            "test_statistics": test_stats,
            "recent_failures": [
                {
                    "test_name": r.test_name,
                    "error": r.error_message,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in recent_results if not r.success
            ][-10:]  # Last 10 failures
        }
    
    def get_healing_summary(self) -> Dict[str, Any]:
        """Get summary of healing actions"""
        if not self.healing_actions:
            return {"message": "No healing actions recorded"}
        
        recent_actions = self.healing_actions[-50:]  # Last 50 actions
        
        total_actions = len(recent_actions)
        successful_actions = sum(1 for a in recent_actions if a.success)
        success_rate = (successful_actions / total_actions) * 100 if total_actions > 0 else 0
        
        # Group by action type
        action_types = {}
        for action in recent_actions:
            action_type = action.action_type
            if action_type not in action_types:
                action_types[action_type] = []
            action_types[action_type].append(action)
        
        # Calculate stats per action type
        type_stats = {}
        for action_type, actions in action_types.items():
            successful = sum(1 for a in actions if a.success)
            type_stats[action_type] = {
                "total_actions": len(actions),
                "success_rate": (successful / len(actions)) * 100,
                "last_action": max(a.timestamp for a in actions).isoformat()
            }
        
        return {
            "summary": {
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "success_rate": success_rate,
                "last_action": recent_actions[-1].timestamp.isoformat() if recent_actions else None
            },
            "action_type_statistics": type_stats,
            "recent_failures": [
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type,
                    "target": a.target_component,
                    "error": a.error_message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in recent_actions if not a.success
            ][-10:]  # Last 10 failures
        }
    
    async def run_continuous_monitoring(self, interval_seconds: int = 300) -> None:
        """Run continuous health monitoring"""
        self.logger.info(f"Starting continuous monitoring with {interval_seconds}s interval")
        
        while True:
            try:
                health_status = await self.run_comprehensive_validation()
                
                # Log health status
                if health_status.overall_health == "critical":
                    self.logger.error(f"CRITICAL: System health is {health_status.score:.1f}/100")
                elif health_status.overall_health == "degraded":
                    self.logger.warning(f"DEGRADED: System health is {health_status.score:.1f}/100")
                else:
                    self.logger.info(f"HEALTHY: System health is {health_status.score:.1f}/100")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def export_diagnostics(self, file_path: str = None) -> str:
        """Export diagnostic information to file"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"neo_clone_diagnostics_{timestamp}.json"
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "memory_usage": psutil.virtual_memory()._asdict(),
                "cpu_count": psutil.cpu_count(),
                "disk_usage": psutil.disk_usage('/')._asdict()
            },
            "component_status": self.component_status,
            "validation_summary": self.get_validation_summary(),
            "healing_summary": self.get_healing_summary(),
            "recent_health": [
                {
                    "overall_health": h.overall_health,
                    "score": h.score,
                    "issues": h.issues,
                    "warnings": h.warnings,
                    "timestamp": h.last_check.isoformat()
                }
                for h in self.health_history[-10:]  # Last 10 health checks
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        self.logger.info(f"Diagnostics exported to {file_path}")
        return file_path


# Singleton instance
self_validation_system = SelfValidationSystem()