"""
Skills Integration for Distributed Tracing and Monitoring

This module integrates distributed tracing and monitoring capabilities
with Neo-Clone skills execution for OpenCode TUI.

Features:
- Automatic tracing of all skill executions
- Performance metrics for individual skills
- Error tracking and alerting for skills
- Resource usage monitoring per skill
- Integration with existing skills architecture
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from functools import wraps

from .distributed_tracing import (
    get_distributed_tracer, 
    TraceOperationType,
    trace_skill_execution
)
from .metrics_collector import (
    get_metrics_collector,
    increment_skill_execution_count,
    record_skill_execution_duration
)

logger = logging.getLogger(__name__)


class TracedSkill:
    """
    Wrapper for individual skills with distributed tracing
    
    Provides automatic tracing and metrics collection for skill executions
    without modifying the core skill implementation.
    """
    
    def __init__(self, skill_name: str, skill_func: Callable, skill_description: str = ""):
        """
        Initialize traced skill
        
        Args:
            skill_name: Name of the skill
            skill_func: The skill function to wrap
            skill_description: Description of the skill
        """
        self.skill_name = skill_name
        self.skill_func = skill_func
        self.skill_description = skill_description
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_metrics_collector()
        
        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_duration = 0.0
        self.last_execution: Optional[datetime] = None
        
        logger.info(f"TracedSkill initialized: {skill_name}")
    
    async def execute_async(self, *args, **kwargs) -> Any:
        """
        Execute skill asynchronously with full tracing
        
        Args:
            *args: Arguments to pass to skill function
            **kwargs: Keyword arguments to pass to skill function
            
        Returns:
            Result of skill execution
        """
        start_time = time.time()
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        # Extract context information
        context = kwargs.get('context', {})
        session_id = context.get('session_id', 'unknown')
        user_input = context.get('user_input', '')
        
        with self.tracer.trace_span(
            f"skill.{self.skill_name}",
            TraceOperationType.SKILL_EXECUTION,
            tags={
                "skill_name": self.skill_name,
                "session_id": session_id,
                "input_length": len(user_input),
                "execution_type": "async",
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
        ):
            try:
                # Log skill execution start
                logger.info(f"Executing skill {self.skill_name} for session {session_id}")
                
                # Execute the actual skill function
                if asyncio.iscoroutinefunction(self.skill_func):
                    result = await self.skill_func(*args, **kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.skill_func, *args, **kwargs)
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                self.success_count += 1
                self.total_duration += duration_ms
                
                increment_skill_execution_count(self.skill_name, True, {
                    "execution_type": "async",
                    "session_id": session_id
                })
                record_skill_execution_duration(self.skill_name, duration_ms, {
                    "success": "true",
                    "session_id": session_id
                })
                
                logger.info(f"Skill {self.skill_name} executed successfully in {duration_ms:.1f}ms")
                return result
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                self.error_count += 1
                self.total_duration += duration_ms
                
                increment_skill_execution_count(self.skill_name, False, {
                    "execution_type": "async",
                    "session_id": session_id,
                    "error_type": type(e).__name__
                })
                record_skill_execution_duration(self.skill_name, duration_ms, {
                    "success": "false",
                    "session_id": session_id,
                    "error_type": type(e).__name__
                })
                
                logger.error(f"Skill {self.skill_name} failed: {e}")
                raise
    
    def execute_sync(self, *args, **kwargs) -> Any:
        """
        Execute skill synchronously with full tracing
        
        Args:
            *args: Arguments to pass to skill function
            **kwargs: Keyword arguments to pass to skill function
            
        Returns:
            Result of skill execution
        """
        start_time = time.time()
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        # Extract context information
        context = kwargs.get('context', {})
        session_id = context.get('session_id', 'unknown')
        user_input = context.get('user_input', '')
        
        with self.tracer.trace_span(
            f"skill.{self.skill_name}",
            TraceOperationType.SKILL_EXECUTION,
            tags={
                "skill_name": self.skill_name,
                "session_id": session_id,
                "input_length": len(user_input),
                "execution_type": "sync",
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
        ):
            try:
                # Log skill execution start
                logger.info(f"Executing skill {self.skill_name} (sync) for session {session_id}")
                
                # Execute the actual skill function
                result = self.skill_func(*args, **kwargs)
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                self.success_count += 1
                self.total_duration += duration_ms
                
                increment_skill_execution_count(self.skill_name, True, {
                    "execution_type": "sync",
                    "session_id": session_id
                })
                record_skill_execution_duration(self.skill_name, duration_ms, {
                    "success": "true",
                    "session_id": session_id
                })
                
                logger.info(f"Skill {self.skill_name} executed successfully in {duration_ms:.1f}ms")
                return result
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                self.error_count += 1
                self.total_duration += duration_ms
                
                increment_skill_execution_count(self.skill_name, False, {
                    "execution_type": "sync",
                    "session_id": session_id,
                    "error_type": type(e).__name__
                })
                record_skill_execution_duration(self.skill_name, duration_ms, {
                    "success": "false",
                    "session_id": session_id,
                    "error_type": type(e).__name__
                })
                
                logger.error(f"Skill {self.skill_name} failed: {e}")
                raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this skill"""
        if self.execution_count == 0:
            return {
                "skill_name": self.skill_name,
                "execution_count": 0,
                "success_rate": 100,
                "error_rate": 0,
                "avg_duration_ms": 0,
                "total_duration_ms": 0
            }
        
        return {
            "skill_name": self.skill_name,
            "skill_description": self.skill_description,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / self.execution_count) * 100,
            "error_rate": (self.error_count / self.execution_count) * 100,
            "avg_duration_ms": self.total_duration / self.execution_count,
            "total_duration_ms": self.total_duration,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }
    
    def __call__(self, *args, **kwargs):
        """Make the skill callable"""
        if asyncio.iscoroutinefunction(self.skill_func):
            return self.execute_async(*args, **kwargs)
        else:
            return self.execute_sync(*args, **kwargs)


class SkillsMonitoringManager:
    """
    Manager for monitoring all skills in the system
    
    Provides centralized monitoring and metrics collection for all skills.
    """
    
    def __init__(self):
        """
        Initialize skills monitoring manager
        """
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_metrics_collector()
        self.traced_skills: Dict[str, TracedSkill] = {}
        self.skill_performance_history: List[Dict[str, Any]] = []
        
        # Setup skill-specific alerts
        self._setup_skill_alerts()
        
        logger.info("SkillsMonitoringManager initialized")
    
    def _setup_skill_alerts(self) -> None:
        """Setup alerts for skill performance"""
        # High skill error rate alert
        self.metrics_collector.create_alert(
            name="skill_error_rate_high",
            metric_name="skill_execution_count",
            condition="gt",  # This would need custom logic for error rate
            threshold=20.0,  # 20% error rate
            severity="warning",
            duration_seconds=120
        )
        
        # Slow skill execution alert
        self.metrics_collector.create_alert(
            name="skill_execution_slow",
            metric_name="skill_execution_duration",
            condition="gt",
            threshold=10000.0,  # 10 seconds
            severity="warning",
            duration_seconds=60
        )
    
    def register_skill(
        self,
        skill_name: str,
        skill_func: Callable,
        skill_description: str = ""
    ) -> TracedSkill:
        """
        Register a skill for monitoring
        
        Args:
            skill_name: Name of the skill
            skill_func: The skill function to monitor
            skill_description: Description of the skill
            
        Returns:
            TracedSkill instance
        """
        traced_skill = TracedSkill(skill_name, skill_func, skill_description)
        self.traced_skills[skill_name] = traced_skill
        
        logger.info(f"Registered skill for monitoring: {skill_name}")
        return traced_skill
    
    def get_skill(self, skill_name: str) -> Optional[TracedSkill]:
        """Get a traced skill by name"""
        return self.traced_skills.get(skill_name)
    
    def get_all_skills_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all registered skills"""
        return {
            skill_name: skill.get_performance_stats()
            for skill_name, skill in self.traced_skills.items()
        }
    
    def get_slow_skills(self, threshold_ms: float = 5000.0) -> List[Dict[str, Any]]:
        """Get list of skills with average execution time above threshold"""
        slow_skills = []
        
        for skill_name, skill in self.traced_skills.items():
            stats = skill.get_performance_stats()
            if stats["avg_duration_ms"] > threshold_ms:
                slow_skills.append({
                    "skill_name": skill_name,
                    "avg_duration_ms": stats["avg_duration_ms"],
                    "execution_count": stats["execution_count"],
                    "error_rate": stats["error_rate"]
                })
        
        return sorted(slow_skills, key=lambda x: x["avg_duration_ms"], reverse=True)
    
    def get_high_error_skills(self, error_rate_threshold: float = 15.0) -> List[Dict[str, Any]]:
        """Get list of skills with error rate above threshold"""
        high_error_skills = []
        
        for skill_name, skill in self.traced_skills.items():
            stats = skill.get_performance_stats()
            if stats["error_rate"] > error_rate_threshold:
                high_error_skills.append({
                    "skill_name": skill_name,
                    "error_rate": stats["error_rate"],
                    "execution_count": stats["execution_count"],
                    "error_count": stats["error_count"]
                })
        
        return sorted(high_error_skills, key=lambda x: x["error_rate"], reverse=True)
    
    def get_most_used_skills(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently used skills"""
        skill_usage = []
        
        for skill_name, skill in self.traced_skills.items():
            stats = skill.get_performance_stats()
            skill_usage.append({
                "skill_name": skill_name,
                "execution_count": stats["execution_count"],
                "success_rate": stats["success_rate"],
                "avg_duration_ms": stats["avg_duration_ms"]
            })
        
        return sorted(skill_usage, key=lambda x: x["execution_count"], reverse=True)[:limit]
    
    def update_performance_history(self) -> None:
        """Update performance history with current stats"""
        current_stats = {
            "timestamp": datetime.now(),
            "total_skills": len(self.traced_skills),
            "skills_performance": self.get_all_skills_performance()
        }
        
        self.skill_performance_history.append(current_stats)
        
        # Keep history manageable
        if len(self.skill_performance_history) > 1000:
            self.skill_performance_history = self.skill_performance_history[-500:]
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        slow_skills = self.get_slow_skills()
        high_error_skills = self.get_high_error_skills()
        most_used_skills = self.get_most_used_skills(5)
        
        return {
            "total_skills_registered": len(self.traced_skills),
            "slow_skills_count": len(slow_skills),
            "high_error_skills_count": len(high_error_skills),
            "slow_skills": slow_skills[:5],  # Top 5 slow skills
            "high_error_skills": high_error_skills[:5],  # Top 5 high error skills
            "most_used_skills": most_used_skills,
            "overall_health": self._calculate_overall_health()
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall health of skills system"""
        if not self.traced_skills:
            return "unknown"
        
        total_executions = sum(skill.execution_count for skill in self.traced_skills.values())
        total_errors = sum(skill.error_count for skill in self.traced_skills.values())
        
        if total_executions == 0:
            return "good"
        
        overall_error_rate = (total_errors / total_executions) * 100
        avg_duration = sum(skill.total_duration for skill in self.traced_skills.values()) / total_executions if total_executions > 0 else 0
        
        # Determine health based on error rate and average duration
        if overall_error_rate > 20 or avg_duration > 10000:
            return "poor"
        elif overall_error_rate > 10 or avg_duration > 5000:
            return "warning"
        else:
            return "good"


def trace_skill(skill_name: str, description: str = ""):
    """
    Decorator for automatically tracing skill functions
    
    Args:
        skill_name: Name of the skill
        description: Description of the skill
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        traced_skill = TracedSkill(skill_name, func, description)
        return traced_skill
    
    return decorator


# Global skills monitoring manager
_skills_monitoring_manager: Optional[SkillsMonitoringManager] = None


def get_skills_monitoring_manager() -> SkillsMonitoringManager:
    """Get global skills monitoring manager"""
    global _skills_monitoring_manager
    
    if _skills_monitoring_manager is None:
        _skills_monitoring_manager = SkillsMonitoringManager()
    
    return _skills_monitoring_manager


# Convenience functions for skill integration
def register_skill_for_monitoring(
    skill_name: str,
    skill_func: Callable,
    description: str = ""
) -> TracedSkill:
    """Register a skill for monitoring"""
    manager = get_skills_monitoring_manager()
    return manager.register_skill(skill_name, skill_func, description)


def get_monitored_skill(skill_name: str) -> Optional[TracedSkill]:
    """Get a monitored skill by name"""
    manager = get_skills_monitoring_manager()
    return manager.get_skill(skill_name)


def get_skills_monitoring_summary() -> Dict[str, Any]:
    """Get skills monitoring summary"""
    manager = get_skills_monitoring_manager()
    return manager.get_monitoring_summary()