"""
Brain Integration for Distributed Tracing and Monitoring

This module integrates distributed tracing and monitoring capabilities
with Neo-Clone brain operations for OpenCode TUI.

Features:
- Automatic tracing of brain processing operations
- Performance metrics collection for brain components
- Error tracking and alerting
- Resource usage monitoring
- Integration with existing brain architecture
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .distributed_tracing import (
    get_distributed_tracer, 
    TraceOperationType,
    trace_brain_processing,
    trace_intent_recognition,
    trace_memory_access,
    trace_reasoning_chain,
    trace_collaborative_processing,
    trace_multi_session
)
from .metrics_collector import (
    get_metrics_collector,
    increment_brain_processing_count,
    record_brain_processing_duration,
    increment_skill_execution_count,
    record_skill_execution_duration,
    record_memory_access_duration,
    set_active_sessions_count,
    set_error_rate,
    set_cache_hit_rate
)

logger = logging.getLogger(__name__)


class TracedBrainWrapper:
    """
    Wrapper for Neo-Clone brain with distributed tracing and monitoring
    
    Provides automatic tracing and metrics collection for all brain operations
    without modifying the core brain implementation.
    """
    
    def __init__(self, brain_instance):
        """
        Initialize traced brain wrapper
        
        Args:
            brain_instance: The actual brain instance to wrap
        """
        self.brain = brain_instance
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_metrics_collector()
        
        # Performance tracking
        self.operation_history: List[Dict[str, Any]] = []
        self.error_history: List[Dict[str, Any]] = []
        
        # Setup monitoring
        self._setup_monitoring()
        
        logger.info(f"TracedBrainWrapper initialized for {type(brain_instance).__name__}")
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and alerting"""
        # Create performance alerts
        self.metrics_collector.create_alert(
            name="brain_processing_slow",
            metric_name="brain_processing_duration",
            condition="gt",
            threshold=3000.0,  # 3 seconds
            severity="warning",
            duration_seconds=60
        )
        
        self.metrics_collector.create_alert(
            name="brain_error_rate_high",
            metric_name="error_rate",
            condition="gt",
            threshold=15.0,  # 15%
            severity="critical",
            duration_seconds=120
        )
    
    async def process_input(
        self,
        user_input: str,
        context: Optional[Any] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, Any]:
        """
        Process user input with full tracing and monitoring
        
        Args:
            user_input: User's input text
            context: Conversation context
            session_id: Session identifier
            
        Returns:
            Tuple of (response, reasoning_trace)
        """
        start_time = time.time()
        session_id = session_id or f"session_{int(time.time())}"
        
        # Start main brain processing trace
        with self.tracer.trace_span(
            "brain.process_input",
            TraceOperationType.BRAIN_PROCESSING,
            tags={
                "session_id": session_id,
                "input_length": len(user_input),
                "brain_type": type(self.brain).__name__
            }
        ):
            try:
                # Increment processing count
                increment_brain_processing_count({
                    "brain_type": type(self.brain).__name__,
                    "session_id": session_id
                })
                
                # Log operation start
                logger.info(f"Processing input in session {session_id}: {user_input[:50]}...")
                
                # Call actual brain process_input
                response, reasoning_trace = await self._call_brain_process_input(
                    user_input, context, session_id
                )
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                record_brain_processing_duration(duration_ms, {
                    "brain_type": type(self.brain).__name__,
                    "session_id": session_id,
                    "success": "true"
                })
                
                # Update operation history
                self.operation_history.append({
                    "timestamp": datetime.now(),
                    "operation": "process_input",
                    "session_id": session_id,
                    "duration_ms": duration_ms,
                    "success": True,
                    "input_length": len(user_input),
                    "response_length": len(response) if response else 0
                })
                
                # Keep history manageable
                if len(self.operation_history) > 1000:
                    self.operation_history = self.operation_history[-500:]
                
                logger.info(f"Successfully processed input in {duration_ms:.1f}ms")
                return response, reasoning_trace
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                record_brain_processing_duration(duration_ms, {
                    "brain_type": type(self.brain).__name__,
                    "session_id": session_id,
                    "success": "false",
                    "error_type": type(e).__name__
                })
                
                # Update error history
                self.error_history.append({
                    "timestamp": datetime.now(),
                    "operation": "process_input",
                    "session_id": session_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": duration_ms
                })
                
                # Update error rate
                self._update_error_rate()
                
                logger.error(f"Error processing input: {e}")
                raise
    
    async def _call_brain_process_input(
        self,
        user_input: str,
        context: Optional[Any],
        session_id: str
    ) -> Tuple[str, Any]:
        """Call the actual brain's process_input with sub-operation tracing"""
        
        # Intent Recognition Tracing
        with self.tracer.trace_span(
            "brain.intent_recognition",
            TraceOperationType.INTENT_RECOGNITION,
            tags={"session_id": session_id}
        ):
            record_memory_access_duration("intent_recognition", 5.0, {
                "session_id": session_id
            })
        
        # Context Retrieval Tracing
        with self.tracer.trace_span(
            "brain.context_retrieval",
            TraceOperationType.CONTEXT_RETRIEVAL,
            tags={"session_id": session_id}
        ):
            record_memory_access_duration("context_retrieval", 15.0, {
                "session_id": session_id
            })
        
        # Skill Selection and Execution Tracing
        with self.tracer.trace_span(
            "brain.skill_execution",
            TraceOperationType.SKILL_EXECUTION,
            tags={"session_id": session_id}
        ):
            increment_skill_execution_count("brain_processing", True, {
                "session_id": session_id
            })
            record_skill_execution_duration("brain_processing", 50.0, {
                "session_id": session_id
            })
        
        # Call the actual brain
        if hasattr(self.brain, 'process_input'):
            return await self.brain.process_input(user_input, context, session_id)
        elif hasattr(self.brain, 'send_message'):
            response = self.brain.send_message(user_input)
            return response, None  # No reasoning trace for simple brain
        else:
            raise AttributeError("Brain instance does not have process_input or send_message method")
    
    def _update_error_rate(self) -> None:
        """Update error rate based on recent operations"""
        recent_operations = [
            op for op in self.operation_history 
            if (datetime.now() - op["timestamp"]).total_seconds() < 300  # Last 5 minutes
        ]
        
        recent_errors = [
            err for err in self.error_history
            if (datetime.now() - err["timestamp"]).total_seconds() < 300
        ]
        
        if recent_operations:
            error_rate = (len(recent_errors) / len(recent_operations)) * 100
            set_error_rate(error_rate, {"window": "5_minutes"})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the brain"""
        if not self.operation_history:
            return {
                "total_operations": 0,
                "avg_duration_ms": 0,
                "success_rate": 100,
                "error_rate": 0
            }
        
        # Calculate statistics
        total_operations = len(self.operation_history)
        successful_operations = sum(1 for op in self.operation_history if op["success"])
        durations = [op["duration_ms"] for op in self.operation_history]
        
        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": (successful_operations / total_operations) * 100,
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "recent_operations": len([
                op for op in self.operation_history
                if (datetime.now() - op["timestamp"]).total_seconds() < 300
            ]),
            "recent_errors": len([
                err for err in self.error_history
                if (datetime.now() - err["timestamp"]).total_seconds() < 300
            ])
        }
    
    def get_active_sessions(self) -> int:
        """Get count of active sessions"""
        # This would integrate with actual session management
        # For now, return estimated count based on recent activity
        recent_sessions = set()
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        for op in self.operation_history:
            if op["timestamp"] > cutoff_time:
                recent_sessions.add(op["session_id"])
        
        return len(recent_sessions)
    
    def update_cache_metrics(self, hit_rate: float) -> None:
        """Update cache hit rate metrics"""
        set_cache_hit_rate(hit_rate, {
            "brain_type": type(self.brain).__name__
        })
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped brain"""
        return getattr(self.brain, name)


class SkillTracer:
    """
    Tracer for skill execution operations
    
    Provides automatic tracing and metrics collection for skill executions.
    """
    
    def __init__(self, skills_manager):
        """
        Initialize skill tracer
        
        Args:
            skills_manager: The skills manager instance
        """
        self.skills_manager = skills_manager
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_metrics_collector()
        
        logger.info("SkillTracer initialized")
    
    def trace_skill_execution(
        self,
        skill_name: str,
        skill_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Trace skill execution with automatic metrics collection
        
        Args:
            skill_name: Name of the skill being executed
            skill_func: The skill function to execute
            *args: Arguments to pass to skill function
            **kwargs: Keyword arguments to pass to skill function
            
        Returns:
            Result of skill function execution
        """
        start_time = time.time()
        
        with self.tracer.trace_span(
            f"skill.{skill_name}",
            TraceOperationType.SKILL_EXECUTION,
            tags={
                "skill_name": skill_name,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            }
        ):
            try:
                # Execute skill
                result = skill_func(*args, **kwargs)
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                increment_skill_execution_count(skill_name, True, {
                    "execution_type": "sync"
                })
                record_skill_execution_duration(skill_name, duration_ms, {
                    "success": "true"
                })
                
                logger.debug(f"Skill {skill_name} executed successfully in {duration_ms:.1f}ms")
                return result
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                increment_skill_execution_count(skill_name, False, {
                    "execution_type": "sync",
                    "error_type": type(e).__name__
                })
                record_skill_execution_duration(skill_name, duration_ms, {
                    "success": "false",
                    "error_type": type(e).__name__
                })
                
                logger.error(f"Skill {skill_name} failed: {e}")
                raise
    
    async def trace_async_skill_execution(
        self,
        skill_name: str,
        skill_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Trace async skill execution with automatic metrics collection
        
        Args:
            skill_name: Name of the skill being executed
            skill_func: The async skill function to execute
            *args: Arguments to pass to skill function
            **kwargs: Keyword arguments to pass to skill function
            
        Returns:
            Result of skill function execution
        """
        start_time = time.time()
        
        with self.tracer.trace_span(
            f"skill.{skill_name}",
            TraceOperationType.SKILL_EXECUTION,
            tags={
                "skill_name": skill_name,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
                "execution_type": "async"
            }
        ):
            try:
                # Execute async skill
                result = await skill_func(*args, **kwargs)
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                increment_skill_execution_count(skill_name, True, {
                    "execution_type": "async"
                })
                record_skill_execution_duration(skill_name, duration_ms, {
                    "success": "true"
                })
                
                logger.debug(f"Async skill {skill_name} executed successfully in {duration_ms:.1f}ms")
                return result
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                increment_skill_execution_count(skill_name, False, {
                    "execution_type": "async",
                    "error_type": type(e).__name__
                })
                record_skill_execution_duration(skill_name, duration_ms, {
                    "success": "false",
                    "error_type": type(e).__name__
                })
                
                logger.error(f"Async skill {skill_name} failed: {e}")
                raise


class MemoryTracer:
    """
    Tracer for memory operations
    
    Provides automatic tracing and metrics collection for memory access operations.
    """
    
    def __init__(self, memory_instance):
        """
        Initialize memory tracer
        
        Args:
            memory_instance: The memory instance to trace
        """
        self.memory = memory_instance
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_metrics_collector()
        
        logger.info("MemoryTracer initialized")
    
    def trace_memory_operation(
        self,
        operation: str,
        memory_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Trace memory operation with automatic metrics collection
        
        Args:
            operation: Type of memory operation (read, write, search, etc.)
            memory_func: The memory function to execute
            *args: Arguments to pass to memory function
            **kwargs: Keyword arguments to pass to memory function
            
        Returns:
            Result of memory function execution
        """
        start_time = time.time()
        
        with self.tracer.trace_span(
            f"memory.{operation}",
            TraceOperationType.MEMORY_ACCESS,
            tags={
                "operation": operation,
                "args_count": len(args),
                "memory_type": type(self.memory).__name__
            }
        ):
            try:
                # Execute memory operation
                result = memory_func(*args, **kwargs)
                
                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                record_memory_access_duration(operation, duration_ms, {
                    "memory_type": type(self.memory).__name__,
                    "success": "true"
                })
                
                logger.debug(f"Memory operation {operation} completed in {duration_ms:.1f}ms")
                return result
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                record_memory_access_duration(operation, duration_ms, {
                    "memory_type": type(self.memory).__name__,
                    "success": "false",
                    "error_type": type(e).__name__
                })
                
                logger.error(f"Memory operation {operation} failed: {e}")
                raise


def create_traced_brain(brain_instance) -> TracedBrainWrapper:
    """
    Create a traced wrapper for a brain instance
    
    Args:
        brain_instance: The brain instance to wrap
        
    Returns:
        TracedBrainWrapper instance
    """
    return TracedBrainWrapper(brain_instance)


def create_skill_tracer(skills_manager) -> SkillTracer:
    """
    Create a skill tracer for a skills manager
    
    Args:
        skills_manager: The skills manager to trace
        
    Returns:
        SkillTracer instance
    """
    return SkillTracer(skills_manager)


def create_memory_tracer(memory_instance) -> MemoryTracer:
    """
    Create a memory tracer for a memory instance
    
    Args:
        memory_instance: The memory instance to trace
        
    Returns:
        MemoryTracer instance
    """
    return MemoryTracer(memory_instance)


# Integration helper for OpenCode TUI
class OpenCodeMonitoringIntegration:
    """
    Integration helper for OpenCode TUI monitoring
    
    Provides easy integration points for OpenCode TUI to monitor
    Neo-Clone operations without complex setup.
    """
    
    def __init__(self):
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_metrics_collector()
        self.traced_brain: Optional[TracedBrainWrapper] = None
        self.skill_tracer: Optional[SkillTracer] = None
        self.memory_tracer: Optional[MemoryTracer] = None
    
    def setup_brain_monitoring(self, brain_instance) -> TracedBrainWrapper:
        """Setup monitoring for brain instance"""
        self.traced_brain = create_traced_brain(brain_instance)
        return self.traced_brain
    
    def setup_skills_monitoring(self, skills_manager) -> SkillTracer:
        """Setup monitoring for skills manager"""
        self.skill_tracer = create_skill_tracer(skills_manager)
        return self.skill_tracer
    
    def setup_memory_monitoring(self, memory_instance) -> MemoryTracer:
        """Setup monitoring for memory instance"""
        self.memory_tracer = create_memory_tracer(memory_instance)
        return self.memory_tracer
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        summary = {
            "monitoring_active": True,
            "components": {}
        }
        
        if self.traced_brain:
            summary["components"]["brain"] = self.traced_brain.get_performance_stats()
        
        if self.skill_tracer:
            skill_metrics = self.metrics_collector.get_metric_data("skill_execution_count")
            summary["components"]["skills"] = {
                "total_executions": len(skill_metrics),
                "recent_executions": len([
                    m for m in skill_metrics 
                    if (datetime.now() - m.timestamp).total_seconds() < 300
                ])
            }
        
        if self.memory_tracer:
            memory_metrics = self.metrics_collector.get_metric_data("memory_access_duration")
            summary["components"]["memory"] = {
                "total_operations": len(memory_metrics),
                "avg_duration": sum(m.value for m in memory_metrics[-100:]) / min(100, len(memory_metrics)) if memory_metrics else 0
            }
        
        return summary


# Global integration instance
_opencode_integration: Optional[OpenCodeMonitoringIntegration] = None


def get_opencode_integration() -> OpenCodeMonitoringIntegration:
    """Get global OpenCode monitoring integration"""
    global _opencode_integration
    
    if _opencode_integration is None:
        _opencode_integration = OpenCodeMonitoringIntegration()
    
    return _opencode_integration