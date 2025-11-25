"""
Parallel Execution System for MCP Tools

This module provides advanced parallel execution capabilities for MCP tools
including concurrent execution, pipeline processing, and resource management.

Author: Neo-Clone Enhanced
Version: 1.0.0
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for parallel processing"""
    CONCURRENT = "concurrent"           # Execute all tools concurrently
    SEQUENTIAL = "sequential"           # Execute tools one by one
    PIPELINE = "pipeline"               # Execute tools in pipeline
    BATCH = "batch"                     # Execute tools in batches
    ADAPTIVE = "adaptive"               # Adaptive execution based on resources


class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ToolTask:
    """Individual tool execution task"""
    task_id: str
    tool_id: str
    parameters: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class TaskResult:
    """Result of tool task execution"""
    task_id: str
    tool_id: str
    status: ExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful"""
        return self.status == ExecutionStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'tool_id': self.tool_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metadata': self.metadata
        }


@dataclass
class ExecutionPlan:
    """Execution plan for multiple tasks"""
    plan_id: str
    tasks: List[ToolTask]
    mode: ExecutionMode
    max_concurrent: int = 5
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = str(uuid.uuid4())


class ResourcePool:
    """Resource pool for managing execution resources"""
    
    def __init__(self, max_workers: int = 10, max_memory_mb: int = 1024):
        self.max_workers = max_workers
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self.semaphore = asyncio.Semaphore(max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.memory_used = 0
        self.memory_lock = threading.Lock()
        self.active_tasks = set()
    
    async def acquire(self, task_id: str, memory_mb: int = 0) -> bool:
        """Acquire resources for task"""
        await self.semaphore.acquire()
        
        with self.memory_lock:
            memory_bytes = memory_mb * 1024 * 1024
            if self.memory_used + memory_bytes > self.max_memory_bytes:
                self.semaphore.release()
                return False
            
            self.memory_used += memory_bytes
            self.active_tasks.add(task_id)
            return True
    
    def release(self, task_id: str, memory_mb: int = 0) -> None:
        """Release resources from task"""
        with self.memory_lock:
            memory_bytes = memory_mb * 1024 * 1024
            self.memory_used = max(0, self.memory_used - memory_bytes)
            self.active_tasks.discard(task_id)
        
        self.semaphore.release()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        return {
            'max_workers': self.max_workers,
            'active_tasks': len(self.active_tasks),
            'available_workers': self.max_workers - len(self.active_tasks),
            'memory_used_mb': self.memory_used / (1024 * 1024),
            'memory_available_mb': (self.max_memory_bytes - self.memory_used) / (1024 * 1024),
            'memory_usage_percent': (self.memory_used / self.max_memory_bytes) * 100
        }
    
    def shutdown(self) -> None:
        """Shutdown resource pool"""
        self.thread_pool.shutdown(wait=True)


class ParallelExecutor:
    """Advanced parallel execution system"""
    
    def __init__(self, max_workers: int = 10, max_memory_mb: int = 1024):
        self.resource_pool = ResourcePool(max_workers, max_memory_mb)
        self.execution_history: List[TaskResult] = []
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.completed_plans: Dict[str, List[TaskResult]] = {}
        
        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
    
    async def execute_concurrent(self, tasks: List[ToolTask], 
                              max_concurrent: Optional[int] = None) -> List[TaskResult]:
        """Execute tasks concurrently"""
        if max_concurrent is None:
            max_concurrent = self.resource_pool.max_workers
        
        logger.info(f"Executing {len(tasks)} tasks concurrently (max: {max_concurrent})")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_task(task: ToolTask) -> TaskResult:
            async with semaphore:
                return await self._execute_task(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_single_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        task_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_results.append(TaskResult(
                    task_id=tasks[i].task_id,
                    tool_id=tasks[i].tool_id,
                    status=ExecutionStatus.FAILED,
                    error=str(result)
                ))
            else:
                task_results.append(result)
        
        return task_results
    
    async def execute_sequential(self, tasks: List[ToolTask]) -> List[TaskResult]:
        """Execute tasks sequentially"""
        logger.info(f"Executing {len(tasks)} tasks sequentially")
        
        results = []
        for task in tasks:
            result = await self._execute_task(task)
            results.append(result)
            
            # Stop on first failure if configured
            if result.status == ExecutionStatus.FAILED and task.metadata.get('stop_on_failure', False):
                logger.warning(f"Stopping sequential execution due to failure in task {task.task_id}")
                break
        
        return results
    
    async def execute_pipeline(self, tasks: List[ToolTask]) -> List[TaskResult]:
        """Execute tasks in pipeline (output of one becomes input to next)"""
        logger.info(f"Executing {len(tasks)} tasks in pipeline")
        
        results = []
        pipeline_data = None
        
        for i, task in enumerate(tasks):
            # Add pipeline data to task parameters
            if pipeline_data is not None and i > 0:
                task.parameters['pipeline_input'] = pipeline_data
                task.metadata['pipeline_stage'] = i
            
            result = await self._execute_task(task)
            results.append(result)
            
            if result.status == ExecutionStatus.COMPLETED:
                pipeline_data = result.result
            else:
                logger.warning(f"Pipeline failed at stage {i} with task {task.task_id}")
                break
        
        return results
    
    async def execute_batch(self, tasks: List[ToolTask], 
                          batch_size: int = 5) -> List[TaskResult]:
        """Execute tasks in batches"""
        logger.info(f"Executing {len(tasks)} tasks in batches of {batch_size}")
        
        all_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            logger.debug(f"Executing batch {i//batch_size + 1} with {len(batch)} tasks")
            
            batch_results = await self.execute_concurrent(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    async def execute_adaptive(self, tasks: List[ToolTask]) -> List[TaskResult]:
        """Execute tasks with adaptive strategy"""
        logger.info(f"Executing {len(tasks)} tasks with adaptive strategy")
        
        # Analyze tasks to determine best strategy
        resource_stats = self.resource_pool.get_usage_stats()
        
        # Choose strategy based on current load and task characteristics
        if resource_stats['memory_usage_percent'] > 80:
            # High memory usage - use sequential
            logger.info("High memory usage detected, using sequential execution")
            return await self.execute_sequential(tasks)
        elif resource_stats['available_workers'] < len(tasks):
            # Limited workers - use batching
            batch_size = max(1, resource_stats['available_workers'])
            logger.info(f"Limited workers, using batch execution with size {batch_size}")
            return await self.execute_batch(tasks, batch_size)
        else:
            # Good resources - use concurrent
            logger.info("Good resource availability, using concurrent execution")
            return await self.execute_concurrent(tasks)
    
    async def execute_plan(self, plan: ExecutionPlan) -> List[TaskResult]:
        """Execute an execution plan"""
        logger.info(f"Executing plan {plan.plan_id} with {len(plan.tasks)} tasks")
        
        self.active_plans[plan.plan_id] = plan
        start_time = time.time()
        
        try:
            # Choose execution strategy
            if plan.mode == ExecutionMode.CONCURRENT:
                results = await self.execute_concurrent(plan.tasks, plan.max_concurrent)
            elif plan.mode == ExecutionMode.SEQUENTIAL:
                results = await self.execute_sequential(plan.tasks)
            elif plan.mode == ExecutionMode.PIPELINE:
                results = await self.execute_pipeline(plan.tasks)
            elif plan.mode == ExecutionMode.BATCH:
                results = await self.execute_batch(plan.tasks, plan.max_concurrent)
            elif plan.mode == ExecutionMode.ADAPTIVE:
                results = await self.execute_adaptive(plan.tasks)
            else:
                raise ValueError(f"Unsupported execution mode: {plan.mode}")
            
            # Handle retries if needed
            if plan.retry_count > 0:
                failed_tasks = [
                    plan.tasks[i] for i, result in enumerate(results)
                    if result.status == ExecutionStatus.FAILED
                ]
                
                if failed_tasks:
                    logger.info(f"Retrying {len(failed_tasks)} failed tasks")
                    await asyncio.sleep(plan.retry_delay)
                    
                    retry_results = await self.execute_plan(ExecutionPlan(
                        plan_id=f"{plan.plan_id}_retry",
                        tasks=failed_tasks,
                        mode=plan.mode,
                        max_concurrent=plan.max_concurrent,
                        timeout=plan.timeout,
                        retry_count=plan.retry_count - 1,
                        retry_delay=plan.retry_delay
                    ))
                    
                    # Merge retry results
                    for i, result in enumerate(results):
                        if result.status == ExecutionStatus.FAILED:
                            results[i] = retry_results[failed_tasks.index(plan.tasks[i])]
            
            execution_time = time.time() - start_time
            logger.info(f"Plan {plan.plan_id} completed in {execution_time:.2f}s")
            
            # Store results
            self.completed_plans[plan.plan_id] = results
            del self.active_plans[plan.plan_id]
            
            return results
            
        except Exception as e:
            logger.error(f"Plan {plan.plan_id} failed: {e}")
            if plan.plan_id in self.active_plans:
                del self.active_plans[plan.plan_id]
            raise
    
    async def _execute_task(self, task: ToolTask) -> TaskResult:
        """Execute a single task"""
        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task.task_id,
            tool_id=task.tool_id,
            status=ExecutionStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Acquire resources
            memory_mb = task.metadata.get('memory_mb', 0)
            if not await self.resource_pool.acquire(task.task_id, memory_mb):
                raise Exception(f"Failed to acquire resources for task {task.task_id}")
            
            # Execute task with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_tool_task(task),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_tool_task(task)
            
            # Update task result
            task_result.result = result
            task_result.status = ExecutionStatus.COMPLETED
            
            # Update statistics
            self.total_executions += 1
            self.successful_executions += 1
            
        except asyncio.TimeoutError:
            task_result.status = ExecutionStatus.TIMEOUT
            task_result.error = f"Task timed out after {task.timeout}s"
            self.total_executions += 1
            self.failed_executions += 1
            
        except Exception as e:
            task_result.status = ExecutionStatus.FAILED
            task_result.error = str(e)
            self.total_executions += 1
            self.failed_executions += 1
        
        finally:
            # Release resources
            memory_mb = task.metadata.get('memory_mb', 0)
            self.resource_pool.release(task.task_id, memory_mb)
            
            # Calculate execution time
            end_time = datetime.now()
            task_result.end_time = end_time
            task_result.execution_time = (end_time - start_time).total_seconds()
            
            # Update total execution time
            self.total_execution_time += task_result.execution_time
            
            # Add to history
            self.execution_history.append(task_result)
            
            # Keep history manageable
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
            
            # Call callback if provided
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(task_result)
                    else:
                        task.callback(task_result)
                except Exception as e:
                    logger.error(f"Task callback failed: {e}")
        
        return task_result
    
    async def _execute_tool_task(self, task: ToolTask) -> Any:
        """Execute the actual tool task"""
        # This would integrate with the MCP client or tool execution system
        # For now, simulate execution
        await asyncio.sleep(0.1)  # Simulate work
        
        # Return mock result
        return {
            'tool_id': task.tool_id,
            'parameters': task.parameters,
            'execution_time': time.time(),
            'task_id': task.task_id
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': (self.successful_executions / self.total_executions * 100) if self.total_executions > 0 else 0,
            'average_execution_time': self.total_execution_time / self.total_executions if self.total_executions > 0 else 0,
            'active_plans': len(self.active_plans),
            'completed_plans': len(self.completed_plans),
            'resource_usage': self.resource_pool.get_usage_stats()
        }
    
    def get_task_history(self, limit: int = 100) -> List[TaskResult]:
        """Get recent task execution history"""
        return self.execution_history[-limit:] if limit > 0 else self.execution_history
    
    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an active execution plan"""
        if plan_id in self.active_plans:
            # In a real implementation, this would cancel running tasks
            del self.active_plans[plan_id]
            logger.info(f"Cancelled plan {plan_id}")
            return True
        return False
    
    def shutdown(self) -> None:
        """Shutdown the parallel executor"""
        logger.info("Shutting down parallel executor")
        self.resource_pool.shutdown()


# Global parallel executor instance
parallel_executor = ParallelExecutor()