"""
Resource Management System for MCP Tools

This module provides comprehensive resource management including CPU, memory,
network, and disk usage monitoring and allocation for MCP tool execution.

Author: Neo-Clone Enhanced
Version: 1.0.0
"""

import asyncio
import psutil
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import gc
import weakref

# Configure logging
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    TEMPORARY_STORAGE = "temporary_storage"


class AllocationStatus(Enum):
    """Resource allocation status"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    EXHAUSTED = "exhausted"
    THROTTLED = "throttled"


@dataclass
class ResourceLimit:
    """Resource limit definition"""
    resource_type: ResourceType
    max_value: float
    current_usage: float = 0.0
    allocated: float = 0.0
    reserved: float = 0.0
    unit: str = ""
    
    @property
    def available(self) -> float:
        """Get available resource amount"""
        return max(0, self.max_value - self.allocated - self.reserved)
    
    @property
    def usage_percent(self) -> float:
        """Get usage percentage"""
        if self.max_value == 0:
            return 0.0
        return (self.current_usage / self.max_value) * 100
    
    @property
    def allocation_percent(self) -> float:
        """Get allocation percentage"""
        if self.max_value == 0:
            return 0.0
        return ((self.allocated + self.reserved) / self.max_value) * 100
    
    def can_allocate(self, amount: float) -> bool:
        """Check if resource can be allocated"""
        return self.available >= amount
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource amount"""
        if self.can_allocate(amount):
            self.allocated += amount
            return True
        return False
    
    def release(self, amount: float) -> None:
        """Release allocated resource"""
        self.allocated = max(0, self.allocated - amount)
    
    def reserve(self, amount: float) -> bool:
        """Reserve resource amount"""
        if self.can_allocate(amount):
            self.reserved += amount
            return True
        return False
    
    def unreserve(self, amount: float) -> None:
        """Unreserve resource amount"""
        self.reserved = max(0, self.reserved - amount)


@dataclass
class ResourceAllocation:
    """Resource allocation record"""
    allocation_id: str
    resource_type: ResourceType
    amount: float
    task_id: Optional[str] = None
    tool_id: Optional[str] = None
    allocated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if allocation is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'allocation_id': self.allocation_id,
            'resource_type': self.resource_type.value,
            'amount': self.amount,
            'task_id': self.task_id,
            'tool_id': self.tool_id,
            'allocated_at': self.allocated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_task = None
        
        # Current system metrics
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.disk_usage = {}
        self.network_io = {}
        self.process_count = 0
        
        # Historical data
        self.history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    async def start_monitoring(self) -> None:
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_metrics(self) -> None:
        """Update system metrics"""
        try:
            # CPU metrics
            self.cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            
            # Disk metrics
            self.disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_usage[partition.mountpoint] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    }
                except Exception:
                    continue
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Process count
            self.process_count = len(psutil.pids())
            
            # Add to history
            with self.lock:
                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': self.cpu_percent,
                    'memory_percent': self.memory_percent,
                    'disk_usage': self.disk_usage.copy(),
                    'network_io': self.network_io.copy(),
                    'process_count': self.process_count
                }
                self.history.append(snapshot)
                
                # Keep history manageable
                if len(self.history) > self.max_history_size:
                    self.history = self.history[-self.max_history_size:]
        
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self.lock:
            return {
                'cpu_percent': self.cpu_percent,
                'memory_percent': self.memory_percent,
                'disk_usage': self.disk_usage.copy(),
                'network_io': self.network_io.copy(),
                'process_count': self.process_count,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_historical_metrics(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical metrics for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            return [
                snapshot for snapshot in self.history
                if datetime.fromisoformat(snapshot['timestamp']) >= cutoff_time
            ]


class ResourceManager:
    """Comprehensive resource management system"""
    
    def __init__(self, max_memory_mb: int = 2048, max_cpu_percent: float = 80.0):
        # Resource limits
        self.resource_limits: Dict[ResourceType, ResourceLimit] = {
            ResourceType.CPU: ResourceLimit(
                resource_type=ResourceType.CPU,
                max_value=max_cpu_percent,
                unit="percent"
            ),
            ResourceType.MEMORY: ResourceLimit(
                resource_type=ResourceType.MEMORY,
                max_value=max_memory_mb,
                unit="MB"
            ),
            ResourceType.DISK: ResourceLimit(
                resource_type=ResourceType.DISK,
                max_value=10240,  # 10GB
                unit="MB"
            ),
            ResourceType.NETWORK: ResourceLimit(
                resource_type=ResourceType.NETWORK,
                max_value=100,  # 100 connections
                unit="connections"
            ),
            ResourceType.TEMPORARY_STORAGE: ResourceLimit(
                resource_type=ResourceType.TEMPORARY_STORAGE,
                max_value=1024,  # 1GB
                unit="MB"
            )
        }
        
        # Active allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.task_allocations: Dict[str, List[str]] = {}
        
        # System monitor
        self.system_monitor = SystemMonitor()
        
        # Resource usage tracking
        self.usage_history: List[Dict[str, Any]] = []
        self.allocation_history: List[ResourceAllocation] = []
        
        # Cleanup task
        self.cleanup_task = None
        self.running = False
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    async def start(self) -> None:
        """Start resource manager"""
        self.running = True
        await self.system_monitor.start_monitoring()
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Resource manager started")
    
    async def stop(self) -> None:
        """Stop resource manager"""
        self.running = False
        await self.system_monitor.stop_monitoring()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Release all allocations
        await self.release_all_allocations()
        logger.info("Resource manager stopped")
    
    def allocate_resources(self, task_id: str, tool_id: str, 
                         resource_requests: Dict[ResourceType, float],
                         timeout_seconds: Optional[float] = None) -> Optional[str]:
        """Allocate resources for a task"""
        with self.lock:
            allocation_id = f"alloc_{int(time.time() * 1000)}_{task_id}"
            
            # Check if all resources can be allocated
            for resource_type, amount in resource_requests.items():
                if resource_type not in self.resource_limits:
                    logger.warning(f"Unknown resource type: {resource_type}")
                    continue
                
                limit = self.resource_limits[resource_type]
                if not limit.can_allocate(amount):
                    logger.warning(f"Cannot allocate {amount} {limit.unit} of {resource_type.value}")
                    return None
            
            # Allocate resources
            allocations = []
            for resource_type, amount in resource_requests.items():
                if resource_type in self.resource_limits:
                    limit = self.resource_limits[resource_type]
                    if limit.allocate(amount):
                        allocation = ResourceAllocation(
                            allocation_id=f"{allocation_id}_{resource_type.value}",
                            resource_type=resource_type,
                            amount=amount,
                            task_id=task_id,
                            tool_id=tool_id,
                            expires_at=datetime.now() + timedelta(seconds=timeout_seconds) if timeout_seconds else None
                        )
                        allocations.append(allocation)
                        self.allocations[allocation.allocation_id] = allocation
                    else:
                        # Rollback allocations
                        for alloc in allocations:
                            self.resource_limits[alloc.resource_type].release(alloc.amount)
                            del self.allocations[alloc.allocation_id]
                        return None
            
            # Track task allocations
            if task_id not in self.task_allocations:
                self.task_allocations[task_id] = []
            self.task_allocations[task_id].extend([alloc.allocation_id for alloc in allocations])
            
            # Add to history
            self.allocation_history.extend(allocations)
            
            logger.info(f"Allocated resources for task {task_id}: {resource_requests}")
            return allocation_id
    
    def release_resources(self, allocation_id: str) -> bool:
        """Release specific resource allocation"""
        with self.lock:
            if allocation_id not in self.allocations:
                return False
            
            allocation = self.allocations[allocation_id]
            
            # Release from resource limit
            if allocation.resource_type in self.resource_limits:
                self.resource_limits[allocation.resource_type].release(allocation.amount)
            
            # Remove from allocations
            del self.allocations[allocation_id]
            
            # Remove from task allocations
            if allocation.task_id and allocation.task_id in self.task_allocations:
                if allocation_id in self.task_allocations[allocation.task_id]:
                    self.task_allocations[allocation.task_id].remove(allocation_id)
                
                if not self.task_allocations[allocation.task_id]:
                    del self.task_allocations[allocation.task_id]
            
            logger.debug(f"Released allocation {allocation_id}")
            return True
    
    async def release_task_resources(self, task_id: str) -> int:
        """Release all resources for a task"""
        with self.lock:
            if task_id not in self.task_allocations:
                return 0
            
            allocation_ids = self.task_allocations[task_id].copy()
            released_count = 0
            
            for allocation_id in allocation_ids:
                if self.release_resources(allocation_id):
                    released_count += 1
            
            logger.info(f"Released {released_count} allocations for task {task_id}")
            return released_count
    
    async def release_all_allocations(self) -> int:
        """Release all active allocations"""
        with self.lock:
            allocation_ids = list(self.allocations.keys())
            released_count = 0
            
            for allocation_id in allocation_ids:
                if self.release_resources(allocation_id):
                    released_count += 1
            
            logger.info(f"Released {released_count} total allocations")
            return released_count
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        with self.lock:
            status = {
                'resource_limits': {},
                'active_allocations': len(self.allocations),
                'active_tasks': len(self.task_allocations),
                'system_metrics': self.system_monitor.get_current_metrics()
            }
            
            for resource_type, limit in self.resource_limits.items():
                status['resource_limits'][resource_type.value] = {
                    'max_value': limit.max_value,
                    'current_usage': limit.current_usage,
                    'allocated': limit.allocated,
                    'reserved': limit.reserved,
                    'available': limit.available,
                    'usage_percent': limit.usage_percent,
                    'allocation_percent': limit.allocation_percent,
                    'unit': limit.unit
                }
            
            return status
    
    def get_task_allocations(self, task_id: str) -> List[ResourceAllocation]:
        """Get allocations for a specific task"""
        with self.lock:
            if task_id not in self.task_allocations:
                return []
            
            allocation_ids = self.task_allocations[task_id]
            return [
                self.allocations[alloc_id] 
                for alloc_id in allocation_ids 
                if alloc_id in self.allocations
            ]
    
    def get_allocation_history(self, limit: int = 100) -> List[ResourceAllocation]:
        """Get recent allocation history"""
        with self.lock:
            return self.allocation_history[-limit:] if limit > 0 else self.allocation_history
    
    def check_resource_availability(self, resource_requests: Dict[ResourceType, float]) -> bool:
        """Check if requested resources are available"""
        with self.lock:
            for resource_type, amount in resource_requests.items():
                if resource_type not in self.resource_limits:
                    continue
                
                limit = self.resource_limits[resource_type]
                if not limit.can_allocate(amount):
                    return False
            
            return True
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get resource optimization suggestions"""
        suggestions = []
        status = self.get_resource_status()
        
        # CPU suggestions
        cpu_status = status['resource_limits'].get('cpu', {})
        if cpu_status.get('allocation_percent', 0) > 80:
            suggestions.append("High CPU allocation detected. Consider reducing concurrent tasks.")
        
        # Memory suggestions
        memory_status = status['resource_limits'].get('memory', {})
        if memory_status.get('allocation_percent', 0) > 85:
            suggestions.append("High memory allocation detected. Consider implementing memory cleanup.")
        
        # System metrics suggestions
        system_metrics = status.get('system_metrics', {})
        if system_metrics.get('memory_percent', 0) > 90:
            suggestions.append("System memory usage is high. Consider garbage collection.")
        
        if system_metrics.get('cpu_percent', 0) > 90:
            suggestions.append("System CPU usage is high. Consider reducing task complexity.")
        
        # Allocation suggestions
        if status['active_allocations'] > 50:
            suggestions.append("Many active allocations. Consider implementing allocation pooling.")
        
        return suggestions
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.running:
            try:
                await self._cleanup_expired_allocations()
                await self._update_system_usage()
                await asyncio.sleep(30)  # Cleanup every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource cleanup error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_allocations(self) -> None:
        """Clean up expired allocations"""
        with self.lock:
            expired_allocations = [
                alloc_id for alloc_id, allocation in self.allocations.items()
                if allocation.is_expired
            ]
            
            for alloc_id in expired_allocations:
                self.release_resources(alloc_id)
                logger.debug(f"Cleaned up expired allocation {alloc_id}")
            
            if expired_allocations:
                logger.info(f"Cleaned up {len(expired_allocations)} expired allocations")
    
    async def _update_system_usage(self) -> None:
        """Update system usage in resource limits"""
        system_metrics = self.system_monitor.get_current_metrics()
        
        # Update memory usage
        if 'memory' in self.resource_limits:
            memory_limit = self.resource_limits[ResourceType.MEMORY]
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            used_memory_mb = (system_metrics.get('memory_percent', 0) / 100) * total_memory_mb
            memory_limit.current_usage = used_memory_mb
        
        # Update CPU usage
        if 'cpu' in self.resource_limits:
            cpu_limit = self.resource_limits[ResourceType.CPU]
            cpu_limit.current_usage = system_metrics.get('cpu_percent', 0)
        
        # Add to usage history
        usage_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'resource_usage': {
                resource_type.value: {
                    'current_usage': limit.current_usage,
                    'allocated': limit.allocated,
                    'reserved': limit.reserved,
                    'available': limit.available
                }
                for resource_type, limit in self.resource_limits.items()
            },
            'system_metrics': system_metrics
        }
        
        self.usage_history.append(usage_snapshot)
        
        # Keep history manageable
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-500:]
    
    def export_resource_report(self) -> Dict[str, Any]:
        """Export comprehensive resource report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'resource_status': self.get_resource_status(),
            'allocation_history': [alloc.to_dict() for alloc in self.get_allocation_history(50)],
            'optimization_suggestions': self.get_optimization_suggestions(),
            'usage_summary': {
                'total_allocations': len(self.allocation_history),
                'active_allocations': len(self.allocations),
                'active_tasks': len(self.task_allocations),
                'cleanup_cycles': len(self.usage_history)
            }
        }


# Global resource manager instance
resource_manager = ResourceManager()