#!/usr/bin/env python3
"""
OpenCode Performance Monitor
============================

Performance monitoring specifically for OpenCode TUI integration.
"""

import logging
import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from .core_performance_monitor import CorePerformanceMonitor, get_performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    timestamp: datetime


class OpenCodePerformanceMonitor:
    """Performance monitor for OpenCode TUI operations"""
    
    def __init__(self):
        self.core_monitor = get_performance_monitor()
        self.start_time = datetime.now()
        self.operation_counts = {}
        self.lock = threading.RLock()
        
    def track_operation(self, operation: str, duration: float, 
                        success: bool = True, metadata: Dict[str, Any] = None):
        """Track an OpenCode operation"""
        with self.lock:
            # Update operation counts
            key = f"{operation}_{'success' if success else 'failure'}"
            self.operation_counts[key] = self.operation_counts.get(key, 0) + 1
            
            # Record to core monitor
            self.core_monitor.record_execution_time(
                operation=operation,
                execution_time=duration,
                metadata=metadata or {"success": success}
            )
    
    def track_file_operation(self, operation: str, file_path: str, 
                            duration: float, success: bool = True):
        """Track file system operations"""
        metadata = {
            "file_path": file_path,
            "operation": operation,
            "success": success
        }
        self.track_operation(f"file_{operation}", duration, success, metadata)
    
    def track_model_operation(self, model_name: str, operation: str,
                            duration: float, tokens: int = None, 
                            success: bool = True):
        """Track model-related operations"""
        metadata = {
            "model": model_name,
            "operation": operation,
            "success": success
        }
        if tokens:
            metadata["tokens"] = tokens
            
        self.track_operation(f"model_{operation}", duration, success, metadata)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            memory_mb=psutil.virtual_memory().used / 1024 / 1024,
            disk_usage_percent=psutil.disk_usage('.').percent,
            timestamp=datetime.now()
        )
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """Get summary of all operations"""
        with self.lock:
            total_ops = sum(self.operation_counts.values())
            success_ops = sum(v for k, v in self.operation_counts.items() if 'success' in k)
            failure_ops = sum(v for k, v in self.operation_counts.items() if 'failure' in k)
            
            return {
                "total_operations": total_ops,
                "successful_operations": success_ops,
                "failed_operations": failure_ops,
                "success_rate": (success_ops / total_ops * 100) if total_ops > 0 else 0,
                "operation_breakdown": dict(self.operation_counts),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        system_metrics = self.get_system_metrics()
        operation_summary = self.get_operation_summary()
        core_summary = self.core_monitor.get_performance_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_mb": system_metrics.memory_mb,
                "disk_usage_percent": system_metrics.disk_usage_percent
            },
            "operations": operation_summary,
            "core_performance": core_summary
        }


# Global instance
_opencode_monitor = None


def get_opencode_monitor() -> OpenCodePerformanceMonitor:
    """Get the global OpenCode performance monitor"""
    global _opencode_monitor
    if _opencode_monitor is None:
        _opencode_monitor = OpenCodePerformanceMonitor()
    return _opencode_monitor


def track_operation(operation: str, duration: float, 
                   success: bool = True, metadata: Dict[str, Any] = None):
    """Convenience function to track operations"""
    monitor = get_opencode_monitor()
    monitor.track_operation(operation, duration, success, metadata)


def track_file_operation(operation: str, file_path: str, 
                        duration: float, success: bool = True):
    """Convenience function to track file operations"""
    monitor = get_opencode_monitor()
    monitor.track_file_operation(operation, file_path, duration, success)


if __name__ == "__main__":
    # Test the OpenCode performance monitor
    monitor = OpenCodePerformanceMonitor()
    
    print("Testing OpenCode Performance Monitor")
    print("=" * 40)
    
    # Track some test operations
    monitor.track_operation("test_op", 0.1, True)
    monitor.track_file_operation("read", "test.txt", 0.05, True)
    monitor.track_model_operation("gpt-4", "completion", 1.2, 150, True)
    
    # Get reports
    system_metrics = monitor.get_system_metrics()
    operation_summary = monitor.get_operation_summary()
    performance_report = monitor.get_performance_report()
    
    print(f"System CPU: {system_metrics.cpu_percent}%")
    print(f"System Memory: {system_metrics.memory_percent}%")
    print(f"Total Operations: {operation_summary['total_operations']}")
    print(f"Success Rate: {operation_summary['success_rate']:.1f}%")
    
    print("\nOpenCode performance monitor working correctly!")