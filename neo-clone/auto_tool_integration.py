#!/usr/bin/env python3
"""
Auto Tool Integration for Neo-Clone + OpenCode TUI
Seamlessly integrates performance tools without manual monitoring overhead
"""

import os
import sys
import time
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List

class AutoToolIntegration:
    """Automatic tool integration for Neo-Clone"""
    
    def __init__(self):
        self.tools = {}
        self.available_tools = {}
        self.auto_enabled = True
        self.performance_mode = "adaptive"  # adaptive, performance, compatibility
        
    def discover_tools(self) -> Dict[str, Any]:
        """Discover available tools without requiring installation"""
        tools = {}
        
        # Core Python tools (always available)
        tools["time"] = time
        tools["os"] = os
        tools["sys"] = sys
        tools["pathlib"] = Path
        
        # Try optional tools with graceful fallback
        optional_tools = {
            "psutil": "psutil",
            "statistics": "statistics", 
            "asyncio_throttle": "asyncio_throttle",
            "threading": "threading",
            "concurrent": "concurrent.futures",
            "multiprocessing": "multiprocessing",
            "json": "json",
            "csv": "csv",
            "re": "re"
        }
        
        for tool_name, module_name in optional_tools.items():
            try:
                module = importlib.import_module(module_name)
                tools[tool_name] = module
                self.available_tools[tool_name] = True
            except ImportError:
                self.available_tools[tool_name] = False
        
        return tools
    
    def get_performance_tools(self) -> Dict[str, Any]:
        """Get performance monitoring tools"""
        perf_tools = {}
        
        # Basic timing (always available)
        perf_tools["timer"] = self._create_timer()
        perf_tools["profiler"] = self._create_simple_profiler()
        
        # Memory tracking (if available)
        if self.available_tools.get("psutil", False):
            perf_tools["memory"] = self._create_memory_tracker()
            perf_tools["cpu"] = self._create_cpu_tracker()
        
        # Concurrency tools
        perf_tools["thread_pool"] = self._create_thread_pool()
        perf_tools["async_executor"] = self._create_async_executor()
        
        return perf_tools
    
    def _create_timer(self):
        """Create a simple timer"""
        class SimpleTimer:
            def __init__(self):
                self.start_time = None
                
            def start(self):
                self.start_time = time.time()
                return self
                
            def stop(self):
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    self.start_time = None
                    return elapsed
                return 0
                
            def __enter__(self):
                return self.start()
                
            def __exit__(self, *args):
                return self.stop()
        
        return SimpleTimer()
    
    def _create_simple_profiler(self):
        """Create a simple profiler"""
        class SimpleProfiler:
            def __init__(self):
                self.stats = {}
                
            def profile(self, name: str):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        start_time = time.time()
                        try:
                            result = func(*args, **kwargs)
                            success = True
                        except Exception as e:
                            result = e
                            success = False
                        
                        elapsed = time.time() - start_time
                        
                        if name not in self.stats:
                            self.stats[name] = {
                                "calls": 0,
                                "total_time": 0,
                                "avg_time": 0,
                                "success_rate": 0,
                                "errors": 0
                            }
                        
                        self.stats[name]["calls"] += 1
                        self.stats[name]["total_time"] += elapsed
                        self.stats[name]["avg_time"] = self.stats[name]["total_time"] / self.stats[name]["calls"]
                        
                        if success:
                            self.stats[name]["success_rate"] = (self.stats[name]["success_rate"] * (self.stats[name]["calls"] - 1) + 1) / self.stats[name]["calls"]
                        else:
                            self.stats[name]["errors"] += 1
                            self.stats[name]["success_rate"] = (self.stats[name]["success_rate"] * (self.stats[name]["calls"] - 1)) / self.stats[name]["calls"]
                        
                        if not success:
                            raise result
                        
                        return result
                    return wrapper
                return decorator
            
            def get_stats(self):
                return self.stats.copy()
        
        return SimpleProfiler()
    
    def _create_memory_tracker(self):
        """Create memory tracker if psutil available"""
        if not self.available_tools.get("psutil", False):
            return None
            
        import psutil
        
        class MemoryTracker:
            def __init__(self):
                self.process = psutil.Process()
                
            def get_memory_usage(self):
                try:
                    memory_info = self.process.memory_info()
                    return {
                        "rss": memory_info.rss / 1024 / 1024,  # MB
                        "vms": memory_info.vms / 1024 / 1024,  # MB
                        "percent": self.process.memory_percent()
                    }
                except:
                    return {"rss": 0, "vms": 0, "percent": 0}
        
        return MemoryTracker()
    
    def _create_cpu_tracker(self):
        """Create CPU tracker if psutil available"""
        if not self.available_tools.get("psutil", False):
            return None
            
        import psutil
        
        class CPUTracker:
            def get_cpu_usage(self):
                try:
                    return {
                        "percent": psutil.cpu_percent(interval=0.1),
                        "count": psutil.cpu_count(),
                        "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                    }
                except:
                    return {"percent": 0, "count": 0, "freq": None}
        
        return CPUTracker()
    
    def _create_thread_pool(self):
        """Create thread pool executor"""
        try:
            from concurrent.futures import ThreadPoolExecutor
            return ThreadPoolExecutor(max_workers=4)
        except ImportError:
            return None
    
    def _create_async_executor(self):
        """Create async executor utilities"""
        class AsyncExecutor:
            def __init__(self):
                self.loop = None
                
            def get_loop(self):
                try:
                    import asyncio
                    return asyncio.get_event_loop()
                except:
                    return None
        
        return AsyncExecutor()
    
    def integrate_with_neo_clone(self, neo_clone_instance=None):
        """Integrate tools with Neo-Clone instance"""
        if not neo_clone_instance:
            return self.get_toolkit()
        
        # Add tools to Neo-Clone instance
        toolkit = self.get_toolkit()
        
        # Add performance tools as methods
        if hasattr(neo_clone_instance, 'add_tool'):
            neo_clone_instance.add_tool("performance", toolkit["performance"])
            neo_clone_instance.add_tool("utilities", toolkit["utilities"])
        
        # Add auto-profiling to key methods
        self._add_auto_profiling(neo_clone_instance)
        
        return toolkit
    
    def _add_auto_profiling(self, neo_clone_instance):
        """Add automatic profiling to key Neo-Clone methods"""
        profiler = self.get_performance_tools()["profiler"]
        
        # Profile key methods if they exist
        key_methods = ["process_request", "execute_skill", "reason", "analyze"]
        
        for method_name in key_methods:
            if hasattr(neo_clone_instance, method_name):
                original_method = getattr(neo_clone_instance, method_name)
                profiled_method = profiler.profile(f"neo_clone.{method_name}")(original_method)
                setattr(neo_clone_instance, method_name, profiled_method)
    
    def get_toolkit(self) -> Dict[str, Any]:
        """Get complete toolkit for Neo-Clone"""
        return {
            "performance": self.get_performance_tools(),
            "utilities": self.discover_tools(),
            "available": self.available_tools,
            "auto_mode": self.auto_enabled
        }
    
    def auto_optimize(self, operation_type: str = "general") -> Dict[str, Any]:
        """Auto-optimize based on operation type"""
        optimizations = {
            "general": {
                "use_threading": True,
                "batch_size": 10,
                "timeout": 30,
                "retry_attempts": 3
            },
            "io_heavy": {
                "use_threading": True,
                "batch_size": 50,
                "timeout": 60,
                "retry_attempts": 5
            },
            "cpu_heavy": {
                "use_threading": False,
                "batch_size": 5,
                "timeout": 120,
                "retry_attempts": 2
            },
            "memory_heavy": {
                "use_threading": True,
                "batch_size": 3,
                "timeout": 180,
                "retry_attempts": 1
            }
        }
        
        return optimizations.get(operation_type, optimizations["general"])

# Global auto integration instance
_auto_integration = None

def get_auto_integration() -> AutoToolIntegration:
    """Get global auto integration instance"""
    global _auto_integration
    if _auto_integration is None:
        _auto_integration = AutoToolIntegration()
    return _auto_integration

def auto_integrate_neo_clone(neo_clone_instance=None) -> Dict[str, Any]:
    """Auto-integrate tools with Neo-Clone - call this once at startup"""
    integration = get_auto_integration()
    return integration.integrate_with_neo_clone(neo_clone_instance)

def get_performance_tools() -> Dict[str, Any]:
    """Get performance tools for immediate use"""
    integration = get_auto_integration()
    return integration.get_performance_tools()

def auto_optimize_operation(operation_type: str = "general") -> Dict[str, Any]:
    """Get optimization settings for operation"""
    integration = get_auto_integration()
    return integration.auto_optimize(operation_type)

# Decorator for auto-profiling any function
def auto_profile(name: str = None):
    """Decorator to auto-profile any function"""
    def decorator(func):
        integration = get_auto_integration()
        profiler = integration.get_performance_tools()["profiler"]
        profile_name = name or f"{func.__module__}.{func.__name__}"
        return profiler.profile(profile_name)(func)
    return decorator

# Context manager for auto-timing
def auto_timer():
    """Context manager for auto-timing operations"""
    integration = get_auto_integration()
    return integration.get_performance_tools()["timer"]

# Quick setup function
def setup_auto_tools():
    """Setup auto tools - call this once"""
    integration = get_auto_integration()
    toolkit = integration.get_toolkit()
    
    print(f"Auto Tools Ready:")
    print(f"  Performance tools: {len(toolkit['performance'])} available")
    print(f"  Utility modules: {len(toolkit['utilities'])} available")
    print(f"  Auto mode: {toolkit['auto_mode']}")
    
    return toolkit

if __name__ == "__main__":
    # Quick test
    toolkit = setup_auto_tools()
    
    # Test timer
    with auto_timer() as t:
        time.sleep(0.1)
    print(f"Timer test: {t:.3f}s")
    
    # Test profiler
    @auto_profile("test_function")
    def test_func():
        time.sleep(0.05)
        return "success"
    
    result = test_func()
    print(f"Profiler test: {result}")
    
    # Show stats
    profiler = toolkit["performance"]["profiler"]
    print(f"Performance stats: {profiler.get_stats()}")