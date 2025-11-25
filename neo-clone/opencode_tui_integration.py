#!/usr/bin/env python3
"""
OpenCode TUI Integration for Neo-Clone
Automatically integrates performance tools when Neo-Clone is used in OpenCode TUI
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def integrate_with_opencode_tui():
    """Integrate Neo-Clone tools with OpenCode TUI - automatically called"""
    
    # Setup auto tools
    try:
        from neo_clone_auto_setup import setup_neo_clone_tools, get_tools, get_status
        setup_result = setup_neo_clone_tools()
        tools = get_tools()
        status = get_status()
        
        print(f"Neo-Clone tools integrated: {status}")
        return True
    except Exception as e:
        print(f"Neo-Clone integration failed: {e}")
        return False

# Auto-integrate when imported
integrate_with_opencode_tui()

# Export tools for OpenCode TUI
def get_neo_clone_enhancements():
    """Get Neo-Clone enhancements for OpenCode TUI"""
    try:
        from neo_clone_auto_setup import get_tools
        return get_tools()
    except:
        return {}

# Decorators for OpenCode TUI to use
def neo_clone_profile(func_name=None):
    """Profile Neo-Clone operations in OpenCode TUI"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                from auto_tool_integration import auto_profile
                name = func_name or f"opencode.{func.__name__}"
                profiled_func = auto_profile(name)(func)
                return profiled_func(*args, **kwargs)
            except:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def neo_clone_timer(func):
    """Time Neo-Clone operations in OpenCode TUI"""
    def wrapper(*args, **kwargs):
        try:
            from auto_tool_integration import auto_timer
            with auto_timer() as t:
                result = func(*args, **kwargs)
            return result
        except:
            return func(*args, **kwargs)
    return wrapper

# Performance optimization for OpenCode TUI
def optimize_for_operation(operation_type="general"):
    """Get optimization settings for OpenCode TUI operations"""
    try:
        from auto_tool_integration import auto_optimize_operation
        return auto_optimize_operation(operation_type)
    except:
        return {
            "use_threading": True,
            "batch_size": 10,
            "timeout": 30,
            "retry_attempts": 3
        }

# Quick access to performance tools
def get_performance_tools():
    """Get performance tools for OpenCode TUI"""
    try:
        from auto_tool_integration import get_performance_tools
        return get_performance_tools()
    except:
        return {}

if __name__ == "__main__":
    print("OpenCode TUI Integration for Neo-Clone")
    print("This module automatically integrates when imported")
    print("Neo-Clone tools are now available in OpenCode TUI")