#!/usr/bin/env python3
"""
Neo-Clone Auto Setup - Automatic Tool Integration
Automatically integrates performance tools with Neo-Clone for OpenCode TUI
No manual setup required - just import and use
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_neo_clone_tools():
    """Setup Neo-Clone with automatic tools - call this once at startup"""
    
    # Import auto tool integration
    try:
        from auto_tool_integration import setup_auto_tools, auto_integrate_neo_clone
    except ImportError:
        print("Auto tool integration not available - using basic setup")
        return {"status": "basic", "tools": {}}
    
    # Setup auto tools
    toolkit = setup_auto_tools()
    
    # Try to integrate with existing Neo-Clone
    try:
        # Try to get Neo-Clone instance if it exists
        neo_clone = None
        
        # Try common import paths
        try:
            from main import NeoClone
            neo_clone = NeoClone()
        except:
            pass
            
        try:
            from enhanced_brain import EnhancedBrain
            neo_clone = EnhancedBrain()
        except:
            pass
        
        # Integrate if we found a Neo-Clone instance
        if neo_clone:
            integrated_toolkit = auto_integrate_neo_clone(neo_clone)
            return {
                "status": "integrated", 
                "tools": integrated_toolkit,
                "neo_clone": neo_clone
            }
        else:
            return {
                "status": "tools_ready", 
                "tools": toolkit,
                "message": "Tools ready, integrate manually with auto_integrate_neo_clone(neo_clone_instance)"
            }
            
    except Exception as e:
        print(f"Integration failed: {e}")
        return {
            "status": "tools_only", 
            "tools": toolkit,
            "error": str(e)
        }

def get_neo_clone_tools():
    """Get available tools for Neo-Clone"""
    try:
        from auto_tool_integration import get_performance_tools, get_auto_integration
        integration = get_auto_integration()
        return {
            "performance": get_performance_tools(),
            "available": integration.available_tools,
            "auto_mode": integration.auto_enabled
        }
    except ImportError:
        return {"performance": {}, "available": {}, "auto_mode": False}

# Auto-execute setup when imported
if __name__ != "__main__":
    # Auto-setup when imported by OpenCode TUI
    setup_result = setup_neo_clone_tools()
    
    # Make tools available globally
    NEO_CLONE_TOOLS = setup_result.get("tools", {})
    NEO_CLONE_STATUS = setup_result.get("status", "unknown")

# Easy access functions
def get_tools():
    """Get Neo-Clone tools"""
    return NEO_CLONE_TOOLS if 'NEO_CLONE_TOOLS' in globals() else get_neo_clone_tools()

def get_status():
    """Get setup status"""
    return NEO_CLONE_STATUS if 'NEO_CLONE_STATUS' in globals() else "not_setup"

# Quick decorators for Neo-Clone methods
def profile_method(method_name=None):
    """Decorator to profile Neo-Clone methods"""
    def decorator(func):
        try:
            from auto_tool_integration import auto_profile
            name = method_name or f"neo_clone.{func.__name__}"
            return auto_profile(name)(func)
        except ImportError:
            return func
    return decorator

def time_method(func):
    """Decorator to time Neo-Clone methods"""
    def wrapper(*args, **kwargs):
        try:
            from auto_tool_integration import auto_timer
            with auto_timer() as t:
                result = func(*args, **kwargs)
            # Store timing info (could be logged or tracked)
            return result
        except ImportError:
            return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    # Test setup
    print("Testing Neo-Clone Auto Setup...")
    result = setup_neo_clone_tools()
    print(f"Setup status: {result['status']}")
    print(f"Available tools: {len(result.get('tools', {}))}")
    
    # Test tools
    tools = get_tools()
    print(f"Performance tools: {list(tools.get('performance', {}).keys())}")
    print(f"Available modules: {list(tools.get('available', {}).keys())}")
    
    print("\nNeo-Clone Auto Setup Complete!")
    print("Tools are now available for Neo-Clone to use automatically.")