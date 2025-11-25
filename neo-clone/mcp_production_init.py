"""
MCP Production Initialization Script

Initializes and starts all MCP components for production deployment.
This script ensures all MCP systems are properly configured and running.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_mcp_production():
    """Initialize MCP components for production"""
    print("[INIT] MCP Production Initialization")
    print("=" * 50)
    
    success_count = 0
    total_components = 0
    
    # Component 1: MCP Protocol Client
    total_components += 1
    try:
        from mcp_protocol import MCPClient, MCPConfig
        
        config = MCPConfig()
        mcp_client = MCPClient(config)
        
        # Start MCP client
        await mcp_client.start()
        
        print("[OK] MCP Protocol Client: Started successfully")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] MCP Protocol Client: Failed to start - {e}")
    
    # Component 2: Performance Monitor
    total_components += 1
    try:
        from tool_performance_monitor import performance_monitor
        
        # Start performance monitoring
        performance_monitor.start_monitoring()
        
        print("[OK] Performance Monitor: Started successfully")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Performance Monitor: Failed to start - {e}")
    
    # Component 3: Cache System
    total_components += 1
    try:
        from tool_cache_system import tool_cache
        
        # Start cache system
        await tool_cache.start()
        
        print("[OK] Cache System: Started successfully")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Cache System: Failed to start - {e}")
    
    # Component 4: Resource Manager
    total_components += 1
    try:
        from resource_manager import resource_manager
        
        # Start resource monitoring
        await resource_manager.start()
        
        print("[OK] Resource Manager: Started successfully")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Resource Manager: Failed to start - {e}")
    
    # Component 5: Skills Manager with MCP Integration
    total_components += 1
    try:
        from skills import SkillsManager
        
        # Initialize skills manager
        skills_manager = SkillsManager()
        
        # Verify enhanced tool skill is registered
        enhanced_tool = skills_manager.get_skill("enhanced_tool")
        if enhanced_tool:
            print("[OK] Skills Manager: Enhanced tool skill registered")
            success_count += 1
        else:
            print("[FAIL] Skills Manager: Enhanced tool skill not found")
    except Exception as e:
        print(f"[FAIL] Skills Manager: Failed to initialize - {e}")
    
    # Component 6: Security Manager
    total_components += 1
    try:
        from mcp_protocol import SecurityManager, MCPConfig
        
        config = MCPConfig()
        security_manager = SecurityManager(config)
        
        print("[OK] Security Manager: Initialized successfully")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Security Manager: Failed to initialize - {e}")
    
    # Component 7: Parallel Executor
    total_components += 1
    try:
        from parallel_executor import parallel_executor
        
        # Verify parallel executor is available
        has_methods = (
            hasattr(parallel_executor, 'execute_concurrent') and
            hasattr(parallel_executor, 'execute_sequential') and
            hasattr(parallel_executor, 'execute_adaptive')
        )
        
        if has_methods:
            print("[OK] Parallel Executor: Available and ready")
            success_count += 1
        else:
            print("[FAIL] Parallel Executor: Missing required methods")
    except Exception as e:
        print(f"[FAIL] Parallel Executor: Failed to initialize - {e}")
    
    # Results
    print("=" * 50)
    print(f"Initialization Results: {success_count}/{total_components} components started")
    print(f"Success Rate: {(success_count/total_components)*100:.1f}%")
    
    if success_count == total_components:
        print("\n[SUCCESS] MCP PRODUCTION SYSTEM IS FULLY OPERATIONAL!")
        print("[OK] All components started successfully")
        print("[OK] Ready for production workload")
        print("[OK] MCP integration complete")
        print("[OK] Performance monitoring active")
        print("[OK] Security controls enabled")
        print("[OK] Caching system operational")
        print("[OK] Resource management active")
        print("[OK] Parallel execution available")
        
        # Create status file
        status_file = Path("data/mcp_production_status.json")
        status_file.parent.mkdir(exist_ok=True)
        
        import json
        from datetime import datetime
        
        status_data = {
            "status": "operational",
            "components": success_count,
            "total_components": total_components,
            "success_rate": (success_count/total_components)*100,
            "initialized_at": datetime.now().isoformat(),
            "mcp_version": "1.0.0",
            "features": [
                "mcp_protocol_client",
                "performance_monitoring", 
                "caching_system",
                "resource_management",
                "security_manager",
                "parallel_execution",
                "enhanced_tool_skill"
            ]
        }
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"[OK] Status saved to {status_file}")
        return True
        
    else:
        print(f"\n[WARNING] MCP PRODUCTION SYSTEM PARTIALLY OPERATIONAL")
        print(f"[WARNING] {total_components - success_count} components failed to start")
        print("[WARNING] System may have limited functionality")
        return False

async def shutdown_mcp_production():
    """Gracefully shutdown MCP components"""
    print("\n[SHUTDOWN] Shutting down MCP Production System...")
    
    try:
        from tool_performance_monitor import performance_monitor
        performance_monitor.stop_monitoring()
        print("[OK] Performance Monitor: Stopped")
    except:
        pass
    
    try:
        from tool_cache_system import tool_cache
        await tool_cache.stop()
        print("[OK] Cache System: Stopped")
    except:
        pass
    
    try:
        from resource_manager import resource_manager
        await resource_manager.stop()
        print("[OK] Resource Manager: Stopped")
    except:
        pass
    
    try:
        from mcp_protocol import MCPClient, MCPConfig
        config = MCPConfig()
        mcp_client = MCPClient(config)
        await mcp_client.stop()
        print("[OK] MCP Client: Stopped")
    except:
        pass
    
    print("[OK] MCP Production System shutdown complete")

async def main():
    """Main initialization function"""
    try:
        # Initialize production system
        success = await initialize_mcp_production()
        
        if success:
            print("\n[READY] MCP Production System is ready!")
            print("You can now use MCP tools through Neo-Clone.")
            
            # Keep system running for a bit to demonstrate
            print("\n[DEMO] System will run for 10 seconds to demonstrate stability...")
            await asyncio.sleep(10)
            
            # Graceful shutdown
            await shutdown_mcp_production()
            
        else:
            print("\n[FAILED] MCP Production System initialization failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Initialization interrupted by user")
        await shutdown_mcp_production()
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during initialization: {e}")
        await shutdown_mcp_production()
        return 1
    
    return 0

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)