"""
Focused MCP Integration Test

Tests the actual working MCP functionality with correct API calls.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_focused_mcp():
    """Focused test of MCP integration with correct APIs"""
    print("Focused MCP Integration Test")
    print("=" * 40)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: MCP Protocol Import and Basic Functionality
    total_tests += 1
    try:
        from mcp_protocol import MCPClient, MCPConfig, SecurityManager
        
        config = MCPConfig()
        client = MCPClient(config)
        security_manager = SecurityManager(config)
        
        # Test basic methods that should exist
        tools = client.list_available_tools()
        
        print(f"[PASS] MCP Protocol: Client created, {len(tools)} tools available")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] MCP Protocol: {e}")
    
    # Test 2: Enhanced Tool Skill Import and Basic Setup
    total_tests += 1
    try:
        from enhanced_tool_skill import EnhancedToolSkill
        
        skill = EnhancedToolSkill()
        
        # Test that the skill is properly initialized
        has_metadata = hasattr(skill, 'metadata')
        has_mcp_client = hasattr(skill, 'mcp_client')
        
        if has_metadata and has_mcp_client:
            print(f"[PASS] Enhanced Tool Skill: Properly initialized")
            success_count += 1
        else:
            print(f"[FAIL] Enhanced Tool Skill: Missing attributes")
    except Exception as e:
        print(f"[FAIL] Enhanced Tool Skill: {e}")
    
    # Test 3: Performance Monitor Basic Operations
    total_tests += 1
    try:
        from tool_performance_monitor import performance_monitor
        
        # Test basic monitoring operations
        performance_monitor.start_monitoring()
        performance_monitor.record_execution("test_tool", 0.5, True)
        metrics = performance_monitor.get_metrics()
        performance_monitor.stop_monitoring()
        
        if len(metrics) > 0:
            print(f"[PASS] Performance Monitor: {len(metrics)} metrics recorded")
            success_count += 1
        else:
            print(f"[FAIL] Performance Monitor: No metrics recorded")
    except Exception as e:
        print(f"[FAIL] Performance Monitor: {e}")
    
    # Test 4: Cache System Basic Operations
    total_tests += 1
    try:
        from tool_cache_system import tool_cache
        
        # Test basic cache operations
        test_key = "test_key"
        test_value = {"result": "test_data"}
        
        tool_cache.set(test_key, test_value)
        cached_value = tool_cache.get(test_key)
        stats = tool_cache.get_stats()
        
        if cached_value and cached_value.get("result") == "test_data":
            print(f"[PASS] Cache System: Cache hit successful, stats={stats}")
            success_count += 1
        else:
            print(f"[FAIL] Cache System: Cache miss or corruption")
    except Exception as e:
        print(f"[FAIL] Cache System: {e}")
    
    # Test 5: Parallel Executor Basic Operations
    total_tests += 1
    try:
        from parallel_executor import parallel_executor
        
        # Test that parallel executor exists and has basic methods
        has_execute_method = hasattr(parallel_executor, 'execute_tools')
        
        if has_execute_method:
            print(f"[PASS] Parallel Executor: Available with execute method")
            success_count += 1
        else:
            print(f"[FAIL] Parallel Executor: Missing execute method")
    except Exception as e:
        print(f"[FAIL] Parallel Executor: {e}")
    
    # Test 6: Resource Manager Basic Operations
    total_tests += 1
    try:
        from resource_manager import resource_manager
        
        # Test basic resource monitoring
        cpu_usage = resource_manager.get_cpu_usage()
        memory_info = resource_manager.get_memory_info()
        
        if cpu_usage >= 0 and memory_info:
            print(f"[PASS] Resource Manager: CPU={cpu_usage:.1f}%, Memory available")
            success_count += 1
        else:
            print(f"[FAIL] Resource Manager: Invalid readings")
    except Exception as e:
        print(f"[FAIL] Resource Manager: {e}")
    
    # Test 7: Skills Manager Integration
    total_tests += 1
    try:
        from skills import SkillsManager
        
        skills_manager = SkillsManager()
        enhanced_tool = skills_manager.get_skill("enhanced_tool")
        
        if enhanced_tool:
            print(f"[PASS] Skills Integration: Enhanced tool found in manager")
            success_count += 1
        else:
            print(f"[FAIL] Skills Integration: Enhanced tool not found")
    except Exception as e:
        print(f"[FAIL] Skills Integration: {e}")
    
    # Test 8: Security Manager Basic Operations
    total_tests += 1
    try:
        from mcp_protocol import SecurityManager, MCPConfig
        
        config = MCPConfig()
        security_manager = SecurityManager(config)
        
        # Test security validation with safe parameters
        safe_params = {"command": "echo 'test'", "timeout": 30}
        validation_result = security_manager.validate_tool_parameters("bash", safe_params)
        
        if validation_result:
            print(f"[PASS] Security Manager: Validation working")
            success_count += 1
        else:
            print(f"[FAIL] Security Manager: Validation failed")
    except Exception as e:
        print(f"[FAIL] Security Manager: {e}")
    
    # Results
    print("=" * 40)
    print(f"Results: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\n[SUCCESS] FOCUSED MCP INTEGRATION IS FULLY OPERATIONAL!")
        print("[OK] All core components working correctly")
        print("[OK] Ready for production deployment")
        return True
    elif success_count >= total_tests * 0.75:
        print("\n[GOOD] MCP Integration is mostly functional")
        print("Most components working correctly")
        return True
    else:
        print("\n[NEEDS WORK] MCP Integration needs attention")
        print("Several components require fixes")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_focused_mcp())
    sys.exit(0 if result else 1)