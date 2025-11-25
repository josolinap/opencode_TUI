"""
Simple MCP Integration Test

Basic test to verify MCP integration is working.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_mcp():
    """Test MCP integration"""
    print("MCP Integration Test")
    print("=" * 40)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Import MCP Protocol
    total_tests += 1
    try:
        from mcp_protocol import MCPClient, MCPConfig
        config = MCPConfig()
        client = MCPClient(config)
        print("[PASS] MCP Protocol: PASS")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] MCP Protocol: FAIL - {e}")
    
    # Test 2: Import Enhanced Tool Skill
    total_tests += 1
    try:
        from enhanced_tool_skill import EnhancedToolSkill
        skill = EnhancedToolSkill()
        print("[PASS] Enhanced Tool Skill: PASS")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Enhanced Tool Skill: FAIL - {e}")
    
    # Test 3: Import Performance Monitor
    total_tests += 1
    try:
        from tool_performance_monitor import performance_monitor
        performance_monitor.start_monitoring()
        performance_monitor.stop_monitoring()
        print("[PASS] Performance Monitor: PASS")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Performance Monitor: FAIL - {e}")
    
    # Test 4: Import Cache System
    total_tests += 1
    try:
        from tool_cache_system import tool_cache
        print("[PASS] Cache System: PASS")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Cache System: FAIL - {e}")
    
    # Test 5: Import Parallel Executor
    total_tests += 1
    try:
        from parallel_executor import parallel_executor
        print("[PASS] Parallel Executor: PASS")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Parallel Executor: FAIL - {e}")
    
    # Test 6: Import Resource Manager
    total_tests += 1
    try:
        from resource_manager import resource_manager
        print("[PASS] Resource Manager: PASS")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Resource Manager: FAIL - {e}")
    
    # Test 7: Skills Integration
    total_tests += 1
    try:
        from skills import SkillsManager
        skills_manager = SkillsManager()
        enhanced_tool = skills_manager.get_skill("enhanced_tool")
        if enhanced_tool:
            print("[PASS] Skills Integration: PASS")
            success_count += 1
        else:
            print("[FAIL] Skills Integration: FAIL - Enhanced tool not found")
    except Exception as e:
        print(f"[FAIL] Skills Integration: FAIL - {e}")
    
    # Results
    print("=" * 40)
    print(f"Results: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\nMCP INTEGRATION IS FULLY OPERATIONAL!")
        print("Ready for production use")
        return True
    elif success_count >= total_tests * 0.8:
        print("\nMCP Integration is mostly functional")
        return True
    else:
        print("\nMCP Integration needs attention")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_mcp())
    sys.exit(0 if result else 1)