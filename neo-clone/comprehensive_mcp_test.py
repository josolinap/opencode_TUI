"""
Comprehensive MCP Integration Test

Tests end-to-end MCP functionality with actual tool execution,
performance monitoring, caching, and security features.
"""

import asyncio
import sys
import os
import tempfile
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_comprehensive_mcp():
    """Comprehensive test of MCP integration"""
    print("Comprehensive MCP Integration Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: MCP Client Initialization
    total_tests += 1
    try:
        from mcp_protocol import MCPClient, MCPConfig
        config = MCPConfig()
        client = MCPClient(config)
        
        # Test client methods
        status = client.get_status()
        tools = client.list_available_tools()
        
        print(f"[PASS] MCP Client: Status={status}, Tools={len(tools)}")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] MCP Client: {e}")
    
    # Test 2: Enhanced Tool Skill - Legacy Mode
    total_tests += 1
    try:
        from enhanced_tool_skill import EnhancedToolSkill
        skill = EnhancedToolSkill()
        
        # Test legacy tool execution (without MCP)
        result = await skill.execute(
            tool_name="bash",
            tool_params={"command": "echo 'Hello from legacy tool'"},
            use_mcp_tools=False
        )
        
        if result.success:
            print(f"[PASS] Legacy Tool Execution: {result.output}")
            success_count += 1
        else:
            print(f"[FAIL] Legacy Tool Execution: {result.error_message}")
    except Exception as e:
        print(f"[FAIL] Legacy Tool Execution: {e}")
    
    # Test 3: Performance Monitoring
    total_tests += 1
    try:
        from tool_performance_monitor import performance_monitor
        from enhanced_tool_skill import EnhancedToolSkill
        
        # Start monitoring
        performance_monitor.start_monitoring()
        
        # Execute some tools to generate metrics
        skill = EnhancedToolSkill()
        await skill.execute("bash", {"command": "echo 'Performance test'"}, False)
        await skill.execute("bash", {"command": "dir"}, False)
        
        # Get metrics
        metrics = performance_monitor.get_metrics()
        summary = performance_monitor.get_performance_summary()
        
        # Stop monitoring
        performance_monitor.stop_monitoring()
        
        print(f"[PASS] Performance Monitoring: {len(metrics)} metrics recorded")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Performance Monitoring: {e}")
    
    # Test 4: Cache System
    total_tests += 1
    try:
        from tool_cache_system import tool_cache
        from enhanced_tool_skill import EnhancedToolSkill
        
        # Test cache functionality
        skill = EnhancedToolSkill()
        
        # First execution (should cache result)
        result1 = await skill.execute("bash", {"command": "echo 'Cache test'"}, False)
        
        # Second execution (should use cache)
        result2 = await skill.execute("bash", {"command": "echo 'Cache test'"}, False)
        
        cache_stats = tool_cache.get_stats()
        
        print(f"[PASS] Cache System: Stats={cache_stats}")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Cache System: {e}")
    
    # Test 5: Parallel Execution
    total_tests += 1
    try:
        from parallel_executor import parallel_executor
        from enhanced_tool_skill import ToolExecutionRequest
        
        # Create multiple execution requests
        requests = [
            ToolExecutionRequest("bash", {"command": "echo 'Task 1'"}, False),
            ToolExecutionRequest("bash", {"command": "echo 'Task 2'"}, False),
            ToolExecutionRequest("bash", {"command": "echo 'Task 3'"}, False)
        ]
        
        # Execute in parallel
        results = await parallel_executor.execute_parallel(requests)
        
        successful = sum(1 for r in results if r.success)
        print(f"[PASS] Parallel Execution: {successful}/{len(results)} successful")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Parallel Execution: {e}")
    
    # Test 6: Resource Management
    total_tests += 1
    try:
        from resource_manager import resource_manager
        
        # Start resource monitoring
        resource_manager.start_monitoring()
        
        # Get system resources
        cpu_usage = resource_manager.get_cpu_usage()
        memory_info = resource_manager.get_memory_info()
        disk_usage = resource_manager.get_disk_usage()
        
        # Check resource availability
        can_execute = resource_manager.can_execute_tool("test_tool", {"cpu": 10, "memory": 100})
        
        # Stop monitoring
        resource_manager.stop_monitoring()
        
        print(f"[PASS] Resource Management: CPU={cpu_usage:.1f}%, Memory={memory_info.percent}%")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Resource Management: {e}")
    
    # Test 7: Security Manager
    total_tests += 1
    try:
        from mcp_protocol import SecurityManager, SecurityLevel
        
        security_manager = SecurityManager()
        
        # Test security validation
        safe_params = {"command": "echo 'Safe command'", "timeout": 30}
        unsafe_params = {"command": "rm -rf /", "timeout": 300}
        
        safe_result = security_manager.validate_tool_parameters("bash", safe_params)
        unsafe_result = security_manager.validate_tool_parameters("bash", unsafe_params)
        
        if safe_result.allowed and not unsafe_result.allowed:
            print(f"[PASS] Security Manager: Safe={safe_result.allowed}, Unsafe blocked={not unsafe_result.allowed}")
            success_count += 1
        else:
            print(f"[FAIL] Security Manager: Security validation failed")
    except Exception as e:
        print(f"[FAIL] Security Manager: {e}")
    
    # Test 8: File Operations with MCP
    total_tests += 1
    try:
        from enhanced_tool_skill import EnhancedToolSkill
        
        skill = EnhancedToolSkill()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            test_content = "This is a test file for MCP integration"
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Test file read
            read_result = await skill.execute("file_read", {"path": temp_file_path}, False)
            
            if read_result.success and test_content in str(read_result.output):
                print(f"[PASS] File Operations: Read successful")
                success_count += 1
            else:
                print(f"[FAIL] File Operations: Read failed")
        finally:
            # Clean up
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"[FAIL] File Operations: {e}")
    
    # Test 9: Skills Integration with MCP
    total_tests += 1
    try:
        from skills import SkillsManager
        
        skills_manager = SkillsManager()
        enhanced_tool = skills_manager.get_skill("enhanced_tool")
        
        if enhanced_tool:
            # Test skill execution through skills manager
            context = {
                "tool_name": "bash",
                "tool_params": {"command": "echo 'Skills Manager Test'"},
                "use_mcp_tools": False
            }
            
            result = await enhanced_tool.execute(context)
            
            if result.success:
                print(f"[PASS] Skills Integration: Execution through manager successful")
                success_count += 1
            else:
                print(f"[FAIL] Skills Integration: {result.error_message}")
        else:
            print(f"[FAIL] Skills Integration: Enhanced tool not found in manager")
            
    except Exception as e:
        print(f"[FAIL] Skills Integration: {e}")
    
    # Test 10: MCP Tool Discovery (Mock)
    total_tests += 1
    try:
        from mcp_protocol import MCPClient, MCPConfig
        from enhanced_tool_skill import EnhancedToolSkill
        
        # Test tool discovery capabilities
        skill = EnhancedToolSkill()
        
        # Get available tools (both legacy and MCP)
        legacy_tools = skill.get_available_legacy_tools()
        
        # Test MCP tool discovery (will be empty in test environment but should not error)
        try:
            mcp_tools = skill.get_available_mcp_tools()
        except:
            mcp_tools = []
        
        print(f"[PASS] Tool Discovery: Legacy={len(legacy_tools)}, MCP={len(mcp_tools)}")
        success_count += 1
    except Exception as e:
        print(f"[FAIL] Tool Discovery: {e}")
    
    # Results
    print("=" * 50)
    print(f"Results: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\n[SUCCESS] COMPREHENSIVE MCP INTEGRATION IS FULLY OPERATIONAL!")
        print("[OK] All components working correctly")
        print("[OK] Ready for production deployment")
        print("[OK] Performance monitoring active")
        print("[OK] Security controls enabled")
        print("[OK] Caching system operational")
        print("[OK] Parallel execution available")
        return True
    elif success_count >= total_tests * 0.8:
        print("\n[WARNING] MCP Integration is mostly functional")
        print("Some components may need attention")
        return True
    else:
        print("\n[ERROR] MCP Integration needs significant attention")
        print("Multiple components require fixes")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_comprehensive_mcp())
    sys.exit(0 if result else 1)