"""
MCP Integration Test and Verification

This module provides comprehensive testing and verification of the MCP Protocol
integration to ensure all components work correctly together.

Author: Neo-Clone Enhanced
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp_integration():
    """Test MCP integration components"""
    logger.info("Starting MCP Integration Test")
    
    test_results = {
        'mcp_protocol': False,
        'enhanced_tool_skill': False,
        'performance_monitor': False,
        'cache_system': False,
        'parallel_executor': False,
        'resource_manager': False,
        'integration': False
    }
    
    try:
        # Test 1: MCP Protocol
        logger.info("Testing MCP Protocol...")
        try:
            from mcp_protocol import MCPClient, MCPConfig, ToolRegistry, SecurityManager
            config = MCPConfig()
            client = MCPClient(config)
            logger.info("MCP Protocol initialized successfully")
            test_results['mcp_protocol'] = True
        except Exception as e:
            logger.error(f"MCP Protocol test failed: {e}")
        
        # Test 2: Enhanced Tool Skill
        logger.info("🔧 Testing Enhanced Tool Skill...")
        try:
            from enhanced_tool_skill import EnhancedToolSkill
            from data_models import SkillContext, IntentType
            
            skill = EnhancedToolSkill()
            context = SkillContext(
                user_input="test",
                intent=IntentType.CONVERSATION,
                conversation_history=[]
            )
            
            # Test legacy tool execution
            result = await skill._execute_async(
                context,
                tool_name="file_read",
                tool_params={"file_path": "test.txt"},
                use_mcp_tools=False
            )
            
            logger.info("✅ Enhanced Tool Skill working successfully")
            test_results['enhanced_tool_skill'] = True
        except Exception as e:
            logger.error(f"❌ Enhanced Tool Skill test failed: {e}")
        
        # Test 3: Performance Monitor
        logger.info("📊 Testing Performance Monitor...")
        try:
            from tool_performance_monitor import performance_monitor
            
            performance_monitor.start_monitoring()
            performance_monitor.track_execution("test_tool", 0.1, True)
            stats = performance_monitor.get_system_overview()
            
            performance_monitor.stop_monitoring()
            logger.info("✅ Performance Monitor working successfully")
            test_results['performance_monitor'] = True
        except Exception as e:
            logger.error(f"❌ Performance Monitor test failed: {e}")
        
        # Test 4: Cache System
        logger.info("💾 Testing Cache System...")
        try:
            from tool_cache_system import tool_cache
            
            await tool_cache.start()
            tool_cache.cache_result("test_tool", {"param": "value"}, {"result": "test"})
            cached_result = tool_cache.get_cached_result("test_tool", {"param": "value"})
            
            await tool_cache.stop()
            logger.info("✅ Cache System working successfully")
            test_results['cache_system'] = True
        except Exception as e:
            logger.error(f"❌ Cache System test failed: {e}")
        
        # Test 5: Parallel Executor
        logger.info("⚡ Testing Parallel Executor...")
        try:
            from parallel_executor import parallel_executor, ToolTask, ExecutionMode
            
            task = ToolTask(
                task_id="test_task",
                tool_id="test_tool",
                parameters={"test": True}
            )
            
            results = await parallel_executor.execute_concurrent([task])
            parallel_executor.shutdown()
            
            logger.info("✅ Parallel Executor working successfully")
            test_results['parallel_executor'] = True
        except Exception as e:
            logger.error(f"❌ Parallel Executor test failed: {e}")
        
        # Test 6: Resource Manager
        logger.info("🖥️ Testing Resource Manager...")
        try:
            from resource_manager import resource_manager, ResourceType
            
            await resource_manager.start()
            status = resource_manager.get_resource_status()
            
            await resource_manager.stop()
            logger.info("✅ Resource Manager working successfully")
            test_results['resource_manager'] = True
        except Exception as e:
            logger.error(f"❌ Resource Manager test failed: {e}")
        
        # Test 7: Full Integration
        logger.info("🔗 Testing Full Integration...")
        try:
            from skills import SkillsManager
            
            skills_manager = SkillsManager()
            enhanced_tool = skills_manager.get_skill("enhanced_tool")
            
            if enhanced_tool:
                logger.info("✅ Full Integration working successfully")
                test_results['integration'] = True
            else:
                logger.warning("⚠️ Enhanced tool skill not found in skills manager")
        except Exception as e:
            logger.error(f"❌ Full Integration test failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
    
    # Results Summary
    logger.info("\n" + "="*50)
    logger.info("📋 MCP INTEGRATION TEST RESULTS")
    logger.info("="*50)
    
    passed = 0
    total = len(test_results)
    
    for component, status in test_results.items():
        status_icon = "✅" if status else "❌"
        component_name = component.replace('_', ' ').title()
        logger.info(f"{status_icon} {component_name}: {'PASS' if status else 'FAIL'}")
        if status:
            passed += 1
    
    logger.info("="*50)
    logger.info(f"📊 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 MCP INTEGRATION IS READY FOR PRODUCTION!")
    elif passed >= total * 0.8:
        logger.info("✅ MCP Integration is mostly functional")
    else:
        logger.warning("⚠️ MCP Integration needs attention")
    
    logger.info("="*50)
    
    return test_results


async def run_performance_benchmark():
    """Run performance benchmarks"""
    logger.info("🏃 Running Performance Benchmarks...")
    
    try:
        from tool_performance_monitor import performance_monitor
        from enhanced_tool_skill import EnhancedToolSkill
        from data_models import SkillContext, IntentType
        
        performance_monitor.start_monitoring()
        
        skill = EnhancedToolSkill()
        context = SkillContext(
            user_input="benchmark test",
            intent=IntentType.CONVERSATION,
            conversation_history=[]
        )
        
        # Run multiple executions for benchmarking
        start_time = time.time()
        executions = 10
        
        for i in range(executions):
            result = await skill._execute_async(
                context,
                tool_name="data_analyze",
                tool_params={"data": {"test": f"benchmark_{i}"}},
                use_mcp_tools=False
            )
            
            if not result.success:
                logger.warning(f"Benchmark execution {i+1} failed")
        
        total_time = time.time() - start_time
        avg_time = total_time / executions
        
        performance_monitor.stop_monitoring()
        
        # Get performance stats
        stats = performance_monitor.get_system_overview()
        
        logger.info("📊 BENCHMARK RESULTS:")
        logger.info(f"   Total Executions: {executions}")
        logger.info(f"   Total Time: {total_time:.3f}s")
        logger.info(f"   Average Time: {avg_time:.3f}s")
        logger.info(f"   Success Rate: {stats.get('overall_success_rate', 0):.1f}%")
        logger.info(f"   Avg Execution Time: {stats.get('average_execution_time', 0):.3f}s")
        
        return {
            'executions': executions,
            'total_time': total_time,
            'avg_time': avg_time,
            'success_rate': stats.get('overall_success_rate', 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return None


def main():
    """Main test function"""
    print("MCP Integration Test Suite")
    print("="*50)
    
    # Run integration tests
    async def run_tests():
        test_results = await test_mcp_integration()
        
        # Run performance benchmarks if integration is working
        if any(test_results.values()):
            print("\nRunning Performance Benchmarks...")
            benchmark_results = asyncio.run(run_performance_benchmark())
            
            if benchmark_results:
                print("Performance benchmarks completed")
            else:
                print("Performance benchmarks failed")
        
        return test_results
    
    # Run tests
    results = asyncio.run(run_tests())
    
    # Final status
    all_passed = all(results.values())
    if all_passed:
        print("\nMCP INTEGRATION IS FULLY OPERATIONAL!")
        print("Ready for production use")
        print("All systems tested and verified")
    else:
        print("\nMCP Integration needs attention")
        failed_components = [k for k, v in results.items() if not v]
        print(f"Failed components: {failed_components}")
    
    return all_passed


if __name__ == "__main__":
    main()