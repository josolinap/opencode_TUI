#!/usr/bin/env python3
"""
Simple MCP Tools Usage Example
Shows how to use MCP tools directly without complex integration
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extended_mcp_tools import ExtendedToolExecutor

async def simple_mcp_demo():
    """Simple demonstration of MCP tools"""
    print("Simple MCP Tools Demo")
    print("=" * 40)
    
    executor = ExtendedToolExecutor()
    
    # Example 1: Analyze text sentiment
    print("\n1. Text Sentiment Analysis:")
    result = await executor.execute_tool("mcp_text_analyzer", {
        "text": "I love this new AI technology! It's absolutely amazing.",
        "analysis_types": ["sentiment"]
    })
    
    if result.get('success'):
        sentiment = result['result']['sentiment']
        print(f"   Sentiment: {sentiment['label']} (score: {sentiment['polarity']})")
    else:
        print(f"   Error: {result.get('error')}")
    
    # Example 2: Check system resources
    print("\n2. System Resource Check:")
    result = await executor.execute_tool("mcp_system_monitor", {
        "metrics": ["cpu", "memory"]
    })
    
    if result.get('success'):
        data = result['result']
        print(f"   CPU: {data['cpu_usage_percent']:.1f}%")
        print(f"   Memory: {data['memory_usage_percent']:.1f}%")
    else:
        print(f"   Error: {result.get('error')}")
    
    # Example 3: Transform text
    print("\n3. Text Transformation:")
    result = await executor.execute_tool("mcp_text_transformer", {
        "text": "Hello World",
        "operations": ["uppercase", "reverse"]
    })
    
    if result.get('success'):
        data = result['result']
        print(f"   Original: {data['original_text']}")
        print(f"   Transformed: {data['transformed_text']}")
    else:
        print(f"   Error: {result.get('error')}")
    
    # Example 4: Make API call
    print("\n4. API Call:")
    result = await executor.execute_tool("mcp_rest_api_call", {
        "url": "https://httpbin.org/json",
        "method": "GET"
    })
    
    if result.get('success'):
        data = result['result']
        print(f"   Status: {data['status_code']}")
        print(f"   Response time: {data['response_time_ms']}ms")
    else:
        print(f"   Error: {result.get('error')}")
    
    print("\n" + "=" * 40)
    print("Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(simple_mcp_demo())