"""
Show Available MCP Tools

Simple demonstration of available MCP tools.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def show_mcp_tools():
    """Show available MCP tools"""
    print("MCP Tools Available")
    print("=" * 40)
    
    try:
        from mcp_protocol import MCPClient, MCPConfig
        
        # Initialize MCP client
        config = MCPConfig()
        mcp_client = MCPClient(config)
        await mcp_client.start()
        
        # Discover available tools
        tools = await mcp_client.discover_tools()
        
        print(f"Found {len(tools)} MCP Tools:")
        print()
        
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}")
            print(f"   ID: {tool.id}")
            print(f"   Category: {tool.category}")
            print(f"   Description: {tool.description}")
            
            if tool.parameters:
                print("   Parameters:")
                for param in tool.parameters:
                    required = "Required" if param.required else "Optional"
                    default = f" (default: {param.default})" if param.default is not None else ""
                    print(f"     - {param.name} ({param.param_type}) - {required}{default}")
                    if param.description:
                        print(f"       {param.description}")
            
            print()
        
        await mcp_client.stop()
        
        print("How to use MCP tools:")
        print("-" * 25)
        print("1. Through EnhancedToolSkill:")
        print("   enhanced_tool.execute({")
        print("       'tool_name': 'mcp_file_reader',")
        print("       'tool_params': {'file_path': '/path/to/file.txt'},")
        print("       'use_mcp_tools': True")
        print("   })")
        print()
        print("2. Through Neo-Clone skills manager:")
        print("   skills_manager.get_skill('enhanced_tool')")
        print()
        print("3. Start MCP production system:")
        print("   py mcp_production_init.py")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(show_mcp_tools())