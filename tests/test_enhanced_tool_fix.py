#!/usr/bin/env python3
"""Test enhanced_tool skill fix"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "skills"))
sys.path.insert(0, str(Path(__file__).parent / "neo-clone"))

try:
    from opencode_skills_manager import OpenCodeSkillsManager
    
    print("Testing enhanced_tool skill...")
    manager = OpenCodeSkillsManager()
    manager.initialize()
    
    result = manager.execute_skill('enhanced_tool', 'test')
    print(f"Result type: {type(result)}")
    print(f"Success: {getattr(result, 'success', 'No success attr')}")
    if hasattr(result, 'error_message'):
        print(f"Error: {result.error_message}")
    else:
        print(f"Result: {result}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()