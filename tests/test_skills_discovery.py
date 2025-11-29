#!/usr/bin/env python3
"""Test script to check skills discovery"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "skills"))
sys.path.insert(0, str(Path(__file__).parent / "neo-clone"))

try:
    from opencode_skills_manager import OpenCodeSkillsManager
    
    print("Initializing skills manager...")
    manager = OpenCodeSkillsManager()
    
    print("Discovering skills...")
    success = manager.initialize()
    
    print(f"Initialization successful: {success}")
    print(f"Number of skills discovered: {len(manager.skills)}")
    
    print("\nDiscovered skills:")
    for name, skill_info in manager.skills.items():
        print(f"  {name}: {skill_info.class_obj.__module__}.{skill_info.class_obj.__name__}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()