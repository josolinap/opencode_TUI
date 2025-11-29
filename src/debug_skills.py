#!/usr/bin/env python3
"""
Debug script for skills discovery
"""

import sys
import os
import importlib.util
from pathlib import Path

# Add paths - go up one level from src to root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_skill_discovery():
    print("=" * 60)
    print("DEBUGGING SKILLS DISCOVERY")
    print("=" * 60)
    
    # Test paths
    skills_path = Path("skills")
    neo_clone_path = Path("neo-clone")
    
    print(f"\n1. Testing paths:")
    print(f"   Skills path exists: {skills_path.exists()}")
    print(f"   Neo-clone path exists: {neo_clone_path.exists()}")
    
    print(f"\n2. Testing imports:")
    
    # Test individual skill imports
    skill_files = [
        "neo-clone/code_generation.py",
        "neo-clone/text_analysis.py", 
        "neo-clone/data_inspector.py",
        "neo-clone/web_search.py",
        "neo-clone/file_manager.py"
    ]
    
    for skill_file in skill_files:
        file_path = Path(skill_file)
        if file_path.exists():
            print(f"\n   Testing {file_path}:")
            try:
                # Load module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                
                if spec is None or spec.loader is None:
                    print(f"     [ERROR] Could not create spec for {module_name}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                print(f"     [OK] Module {module_name} loaded successfully")
                
                # Look for skill classes
                skill_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    if (isinstance(attr, type) and 
                        attr_name.endswith("Skill") and 
                        hasattr(attr, '__bases__') and
                        attr_name != "BaseSkill"):
                        skill_classes.append(attr_name)
                        print(f"     [OK] Found skill class: {attr_name}")
                
                if not skill_classes:
                    print(f"     [WARN] No skill classes found in {module_name}")
                    
            except Exception as e:
                print(f"     [ERROR] Error loading {module_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   [ERROR] File not found: {skill_file}")
    
    print(f"\n3. Testing BaseSkill import:")
    try:
        from skills.base_skill import BaseSkill
        print("   [OK] BaseSkill imported successfully")
    except Exception as e:
        print(f"   [ERROR] BaseSkill import failed: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_skill_discovery()