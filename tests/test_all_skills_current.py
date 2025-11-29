#!/usr/bin/env python3
"""Test all discovered skills"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "skills"))
sys.path.insert(0, str(Path(__file__).parent / "neo-clone"))

try:
    from opencode_skills_manager import OpenCodeSkillsManager
    
    print("Testing all discovered skills...")
    manager = OpenCodeSkillsManager()
    manager.initialize()
    
    print(f"Found {len(manager.skills)} skills\n")
    
    working_skills = 0
    total_skills = len(manager.skills)
    
    for skill_name, skill_info in manager.skills.items():
        print(f"Testing {skill_name}...")
        try:
            # Test skill execution with simple context
            result = manager.execute_skill(skill_name, "test")
            if result and hasattr(result, 'success') and result.success:
                print(f"  [OK] {skill_name}: SUCCESS")
                working_skills += 1
            else:
                print(f"  [FAIL] {skill_name}: FAILED - {getattr(result, 'error_message', 'Unknown error')}")
        except Exception as e:
            print(f"  [ERROR] {skill_name}: ERROR - {str(e)}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Working skills: {working_skills}/{total_skills}")
    print(f"Success rate: {(working_skills/total_skills)*100:.1f}%")
    
    if working_skills >= total_skills * 0.8:
        print("TARGET ACHIEVED 80%+ SUCCESS RATE!")
    else:
        print(f"Need {total_skills * 0.8 - working_skills:.1f} more skills working to reach 80% target")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()