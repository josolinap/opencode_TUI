#!/usr/bin/env python3
"""
Test current skills status
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "skills"))

try:
    from opencode_skills_manager import OpenCodeSkillsManager
    
    manager = OpenCodeSkillsManager()
    manager.initialize()
    
    skills = manager.list_skills()
    print('Testing individual skills:')
    print('=' * 50)
    
    working = 0
    failing = 0
    
    for skill_name in skills:
        try:
            result = manager.execute_skill(skill_name, 'test request')
            if hasattr(result, 'success') and result.success:
                print(f'[OK] {skill_name}: SUCCESS')
                working += 1
            else:
                error_msg = getattr(result, 'error_message', str(result))
                print(f'[FAIL] {skill_name}: FAILED - {error_msg}')
                failing += 1
        except Exception as e:
            print(f'[ERROR] {skill_name}: ERROR - {str(e)}')
            failing += 1
    
    print('=' * 50)
    print(f'Total Skills: {len(skills)}')
    print(f'Working: {working}')
    print(f'Failing: {failing}')
    print(f'Success Rate: {working/len(skills)*100:.1f}%')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()