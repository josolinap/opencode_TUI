#!/usr/bin/env python3
"""
Check skill methods
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "skills"))

try:
    from opencode_skills_manager import OpenCodeSkillsManager
    
    manager = OpenCodeSkillsManager()
    manager.initialize()
    
    print('Registered skills:')
    for name, skill_info in manager.skills.items():
        print(f'{name}: {type(skill_info.instance)}')
        print(f'  Has execute: {hasattr(skill_info.instance, "execute")}')
        print(f'  Has _execute_async: {hasattr(skill_info.instance, "_execute_async")}')
        print()
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()