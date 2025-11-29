#!/usr/bin/env python3
"""
Simple test for Neo-Clone skills functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

def test_skills():
    """Test skills functionality"""
    print("=" * 60)
    print("TESTING NEO-CLONE SKILLS")
    print("=" * 60)
    
    try:
        # Import skills manager
        from skills.opencode_skills_manager import OpenCodeSkillsManager
        
        print("1. Creating Skills Manager...")
        manager = OpenCodeSkillsManager()
        print("   SUCCESS: Skills Manager created")
        
        print("2. Initializing Skills Manager...")
        manager.initialize()
        print("   SUCCESS: Skills Manager initialized")
        
        print("3. Listing available skills...")
        skills = manager.list_skills()
        print(f"   SUCCESS: Found {len(skills)} skills:")
        for skill_name, skill_info in skills.items():
            desc = skill_info.get('description', 'No description')
            print(f"      - {skill_name}: {desc}")
        
        print("4. Testing skill execution...")
        if 'free_programming_books' in skills:
            result = manager.execute_skill('free_programming_books', {'action': 'list_categories'})
            if result.get('success', False):
                print("   SUCCESS: Free Programming Books skill executed")
                categories = result.get('data', {}).get('categories', [])
                print(f"   Found {len(categories)} categories")
            else:
                print(f"   ERROR: Skill execution failed: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 60)
        print("SKILLS TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_skills()
    if success:
        print("\nOVERALL RESULT: Skills are working correctly!")
    else:
        print("\nOVERALL RESULT: Skills test failed")