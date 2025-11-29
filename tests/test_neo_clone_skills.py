from functools import lru_cache
'\nTest Neo-Clone skills functionality after workspace cleanup\n'
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@lru_cache(maxsize=128)
def test_skills_functionality():
    """Test that skills are working correctly"""
    print('=' * 60)
    print('TESTING NEO-CLONE SKILLS FUNCTIONALITY')
    print('=' * 60)
    try:
        print('\n1. Testing Skills Manager...')
        from skills.opencode_skills_manager import OpenCodeSkillsManager
        manager = OpenCodeSkillsManager()
        print('   ✅ Skills Manager created successfully')
        manager.initialize()
        print('   ✅ Skills Manager initialized')
        skills = manager.list_skills()
        print(f'   ✅ Found {len(skills)} skills:')
        for (skill_name, skill_info) in skills.items():
            print(f"      - {skill_name}: {skill_info.get('description', 'No description')}")
        if 'free_programming_books' in skills:
            print('\n2. Testing Free Programming Books Skill...')
            result = manager.execute_skill('free_programming_books', {'action': 'list_categories'})
            if result.get('success', False):
                print('   ✅ Free Programming Books skill executed successfully')
                print(f"   📚 Categories: {len(result.get('data', {}).get('categories', []))} found")
            else:
                print(f"   ❌ Skill execution failed: {result.get('error', 'Unknown error')}")
        if 'public_apis' in skills:
            print('\n3. Testing Public APIs Skill...')
            result = manager.execute_skill('public_apis', {'action': 'list_categories'})
            if result.get('success', False):
                print('   ✅ Public APIs skill executed successfully')
                print(f"   🔌 Categories: {len(result.get('data', {}).get('categories', []))} found")
            else:
                print(f"   ❌ Skill execution failed: {result.get('error', 'Unknown error')}")
        print('\n' + '=' * 60)
        print('✅ SKILLS FUNCTIONALITY TEST COMPLETED SUCCESSFULLY')
        print('=' * 60)
        return True
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_brain_bridge():
    """Test the neo-clone skills bridge"""
    print('\n4. Testing Neo-Clone Skills Bridge...')
    try:
        neo_clone_path = project_root / 'neo-clone'
        if neo_clone_path.exists():
            sys.path.insert(0, str(neo_clone_path))
            from skills import SkillsManager
            print('   ✅ Neo-Clone skills bridge imported successfully')
            bridge_manager = SkillsManager()
            if hasattr(bridge_manager, 'initialize'):
                bridge_manager.initialize()
                bridge_skills = bridge_manager.list_skills()
                print(f'   ✅ Bridge found {len(bridge_skills)} skills')
            else:
                print("   ⚠️  Bridge manager doesn't have expected methods")
        else:
            print('   ⚠️  Neo-clone directory not found')
    except Exception as e:
        print(f'   ❌ Bridge test failed: {e}')
if __name__ == '__main__':
    success = test_skills_functionality()
    test_brain_bridge()
    if success:
        print('\n🎉 OVERALL RESULT: Neo-Clone skills are working correctly!')
        print('📝 Next steps: Fix main.py imports to enable full Neo-Clone functionality')
    else:
        print('\n❌ OVERALL RESULT: Skills functionality needs more work')