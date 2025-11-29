from functools import lru_cache
'\nSimple brain status check without autonomous evolution\n'
import sys
import os
from pathlib import Path
neo_clone_path = Path(__file__).parent / 'neo-clone'
sys.path.insert(0, str(neo_clone_path))
os.environ['DISABLE_AUTONOMOUS_EVOLUTION'] = '1'

@lru_cache(maxsize=128)
def check_brain_status():
    """Check brain status without triggering evolution"""
    print('Neo-Clone Brain Status Check')
    print('=' * 40)
    try:
        print('1. Testing brain import...')
        from brain.brain import Brain
        print('   [OK] Brain imported successfully')
        print('2. Testing configuration...')
        from config import get_config
        config = get_config()
        print(f'   [OK] Config: {config.provider} - {config.model_name}')
        print('3. Testing skills manager...')
        from skills import OpenCodeSkillsManager as SkillsManager
        skills_manager = SkillsManager()
        skills_count = len(skills_manager.list_skills())
        print(f'   [OK] Skills manager: {skills_count} skills')
        print('4. Testing brain initialization...')
        original_enable_optimization = config.enable_optimization
        config.enable_optimization = False
        brain = Brain(config, skills_manager)
        print('   [OK] Brain initialized')
        print('5. Checking brain components...')
        components = []
        if hasattr(brain, 'base_brain') and brain.base_brain:
            components.append('Base Brain')
        if hasattr(brain, 'enhanced_brain') and brain.enhanced_brain:
            components.append('Enhanced Brain')
        if hasattr(brain, 'llm') and brain.llm:
            components.append('LLM Client')
        if hasattr(brain, 'skills') and brain.skills:
            components.append('Skills Manager')
        print(f'   [OK] Active components: {len(components)}')
        print('6. Checking brain methods...')
        methods = ['process', 'chat', 'respond']
        available_methods = []
        for method in methods:
            if hasattr(brain, method):
                available_methods.append(method)
        print(f'   [OK] Available methods: {len(available_methods)}')
        print('\n' + '=' * 40)
        print('BRAIN STATUS: WORKING')
        print(f'  - Provider: {config.provider}')
        print(f'  - Model: {config.model_name}')
        print(f'  - Skills: {skills_count}')
        print(f'  - Components: {len(components)}')
        print(f'  - Methods: {len(available_methods)}')
        config.enable_optimization = original_enable_optimization
        return True
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        return False
if __name__ == '__main__':
    success = check_brain_status()
    print(f'\nExit code: {(0 if success else 1)}')
    sys.exit(0 if success else 1)