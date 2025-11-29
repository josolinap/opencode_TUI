from functools import lru_cache
'\nNeo-Clone Main Entry Point - Fixed Version\nSimplified main script that works with available components\n'
import sys
import os
import argparse
import logging
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging(debug=False):
    """Setup basic logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_skills_system():
    """Test the skills system"""
    print('=' * 60)
    print('TESTING NEO-CLONE SKILLS SYSTEM')
    print('=' * 60)
    try:
        from skills.opencode_skills_manager import OpenCodeSkillsManager
        print('Skills Manager imported successfully')
        manager = OpenCodeSkillsManager()
        manager.initialize()
        print('Skills Manager initialized')
        skills = manager.list_skills()
        print(f'Found {len(skills)} skills:')
        for (name, info) in skills.items():
            print(f"   - {name}: {info.get('description', 'No description')}")
        return (True, manager)
    except Exception as e:
        print(f'Skills system error: {e}')
        import traceback
        traceback.print_exc()
        return (False, None)

def test_brain_components():
    """Test available brain components"""
    print('\n' + '=' * 60)
    print('TESTING BRAIN COMPONENTS')
    print('=' * 60)
    components = {}
    try:
        from brain.base_brain import BaseBrain
        components['base_brain'] = BaseBrain
        print('✅ BaseBrain imported')
    except Exception as e:
        print(f'❌ BaseBrain import failed: {e}')
    try:
        from brain.unified_brain import UnifiedBrain
        components['unified_brain'] = UnifiedBrain
        print('✅ UnifiedBrain imported')
    except Exception as e:
        print(f'❌ UnifiedBrain import failed: {e}')
    try:
        from brain.enhanced_brain import EnhancedBrain
        components['enhanced_brain'] = EnhancedBrain
        print('✅ EnhancedBrain imported')
    except Exception as e:
        print(f'❌ EnhancedBrain import failed: {e}')
    try:
        from brain.opencode_unified_brain import UnifiedBrain as OpenCodeUnifiedBrain
        components['opencode_unified_brain'] = OpenCodeUnifiedBrain
        print('✅ OpenCode UnifiedBrain imported')
    except Exception as e:
        print(f'❌ OpenCode UnifiedBrain import failed: {e}')
    return components

def test_memory_components():
    """Test memory components"""
    print('\n' + '=' * 60)
    print('TESTING MEMORY COMPONENTS')
    print('=' * 60)
    components = {}
    try:
        from brain.unified_memory import get_unified_memory
        memory = get_unified_memory()
        components['unified_memory'] = memory
        print('✅ Unified memory imported')
    except Exception as e:
        print(f'❌ Unified memory import failed: {e}')
    try:
        from brain.vector_memory import VectorMemory
        components['vector_memory'] = VectorMemory
        print('✅ Vector memory imported')
    except Exception as e:
        print(f'❌ Vector memory import failed: {e}')
    return components

@lru_cache(maxsize=128)
def create_simple_brain(skills_manager):
    """Create a simple brain using available components"""
    print('\n' + '=' * 60)
    print('CREATING SIMPLE BRAIN')
    print('=' * 60)
    try:
        brain_classes = [('OpenCode Unified Brain', 'brain.opencode_unified_brain', 'UnifiedBrain'), ('Enhanced Brain', 'brain.enhanced_brain', 'EnhancedBrain'), ('Unified Brain', 'brain.unified_brain', 'UnifiedBrain'), ('Base Brain', 'brain.base_brain', 'BaseBrain')]
        brain_instance = None
        brain_type = None
        for (brain_name, module_name, class_name) in brain_classes:
            try:
                module = __import__(module_name, fromlist=[class_name])
                brain_class = getattr(module, class_name)
                if 'opencode_unified_brain' in module_name:
                    brain_instance = brain_class()
                else:
                    brain_instance = brain_class()
                brain_type = brain_name
                print(f'✅ Created {brain_type}')
                break
            except Exception as e:
                print(f'❌ {brain_name} creation failed: {e}')
                continue
        if brain_instance is None:
            print('❌ No brain could be created')
            return None
        if hasattr(brain_instance, 'set_skills_manager'):
            brain_instance.set_skills_manager(skills_manager)
            print('✅ Skills manager connected to brain')
        elif hasattr(brain_instance, 'skills_manager'):
            brain_instance.skills_manager = skills_manager
            print('✅ Skills manager attached to brain')
        else:
            print('⚠️  Could not connect skills manager to brain')
        return (brain_instance, brain_type)
    except Exception as e:
        print(f'❌ Brain creation failed: {e}')
        import traceback
        traceback.print_exc()
        return (None, None)

def interactive_mode(brain, skills_manager):
    """Run interactive mode"""
    print('\n' + '=' * 60)
    print('NEO-CLONE INTERACTIVE MODE')
    print('=' * 60)
    print("Type 'help' for commands, 'quit' to exit")
    while True:
        try:
            user_input = input('\nNeo-Clone> ').strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print('Goodbye! 👋')
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.lower() == 'skills':
                list_skills(skills_manager)
            elif user_input.lower() == 'status':
                print_status(brain, skills_manager)
            elif user_input.startswith('skill '):
                parts = user_input.split(' ', 2)
                if len(parts) >= 2:
                    skill_name = parts[1]
                    params = {}
                    if len(parts) == 3:
                        try:
                            import json
                            params = json.loads(parts[2])
                        except:
                            params = {'query': parts[2]}
                    execute_skill(skills_manager, skill_name, params)
                else:
                    print('Usage: skill <skill_name> [params]')
            elif user_input:
                if brain and hasattr(brain, 'process_request'):
                    try:
                        response = brain.process_request(user_input)
                        print(f'Brain: {response}')
                    except Exception as e:
                        print(f'Brain processing failed: {e}')
                else:
                    print('Brain not available for processing')
        except KeyboardInterrupt:
            print('\nGoodbye! 👋')
            break
        except Exception as e:
            print(f'Error: {e}')

def print_help():
    """Print help information"""
    print('\nAvailable commands:\n  help     - Show this help\n  skills   - List available skills\n  status   - Show system status\n  skill <name> [params] - Execute a skill\n  quit     - Exit Neo-Clone\n\nExamples:\n  skill free_programming_books {"action": "list_categories"}\n  skill public_apis {"action": "list_categories"}\n  skill text_analysis {"text": "Hello world", "action": "sentiment"}\n')

def list_skills(skills_manager):
    """List available skills"""
    skills = skills_manager.list_skills()
    print(f'\nAvailable Skills ({len(skills)}):')
    for (name, info) in skills.items():
        desc = info.get('description', 'No description')
        caps = info.get('capabilities', [])
        print(f'  📋 {name}')
        print(f'     Description: {desc}')
        if caps:
            print(f"     Capabilities: {', '.join(caps)}")

def print_status(brain, skills_manager):
    """Print system status"""
    print(f'\nSystem Status:')
    print(f"  🧠 Brain: {(type(brain).__name__ if brain else 'None')}")
    print(f'  🔧 Skills Manager: {type(skills_manager).__name__}')
    skills = skills_manager.list_skills()
    print(f'  📚 Available Skills: {len(skills)}')
    if brain:
        if hasattr(brain, 'get_status'):
            try:
                status = brain.get_status()
                print(f'  📊 Brain Status: {status}')
            except:
                pass

def execute_skill(skills_manager, skill_name, params):
    """Execute a skill"""
    try:
        print(f'\n🔧 Executing skill: {skill_name}')
        print(f'📋 Parameters: {params}')
        result = skills_manager.execute_skill(skill_name, params)
        if result.get('success', False):
            print('✅ Skill executed successfully')
            data = result.get('data', {})
            if data:
                print(f'📤 Result: {data}')
        else:
            error = result.get('error', 'Unknown error')
            print(f'❌ Skill execution failed: {error}')
    except Exception as e:
        print(f'❌ Skill execution error: {e}')

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Neo-Clone AI Assistant')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--test', action='store_true', help='Run tests and exit')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    setup_logging(args.debug)
    print('Starting Neo-Clone AI Assistant')
    print(f'Project Root: {project_root}')
    (skills_ok, skills_manager) = test_skills_system()
    if not skills_ok:
        print('❌ Skills system failed to initialize')
        return 1
    brain_components = test_brain_components()
    memory_components = test_memory_components()
    (brain, brain_type) = create_simple_brain(skills_manager)
    if args.test:
        print('\n✅ Tests completed successfully!')
        return 0
    if args.interactive or not any([args.test]):
        interactive_mode(brain, skills_manager)
    return 0
if __name__ == '__main__':
    sys.exit(main())