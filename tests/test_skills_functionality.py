from functools import lru_cache
'\nTest Skills Functionality\nTests the actual execution of loaded skills\n'
import sys
import os
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@lru_cache(maxsize=128)
def test_skills():
    """Test skill functionality"""
    print('=' * 60)
    print('TESTING SKILLS FUNCTIONALITY')
    print('=' * 60)
    try:
        from skills.opencode_skills_manager import OpenCodeSkillsManager
        manager = OpenCodeSkillsManager()
        manager.initialize()
        skill_names = manager.list_skills()
        print(f'Found {len(skill_names)} skills')
        test_results = {}
        for skill_name in skill_names:
            print(f'\n--- Testing {skill_name} ---')
            skill_info = manager.get_skill_info(skill_name)
            if not skill_info:
                print(f'  [FAIL] Could not get skill info for {skill_name}')
                test_results[skill_name] = 'FAILED - No skill info'
                continue
            print(f'  Description: {skill_info.description}')
            try:
                test_params = get_test_params(skill_name)
                print(f'  Testing with params: {test_params}')
                if skill_name == 'enhanced_tool':
                    result = manager.execute_skill(skill_name, None, **test_params)
                else:
                    result = manager.execute_skill(skill_name, test_params)
                if hasattr(result, 'success'):
                    if result.success:
                        print(f'  [SUCCESS]')
                        if hasattr(result, 'output') and result.output:
                            output_str = str(result.output)
                            if len(output_str) > 200:
                                print(f'  Result: {output_str[:200]}...')
                            else:
                                print(f'  Result: {output_str}')
                        else:
                            print(f'  Result: {result}')
                        test_results[skill_name] = 'SUCCESS'
                    else:
                        error = getattr(result, 'error_message', str(result))
                        print(f'  [FAILED]: {error}')
                        test_results[skill_name] = f'FAILED - {error}'
                elif isinstance(result, dict):
                    if result.get('success', False):
                        print(f'  [SUCCESS]')
                        data = result.get('data', {})
                        if isinstance(data, str) and len(data) > 200:
                            print(f'  Result: {data[:200]}...')
                        else:
                            print(f'  Result: {data}')
                        test_results[skill_name] = 'SUCCESS'
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f'  [FAILED]: {error}')
                        test_results[skill_name] = f'FAILED - {error}'
                else:
                    print(f'  [UNKNOWN FORMAT]: {result}')
                    test_results[skill_name] = 'UNKNOWN FORMAT'
            except Exception as e:
                print(f'  [EXCEPTION]: {e}')
                test_results[skill_name] = f'EXCEPTION - {e}'
        print('\n' + '=' * 60)
        print('SKILLS TEST SUMMARY')
        print('=' * 60)
        success_count = sum((1 for result in test_results.values() if result == 'SUCCESS'))
        total_count = len(test_results)
        print(f'Total Skills: {total_count}')
        print(f'Successful: {success_count}')
        print(f'Failed: {total_count - success_count}')
        print(f'Success Rate: {success_count / total_count * 100:.1f}%')
        print('\nDetailed Results:')
        for (skill_name, result) in test_results.items():
            status = '[OK]' if result == 'SUCCESS' else '[FAIL]'
            print(f'  {status} {skill_name}: {result}')
        return (success_count, total_count)
    except Exception as e:
        print(f'Skills test failed: {e}')
        import traceback
        traceback.print_exc()
        return (0, 0)

def get_test_params(skill_name):
    """Get test parameters for a skill"""
    test_params_map = {'freeprogrammingbooks': {'action': 'list_categories'}, 'publicapis': {'action': 'list_categories'}, 'codegeneration': {'prompt': 'Write a hello world function in Python'}, 'datainspector': {'data': {'test': [1, 2, 3]}, 'action': 'analyze'}, 'filemanager': {'text': 'list files in current directory'}, 'websearch': {'query': 'Python programming', 'max_results': 3}, 'textanalysis': {'text': 'This is a test sentence for analysis.', 'action': 'sentiment'}, 'advanced_memory': {'action': 'store', 'key': 'test', 'value': 'test_value'}, 'multisession': {'action': 'list_sessions'}, 'openspec': {'action': 'help'}, 'systemhealer': {'action': 'diagnose'}, 'enhanced_tool': {'tool_name': 'list_tools', 'tool_params': {}}, 'enhancedtool': {'action': 'status'}, 'testmultisession': {'action': 'test'}, 'TONL Encoder/Decoder': {'action': 'encode', 'data': 'test'}}
    return test_params_map.get(skill_name, {'action': 'test'})
if __name__ == '__main__':
    (success_count, total_count) = test_skills()
    if success_count > 0:
        print(f'\n[SUCCESS] Skills functionality test completed!')
        print(f'   {success_count}/{total_count} skills working correctly')
    else:
        print(f'\n[FAILED] No skills are working correctly')
    sys.exit(0 if success_count > 0 else 1)