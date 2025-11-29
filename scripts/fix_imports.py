from functools import lru_cache
'\nFix import paths after workspace reorganization\n'
import os
import re
from pathlib import Path

@lru_cache(maxsize=128)
def fix_imports():
    """Fix import paths in affected files"""
    files_to_fix = ['src/debug_skills.py', 'src/test_skills_manager.py', 'scripts/neo_clone_fixed_demo.py', 'tests/simple/simple_neo_clone_demo.py', 'src/check_models_simple.py']
    root_dir = Path.cwd()
    for file_path in files_to_fix:
        full_path = root_dir / file_path
        if full_path.exists():
            print(f'Fixing imports in {file_path}...')
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                original_content = content
                content = re.sub('from skills\\.([^ ]+) import', 'from skills.\\1 import', content)
                if 'from skills.' in content and 'import sys' not in content:
                    lines = content.split('\n')
                    insert_idx = 0
                    for (i, line) in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            insert_idx = i
                            break
                    sys_path_lines = ['import sys', 'from pathlib import Path', '', '# Add project root to Python path for skills imports', 'project_root = Path(__file__).parent.parent', 'sys.path.insert(0, str(project_root))', '']
                    lines[insert_idx:insert_idx] = sys_path_lines
                    content = '\n'.join(lines)
                content = re.sub('from neo-clone\\.([^ ]+) import', 'from neo_clone.\\1 import', content)
                if content != original_content:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f'  ✅ Fixed imports in {file_path}')
                else:
                    print(f'  ℹ️  No changes needed for {file_path}')
            except Exception as e:
                print(f'  ❌ Error fixing {file_path}: {e}')
        else:
            print(f'  ⚠️  File not found: {file_path}')
    skills_files = ['skills/free_programming_books.py', 'skills/public_apis.py']
    for file_path in skills_files:
        full_path = root_dir / file_path
        if full_path.exists():
            print(f'Fixing imports in {file_path}...')
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                original_content = content
                content = re.sub('from base_skill import', 'from .base_skill import', content)
                if content != original_content:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f'  ✅ Fixed imports in {file_path}')
                else:
                    print(f'  ℹ️  No changes needed for {file_path}')
            except Exception as e:
                print(f'  ❌ Error fixing {file_path}: {e}')
    print('\n✅ Import fixes completed!')
if __name__ == '__main__':
    fix_imports()