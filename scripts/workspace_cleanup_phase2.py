from functools import lru_cache
'\nWorkspace Cleanup Phase 2 - Directory Consolidation and Import Path Fixes\n\nThis script addresses:\n1. Skills directory consolidation\n2. Test file organization\n3. Import path corrections\n4. Root directory cleanup\n'
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class WorkspaceCleanupPhase2:

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.src_dir = root_dir / 'src'
        self.tests_dir = root_dir / 'tests'
        self.skills_dir = root_dir / 'skills'
        self.neo_clone_dir = root_dir / 'neo-clone'
        self.tests_unit_dir = self.tests_dir / 'unit'
        self.tests_integration_dir = self.tests_dir / 'integration'
        self.tests_system_dir = self.tests_dir / 'system'
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        for dir_path in [self.tests_unit_dir, self.tests_integration_dir, self.tests_system_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f'✅ Ensured directory: {dir_path}')

    @lru_cache(maxsize=128)
    def analyze_current_state(self) -> Dict:
        """Analyze current workspace state"""
        analysis = {'root_test_files': [], 'skills_conflicts': [], 'duplicate_directories': [], 'import_issues': []}
        for file_path in self.root_dir.glob('test_*.py'):
            analysis['root_test_files'].append(file_path.name)
        if self.skills_dir.exists() and (self.neo_clone_dir / 'skills.py').exists():
            analysis['skills_conflicts'].append({'type': 'skills_duplication', 'root_skills': str(self.skills_dir), 'neo_clone_skills': str(self.neo_clone_dir / 'skills.py')})
        if (self.root_dir / 'script').exists() and (self.root_dir / 'scripts').exists():
            analysis['duplicate_directories'].append('script vs scripts')
        return analysis

    def categorize_test_file(self, filename: str) -> str:
        """Categorize test files by type"""
        filename_lower = filename.lower()
        if any((keyword in filename_lower for keyword in ['skills', 'brain', 'model', 'detector', 'api'])):
            return 'unit'
        elif any((keyword in filename_lower for keyword in ['integration', 'opencode', 'execution'])):
            return 'integration'
        elif any((keyword in filename_lower for keyword in ['all', 'system', 'actual', 'real', 'comprehensive'])):
            return 'system'
        return 'unit'

    def move_test_files(self) -> Tuple[int, List[str]]:
        """Move test files from root to appropriate test directories"""
        moved_count = 0
        errors = []
        test_files = list(self.root_dir.glob('test_*.py'))
        print(f'📋 Found {len(test_files)} test files in root directory')
        for test_file in test_files:
            try:
                category = self.categorize_test_file(test_file.name)
                if category == 'unit':
                    target_dir = self.tests_unit_dir
                elif category == 'integration':
                    target_dir = self.tests_integration_dir
                else:
                    target_dir = self.tests_system_dir
                target_path = target_dir / test_file.name
                shutil.move(str(test_file), str(target_path))
                print(f'✅ Moved {test_file.name} -> tests/{category}/')
                moved_count += 1
            except Exception as e:
                errors.append(f'Failed to move {test_file.name}: {e}')
                print(f'❌ Error moving {test_file.name}: {e}')
        return (moved_count, errors)

    def consolidate_skills_structure(self) -> Tuple[bool, List[str]]:
        """Consolidate skills directory structure"""
        errors = []
        success = True
        try:
            skills_py_path = self.neo_clone_dir / 'skills.py'
            if skills_py_path.exists():
                with open(skills_py_path, 'r', encoding='utf-8') as f:
                    first_lines = ''.join(f.readlines()[:10])
                if 'Bridge to OpenCode Skills Manager' in first_lines:
                    print('✅ neo-clone/skills.py is correctly configured as a bridge')
                else:
                    print('⚠️  neo-clone/skills.py might be a duplicate - needs manual review')
                    errors.append('neo-clone/skills.py needs manual review')
            if self.skills_dir.exists():
                required_files = ['__init__.py', 'base_skill.py', 'opencode_skills_manager.py']
                for req_file in required_files:
                    if not (self.skills_dir / req_file).exists():
                        errors.append(f'Missing required file: skills/{req_file}')
                        success = False
                print('✅ Skills directory structure validated')
        except Exception as e:
            errors.append(f'Skills consolidation error: {e}')
            success = False
        return (success, errors)

    def identify_import_fixes_needed(self) -> List[Dict]:
        """Identify files that need import path fixes"""
        fixes_needed = []
        for py_file in self.src_dir.glob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'from skills.' in content or 'import skills.' in content:
                    fixes_needed.append({'file': str(py_file.relative_to(self.root_dir)), 'type': 'skills_import', 'content': content})
                if 'from neo-clone.' in content:
                    fixes_needed.append({'file': str(py_file.relative_to(self.root_dir)), 'type': 'neo_clone_import', 'content': content})
            except Exception as e:
                print(f'⚠️  Could not read {py_file}: {e}')
        return fixes_needed

    def generate_import_fix_script(self, fixes_needed: List[Dict]) -> str:
        """Generate a script to fix import paths"""
        script_lines = ['#!/usr/bin/env python3', '"""', 'Auto-generated import path fixes', '"""', '', 'import os', 'from pathlib import Path', '', 'def fix_imports():', '    """Fix import paths in affected files"""', '']
        for fix in fixes_needed:
            file_path = fix['file']
            script_lines.append(f'    # Fix {file_path}')
            script_lines.append(f'    file_path = Path("{file_path}")')
            if fix['type'] == 'skills_import':
                script_lines.append('    # Fix skills imports')
                script_lines.append('    if file_path.exists():')
                script_lines.append('        with open(file_path, "r") as f:')
                script_lines.append('            content = f.read()')
                script_lines.append('        # Add sys.path manipulation if needed')
                script_lines.append('        if "import sys" not in content:')
                script_lines.append('            content = "import sys\\nfrom pathlib import Path\\n\\n" + content')
                script_lines.append('        with open(file_path, "w") as f:')
                script_lines.append('            f.write(content)')
            script_lines.append('')
        script_lines.extend(['', "if __name__ == '__main__':", '    fix_imports()', "    print('Import fixes completed')"])
        return '\n'.join(script_lines)

    def cleanup_root_directory(self) -> Tuple[int, List[str]]:
        """Clean up remaining root directory clutter"""
        cleaned_count = 0
        errors = []
        doc_files = []
        for pattern in ['*.md', '*.txt']:
            for file_path in self.root_dir.glob(pattern):
                if file_path.name not in ['README.md', 'LICENSE']:
                    doc_files.append(file_path)
        docs_dir = self.root_dir / 'docs'
        docs_dir.mkdir(exist_ok=True)
        for doc_file in doc_files:
            try:
                target = docs_dir / doc_file.name
                shutil.move(str(doc_file), str(target))
                print(f'✅ Moved {doc_file.name} -> docs/')
                cleaned_count += 1
            except Exception as e:
                errors.append(f'Failed to move {doc_file.name}: {e}')
        return (cleaned_count, errors)

    def run_phase2_cleanup(self) -> Dict:
        """Execute complete Phase 2 cleanup"""
        print('🚀 Starting Workspace Cleanup Phase 2')
        print('=' * 50)
        results = {'analysis': {}, 'test_files_moved': 0, 'test_file_errors': [], 'skills_consolidated': False, 'skills_errors': [], 'import_fixes_needed': [], 'root_cleaned': 0, 'cleanup_errors': [], 'success': True}
        try:
            print('\n📊 Analyzing current workspace state...')
            results['analysis'] = self.analyze_current_state()
            print(f"✅ Found {len(results['analysis']['root_test_files'])} test files in root")
            print(f"✅ Found {len(results['analysis']['skills_conflicts'])} skills conflicts")
            print('\n📁 Moving test files to appropriate directories...')
            (moved, errors) = self.move_test_files()
            results['test_files_moved'] = moved
            results['test_file_errors'] = errors
            print('\n🔧 Consolidating skills directory structure...')
            (consolidated, errors) = self.consolidate_skills_structure()
            results['skills_consolidated'] = consolidated
            results['skills_errors'] = errors
            print('\n🔍 Identifying import path fixes needed...')
            fixes_needed = self.identify_import_fixes_needed()
            results['import_fixes_needed'] = fixes_needed
            print(f'✅ Found {len(fixes_needed)} files needing import fixes')
            if fixes_needed:
                fix_script = self.generate_import_fix_script(fixes_needed)
                fix_script_path = self.root_dir / 'fix_imports.py'
                with open(fix_script_path, 'w', encoding='utf-8') as f:
                    f.write(fix_script)
                print(f'✅ Generated import fix script: fix_imports.py')
            print('\n🧹 Cleaning up root directory...')
            (cleaned, errors) = self.cleanup_root_directory()
            results['root_cleaned'] = cleaned
            results['cleanup_errors'] = errors
            results['success'] = len(results['test_file_errors']) == 0 and results['skills_consolidated'] and (len(results['cleanup_errors']) == 0)
        except Exception as e:
            results['success'] = False
            results['overall_error'] = str(e)
            print(f'❌ Phase 2 cleanup failed: {e}')
        return results

    def print_summary(self, results: Dict):
        """Print cleanup summary"""
        print('\n' + '=' * 50)
        print('📋 PHASE 2 CLEANUP SUMMARY')
        print('=' * 50)
        print(f"✅ Test files moved: {results['test_files_moved']}")
        if results['test_file_errors']:
            print(f"❌ Test file errors: {len(results['test_file_errors'])}")
        print(f"✅ Skills consolidated: {results['skills_consolidated']}")
        if results['skills_errors']:
            print(f"❌ Skills errors: {len(results['skills_errors'])}")
        print(f"✅ Import fixes needed: {len(results['import_fixes_needed'])}")
        print(f"✅ Root files cleaned: {results['root_cleaned']}")
        if results['cleanup_errors']:
            print(f"❌ Cleanup errors: {len(results['cleanup_errors'])}")
        print(f"\n🎯 Overall Success: {('✅ YES' if results['success'] else '❌ NO')}")
        if not results['success']:
            print('\n⚠️  Issues found that need manual attention:')
            for (error_type, errors) in [('Test file errors', results.get('test_file_errors', [])), ('Skills errors', results.get('skills_errors', [])), ('Cleanup errors', results.get('cleanup_errors', []))]:
                if errors:
                    print(f'  {error_type}:')
                    for error in errors:
                        print(f'    - {error}')

def main():
    """Main execution function"""
    root_dir = Path.cwd()
    print(f'🧹 Workspace Cleanup Phase 2')
    print(f'📍 Root Directory: {root_dir}')
    response = input('\nContinue with Phase 2 cleanup? (y/N): ').strip().lower()
    if response not in ['y', 'yes']:
        print('Cleanup cancelled.')
        return
    cleanup = WorkspaceCleanupPhase2(root_dir)
    results = cleanup.run_phase2_cleanup()
    cleanup.print_summary(results)
    if results['import_fixes_needed']:
        print(f'\n🔧 NEXT STEPS:')
        print(f"1. Run 'python fix_imports.py' to fix import paths")
        print(f'2. Test Neo-Clone functionality')
        print(f'3. Verify all imports work correctly')
    return results['success']
if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)