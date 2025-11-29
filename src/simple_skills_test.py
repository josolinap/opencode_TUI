#!/usr/bin/env python3
"""
Simple skills test
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), 'skills'))
sys.path.insert(0, os.path.join(os.getcwd(), 'neo-clone'))

def test_simple_skills():
    print("=" * 60)
    print("SIMPLE SKILLS TEST")
    print("=" * 60)
    
    # Test direct skill imports
    skills_to_test = [
        ("code_generation", "CodeGenerationSkill", "neo-clone/code_generation.py"),
        ("text_analysis", "TextAnalysisSkill", "neo-clone/text_analysis.py"),
        ("data_inspector", "DataInspectorSkill", "neo-clone/data_inspector.py"),
        ("web_search", "WebSearchSkill", "neo-clone/web_search.py"),
        ("file_manager", "FileManagerSkill", "neo-clone/file_manager.py"),
        ("free_programming_books", "FreeProgrammingBooksSkill", "skills/free_programming_books.py"),
        ("public_apis", "PublicApisSkill", "skills/public_apis.py")
    ]
    
    working_skills = {}
    
    for skill_name, class_name, file_path in skills_to_test:
        print(f"\nTesting {skill_name} from {file_path}:")
        
        try:
            # Import module
            path_parts = file_path.split('/')
            module_name = path_parts[-1].replace('.py', '')
            
            if len(path_parts) == 2:
                # neo-clone skills
                sys.path.insert(0, os.path.join(os.getcwd(), 'neo-clone'))
            else:
                # skills directory
                sys.path.insert(0, os.path.join(os.getcwd(), 'skills'))
            
            module = __import__(module_name)
            skill_class = getattr(module, class_name)
            
            # Create skill instance
            skill_instance = skill_class()
            
            print(f"  [OK] {class_name} imported and created")
            print(f"       Name: {skill_instance.name}")
            print(f"       Description: {skill_instance.description}")
            
            working_skills[skill_name] = skill_instance
            
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY: {len(working_skills)} working skills")
    
    for skill_name, skill_instance in working_skills.items():
        print(f"  - {skill_name}: {skill_instance.name}")
    
    # Test skill execution
    if 'code_generation' in working_skills:
        print(f"\nTesting code generation skill execution:")
        try:
            result = working_skills['code_generation'].execute({
                'prompt': 'Create a hello world function',
                'language': 'python'
            })
            print(f"  [OK] Execution successful: {result.success}")
            print(f"       Message: {result.message}")
        except Exception as e:
            print(f"  [ERROR] Execution failed: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_simple_skills()