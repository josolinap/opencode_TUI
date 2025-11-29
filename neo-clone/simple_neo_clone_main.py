#!/usr/bin/env python3
"""
Simple Neo-Clone entry point for Opencode integration
Uses the fixed skills system
"""

import sys
import os
import json
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'skills'))
sys.path.insert(0, os.path.dirname(__file__))

from opencode_skills_manager import OpenCodeSkillsManager

def main():
    """Simple Neo-Clone main function"""
    if len(sys.argv) < 2:
        print("Usage: python simple_neo_clone_main.py <message>")
        return 1
    
    message = ' '.join(sys.argv[1:])
    
    try:
        # Initialize skills manager
        manager = OpenCodeSkillsManager()
        if not manager.initialize():
            print("Failed to initialize skills manager")
            return 1
        
        print(f"Neo-Clone Skills Manager initialized with {len(manager.skills)} skills")
        
        # Simple intent detection and skill selection
        message_lower = message.lower()
        
        # Determine which skill to use
        if any(word in message_lower for word in ['code', 'generate', 'create', 'write', 'function', 'class']):
            skill_name = 'code_generation'
            params = {'prompt': message, 'language': 'python'}
        elif any(word in message_lower for word in ['analyze', 'inspect', 'data', 'file', 'csv', 'json']):
            skill_name = 'data_inspector'
            params = {'prompt': message}
        elif any(word in message_lower for word in ['search', 'web', 'find', 'look up']):
            skill_name = 'web_search'
            params = {'query': message}
        elif any(word in message_lower for word in ['text', 'sentiment', 'analyze text']):
            skill_name = 'text_analysis'
            params = {'text': message}
        elif any(word in message_lower for word in ['file', 'read', 'directory']):
            skill_name = 'file_manager'
            params = {'action': 'info', 'path': '.'}
        else:
            # Default to code generation for general requests
            skill_name = 'code_generation'
            params = {'prompt': message, 'language': 'python'}
        
        # Execute the selected skill
        if skill_name in manager.skills:
            print(f"Executing skill: {skill_name}")
            result = manager.execute_skill(skill_name, params)
            
            if result.success:
                print(f"[SUCCESS] {result.message}")
                if result.data:
                    print(f"[DATA] {json.dumps(result.data, indent=2)}")
            else:
                print(f"[ERROR] {result.message}")
        else:
            print(f"[ERROR] Skill '{skill_name}' not found")
            print(f"Available skills: {list(manager.skills.keys())}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Neo-Clone execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())