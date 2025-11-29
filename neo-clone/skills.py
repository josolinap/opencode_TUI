#!/usr/bin/env python3
"""
Skills Module - Bridge to OpenCode Skills Manager

This module provides the interface that the brain system expects
while delegating to the actual skills manager.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path to import skills
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try multiple import paths
SkillsManager = None
OpenCodeSkillsManager = None

try:
    # Try from skills directory first
    sys.path.insert(0, str(project_root / "skills"))
    from opencode_skills_manager import OpenCodeSkillsManager
    SkillsManager = OpenCodeSkillsManager
    print("SUCCESS: Loaded SkillsManager from skills directory")
except ImportError as e1:
    print(f"Warning: Could not import from skills directory: {e1}")
    try:
        # Try direct import with full path
        skills_file = project_root / "skills" / "opencode_skills_manager.py"
        if skills_file.exists():
            spec = __import__('importlib.util').util.spec_from_file_location("opencode_skills_manager", skills_file)
            module = __import__('importlib.util').util.module_from_spec(spec)
            sys.modules["opencode_skills_manager"] = module
            spec.loader.exec_module(module)
            OpenCodeSkillsManager = module.OpenCodeSkillsManager
            SkillsManager = OpenCodeSkillsManager
            print("SUCCESS: Loaded SkillsManager using direct file import")
    except Exception as e2:
        print(f"Warning: Could not import using direct file import: {e2}")
        # Fallback dummy class
        class SkillsManager:
            def __init__(self):
                self.skills = {}
                self.initialized = False
                print("Initialized fallback SkillsManager")
            
            def initialize(self):
                self.initialized = True
                print("Fallback: SkillsManager initialized")
                return True
            
            def get_skill(self, name):
                return None
            
            def list_skills(self):
                return []
            
            def initialize_skills(self):
                print("Fallback: initialize_skills called")
                return True
        
        OpenCodeSkillsManager = SkillsManager
        SkillsManager = OpenCodeSkillsManager
        print("Using fallback SkillsManager")

# Export the expected interface
__all__ = ['OpenCodeSkillsManager', 'SkillsManager']