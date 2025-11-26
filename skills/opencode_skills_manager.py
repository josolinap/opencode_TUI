#!/usr/bin/env python3
"""
OpenCode Skills Manager

Central management system for all Neo-Clone skills with registration,
discovery, and execution capabilities.
"""

import logging
import importlib
import sys
import time
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """Information about a registered skill"""
    name: str
    class_obj: Type
    instance: Any
    category: str
    description: str
    capabilities: List[str]
    parameters: Dict[str, Any]
    examples: List[str]
    registration_time: float
    execution_count: int = 0
    success_count: int = 0
    average_execution_time: float = 0.0


class OpenCodeSkillsManager:
    """Central manager for Neo-Clone skills"""
    
    def __init__(self):
        self.skills: Dict[str, SkillInfo] = {}
        self.categories: Dict[str, List[str]] = {}
        self.skills_path = Path(__file__).parent
        self.initialized = False
        
        # Performance tracking
        self.total_executions = 0
        self.total_successes = 0
        self.total_failures = 0
        
    def initialize(self) -> bool:
        """Initialize the skills manager and discover skills"""
        if self.initialized:
            return True
            
        try:
            # Discover and register skills
            self._discover_skills()
            self._build_category_index()
            self.initialized = True
            
            logger.info(f"Skills Manager initialized with {len(self.skills)} skills")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize skills manager: {e}")
            return False
    
    def _discover_skills(self):
        """Discover and register all available skills"""
        # Add neo-clone directory to path for imports
        neo_clone_path = self.skills_path.parent / "neo-clone"
        if neo_clone_path.exists():
            sys.path.insert(0, str(neo_clone_path))
        
        # Discover skills from multiple sources
        self._discover_from_directory(self.skills_path)
        self._discover_from_directory(neo_clone_path)
        
        # Register built-in skills
        self._register_builtin_skills()
    
    def _discover_from_directory(self, directory: Path):
        """Discover skills from a directory"""
        if not directory.exists():
            return
            
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name == "__init__.py":
                continue
                
            try:
                self._register_skill_from_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to register skill from {file_path}: {e}")
    
    def _register_skill_from_file(self, file_path: Path):
        """Register a skill from a Python file"""
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None or spec.loader is None:
            return
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for skill classes
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Check if it's a skill class
            if (isinstance(attr, type) and 
                attr_name.endswith("Skill") and 
                hasattr(attr, '__bases__')):
                
                try:
                    self._register_skill_class(attr, module_name)
                except Exception as e:
                    logger.warning(f"Failed to register skill class {attr_name}: {e}")
    
    def _register_skill_class(self, skill_class: Type, module_name: str):
        """Register a skill class"""
        try:
            # Create skill instance
            skill_instance = skill_class()
            
            # Get skill metadata
            metadata = getattr(skill_instance, 'metadata', None)
            if metadata is None:
                # Create basic metadata
                metadata = type('Metadata', (), {
                    'name': skill_class.__name__.lower().replace('skill', ''),
                    'category': 'general',
                    'description': skill_class.__doc__ or 'No description available',
                    'capabilities': [],
                    'parameters': {},
                    'examples': []
                })()
            
            # Create skill info
            skill_info = SkillInfo(
                name=metadata.name or skill_class.__name__.lower().replace('skill', ''),
                class_obj=skill_class,
                instance=skill_instance,
                category=getattr(metadata, 'category', 'general'),
                description=getattr(metadata, 'description', 'No description'),
                capabilities=getattr(metadata, 'capabilities', []),
                parameters=getattr(metadata, 'parameters', {}),
                examples=getattr(metadata, 'examples', []),
                registration_time=time.time()
            )
            
            # Register skill
            self.skills[skill_info.name] = skill_info
            
            logger.info(f"Registered skill: {skill_info.name} in category {skill_info.category}")
            
        except Exception as e:
            logger.error(f"Failed to register skill class {skill_class.__name__}: {e}")
            raise
    
    def _register_builtin_skills(self):
        """Register built-in skills that might not be in files"""
        # Try to register additional skills from neo-clone
        try:
            from additional_skills import (
                PlanningSkill, WebSearchSkill, MLTrainingSkill,
                FileManagerSkill, TextAnalysisSkill, DataInspectorSkill
            )
            
            builtin_skills = [
                PlanningSkill, WebSearchSkill, MLTrainingSkill,
                FileManagerSkill, TextAnalysisSkill, DataInspectorSkill
            ]
            
            for skill_class in builtin_skills:
                try:
                    self._register_skill_class(skill_class, "additional_skills")
                except Exception as e:
                    logger.warning(f"Failed to register builtin skill {skill_class.__name__}: {e}")
                    
        except ImportError:
            logger.debug("Additional skills module not available")
    
    def _build_category_index(self):
        """Build index of skills by category"""
        self.categories.clear()
        
        for skill_name, skill_info in self.skills.items():
            category = skill_info.category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(skill_name)
    
    def get_skill(self, skill_name: str) -> Optional[Any]:
        """Get a skill instance by name"""
        skill_info = self.skills.get(skill_name)
        if skill_info:
            skill_info.execution_count += 1
            return skill_info.instance
        return None
    
    def get_skill_info(self, skill_name: str) -> Optional[SkillInfo]:
        """Get detailed information about a skill"""
        return self.skills.get(skill_name)
    
    def list_skills(self, category: Optional[str] = None) -> List[str]:
        """List all skills, optionally filtered by category"""
        if category:
            return self.categories.get(category, [])
        return list(self.skills.keys())
    
    def list_categories(self) -> List[str]:
        """List all skill categories"""
        return list(self.categories.keys())
    
    def search_skills(self, query: str) -> List[SkillInfo]:
        """Search for skills by name, description, or capabilities"""
        query = query.lower()
        results = []
        
        for skill_info in self.skills.values():
            # Search in name
            if query in skill_info.name.lower():
                results.append(skill_info)
                continue
            
            # Search in description
            if query in skill_info.description.lower():
                results.append(skill_info)
                continue
            
            # Search in capabilities
            for capability in skill_info.capabilities:
                if query in capability.lower():
                    results.append(skill_info)
                    break
        
        return results
    
    def execute_skill(self, skill_name: str, context: Any, **kwargs) -> Any:
        """Execute a skill by name"""
        skill_info = self.skills.get(skill_name)
        if not skill_info:
            raise ValueError(f"Skill '{skill_name}' not found")
        
        start_time = time.time()
        
        try:
            # Execute the skill
            if hasattr(skill_info.instance, '_execute_async'):
                # Async skill
                import asyncio
                result = asyncio.run(skill_info.instance._execute_async(context, **kwargs))
            elif hasattr(skill_info.instance, 'execute'):
                # Sync skill
                result = skill_info.instance.execute(context, **kwargs)
            else:
                raise ValueError(f"Skill '{skill_name}' has no execute method")
            
            # Update statistics
            execution_time = time.time() - start_time
            skill_info.execution_count += 1
            skill_info.average_execution_time = (
                (skill_info.average_execution_time * (skill_info.execution_count - 1) + execution_time) /
                skill_info.execution_count
            )
            
            # Check if execution was successful
            success = getattr(result, 'success', True)
            if success:
                skill_info.success_count += 1
                self.total_successes += 1
            else:
                self.total_failures += 1
            
            self.total_executions += 1
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            skill_info.execution_count += 1
            skill_info.average_execution_time = (
                (skill_info.average_execution_time * (skill_info.execution_count - 1) + execution_time) /
                skill_info.execution_count
            )
            
            self.total_failures += 1
            self.total_executions += 1
            
            logger.error(f"Skill execution failed: {skill_name}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_skills": len(self.skills),
            "total_categories": len(self.categories),
            "total_executions": self.total_executions,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": (self.total_successes / self.total_executions * 100) if self.total_executions > 0 else 0,
            "categories": {
                category: len(skills) 
                for category, skills in self.categories.items()
            },
            "top_skills": sorted(
                [(name, info.execution_count) for name, info in self.skills.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def get_skill_details(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific skill"""
        skill_info = self.skills.get(skill_name)
        if not skill_info:
            return None
        
        return {
            "name": skill_info.name,
            "category": skill_info.category,
            "description": skill_info.description,
            "capabilities": skill_info.capabilities,
            "parameters": skill_info.parameters,
            "examples": skill_info.examples,
            "statistics": {
                "execution_count": skill_info.execution_count,
                "success_count": skill_info.success_count,
                "average_execution_time": skill_info.average_execution_time,
                "success_rate": (skill_info.success_count / skill_info.execution_count * 100) if skill_info.execution_count > 0 else 0
            },
            "registration_time": skill_info.registration_time
        }
    
    def reload_skill(self, skill_name: str) -> bool:
        """Reload a skill"""
        skill_info = self.skills.get(skill_name)
        if not skill_info:
            return False
        
        try:
            # Create new instance
            new_instance = skill_info.class_obj()
            skill_info.instance = new_instance
            
            logger.info(f"Reloaded skill: {skill_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload skill {skill_name}: {e}")
            return False
    
    def unregister_skill(self, skill_name: str) -> bool:
        """Unregister a skill"""
        if skill_name in self.skills:
            del self.skills[skill_name]
            self._build_category_index()
            logger.info(f"Unregistered skill: {skill_name}")
            return True
        return False


# Global skills manager instance
_skills_manager = None


def get_skills_manager() -> OpenCodeSkillsManager:
    """Get the global skills manager instance"""
    global _skills_manager
    if _skills_manager is None:
        _skills_manager = OpenCodeSkillsManager()
        _skills_manager.initialize()
    return _skills_manager


def register_skill(skill_class: Type, module_name: str = "unknown") -> bool:
    """Register a skill class with the global manager"""
    try:
        manager = get_skills_manager()
        manager._register_skill_class(skill_class, module_name)
        manager._build_category_index()
        return True
    except Exception as e:
        logger.error(f"Failed to register skill: {e}")
        return False


if __name__ == "__main__":
    # Test the skills manager
    manager = OpenCodeSkillsManager()
    manager.initialize()
    
    print("Skills Manager Test")
    print("=" * 40)
    print(f"Total Skills: {len(manager.skills)}")
    print(f"Categories: {list(manager.categories.keys())}")
    
    for category, skills in manager.categories.items():
        print(f"\n{category.upper()}:")
        for skill in skills:
            print(f"  - {skill}")
    
    print("\nStatistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")