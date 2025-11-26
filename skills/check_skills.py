#!/usr/bin/env python3
"""
Skills Health Check Module

Provides functionality to check the health and status of all Neo-Clone skills.
"""

import logging
import importlib
import importlib.util
import sys
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillsHealthChecker:
    """Health checker for Neo-Clone skills"""
    
    def __init__(self):
        self.skills_path = Path(__file__).parent
        self.results = {}
        
    def check_all_skills(self) -> Dict[str, Any]:
        """Check health of all available skills"""
        results = {
            "total_skills": 0,
            "healthy_skills": 0,
            "unhealthy_skills": 0,
            "skill_details": {},
            "errors": []
        }
        
        # Get all Python files in skills directory
        skill_files = list(self.skills_path.glob("*.py"))
        skill_files = [f for f in skill_files if f.name != "__init__.py" and not f.name.startswith("_")]
        
        results["total_skills"] = len(skill_files)
        
        for skill_file in skill_files:
            skill_name = skill_file.stem
            try:
                health = self._check_skill_health(skill_file)
                results["skill_details"][skill_name] = health
                
                if health["status"] == "healthy":
                    results["healthy_skills"] += 1
                else:
                    results["unhealthy_skills"] += 1
                    
            except Exception as e:
                error_msg = f"Failed to check {skill_name}: {str(e)}"
                results["errors"].append(error_msg)
                results["skill_details"][skill_name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["unhealthy_skills"] += 1
        
        return results
    
    def _check_skill_health(self, skill_file: Path) -> Dict[str, Any]:
        """Check health of a specific skill file"""
        skill_name = skill_file.stem
        
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location(skill_name, skill_file)
            if spec is None or spec.loader is None:
                return {
                    "status": "unhealthy",
                    "error": "Could not create module spec"
                }
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required attributes
            health_info = {
                "status": "healthy",
                "has_classes": False,
                "has_functions": False,
                "classes": [],
                "functions": [],
                "imports": []
            }
            
            # Check for classes
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and name.endswith("Skill"):
                    health_info["has_classes"] = True
                    health_info["classes"].append(name)
            
            # Check for functions
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith("_"):
                    health_info["has_functions"] = True
                    health_info["functions"].append(name)
            
            # Check imports
            if hasattr(module, "__all__"):
                health_info["imports"] = module.__all__
            
            return health_info
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_skill_dependencies(self) -> Dict[str, Any]:
        """Check if all skill dependencies are available"""
        dependencies = {
            "skills": ["BaseSkill", "SkillParameter", "SkillParameterType", "SkillStatus"],
            "data_models": ["SkillResult", "SkillContext", "SkillMetadata", "SkillCategory"],
            "mcp_protocol": ["MCPClient", "MCPConfig", "MCPTool"]
        }
        
        results = {
            "all_dependencies_available": True,
            "missing_dependencies": [],
            "available_dependencies": {}
        }
        
        for module_name, required_imports in dependencies.items():
            module_results = {
                "available": [],
                "missing": []
            }
            
            try:
                module = importlib.import_module(module_name)
                for import_name in required_imports:
                    if hasattr(module, import_name):
                        module_results["available"].append(import_name)
                    else:
                        module_results["missing"].append(import_name)
                        results["missing_dependencies"].append(f"{module_name}.{import_name}")
                        results["all_dependencies_available"] = False
                        
            except ImportError:
                module_results["missing"] = required_imports
                results["missing_dependencies"].extend([f"{module_name}.{imp}" for imp in required_imports])
                results["all_dependencies_available"] = False
            
            results["available_dependencies"][module_name] = module_results
        
        return results
    
    def generate_health_report(self) -> str:
        """Generate a comprehensive health report"""
        skills_health = self.check_all_skills()
        deps_health = self.check_skill_dependencies()
        
        report = []
        report.append("=" * 60)
        report.append("NEO-CLONE SKILLS HEALTH REPORT")
        report.append("=" * 60)
        
        # Summary
        report.append(f"\nSUMMARY:")
        report.append(f"  Total Skills: {skills_health['total_skills']}")
        report.append(f"  Healthy Skills: {skills_health['healthy_skills']}")
        report.append(f"  Unhealthy Skills: {skills_health['unhealthy_skills']}")
        report.append(f"  Dependencies OK: {deps_health['all_dependencies_available']}")
        
        # Skill details
        report.append(f"\nSKILL DETAILS:")
        for skill_name, health in skills_health["skill_details"].items():
            status_icon = "[OK]" if health["status"] == "healthy" else "[FAIL]"
            report.append(f"  {status_icon} {skill_name}: {health['status']}")
            
            if health["status"] == "unhealthy" and "error" in health:
                report.append(f"    Error: {health['error']}")
            elif health["status"] == "healthy":
                if health.get("has_classes"):
                    report.append(f"    Classes: {', '.join(health['classes'])}")
                if health.get("has_functions"):
                    report.append(f"    Functions: {len(health['functions'])} available")
        
        # Dependencies
        if not deps_health["all_dependencies_available"]:
            report.append(f"\nMISSING DEPENDENCIES:")
            for dep in deps_health["missing_dependencies"]:
                report.append(f"  [MISSING] {dep}")
        
        # Overall status
        overall_healthy = (
            skills_health["unhealthy_skills"] == 0 and 
            deps_health["all_dependencies_available"]
        )
        
        status_icon = "HEALTHY" if overall_healthy else "NEEDS ATTENTION"
        report.append(f"\nOVERALL STATUS: {status_icon}")
        report.append("=" * 60)
        
        return "\n".join(report)


def check_skills_health() -> Dict[str, Any]:
    """Convenience function to check all skills health"""
    checker = SkillsHealthChecker()
    return checker.check_all_skills()


def generate_skills_report() -> str:
    """Convenience function to generate skills health report"""
    checker = SkillsHealthChecker()
    return checker.generate_health_report()


if __name__ == "__main__":
    # Run health check and print report
    print(generate_skills_report())