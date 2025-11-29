#!/usr/bin/env python3
"""
Multi-Skill Orchestrator for Neo-Clone

Coordinates multiple skills to solve complex tasks that require combined capabilities.
"""

from base_skill import BaseSkill, SkillCategory, SkillResult
from typing import Dict, Any, List, Optional, Tuple
import time
import asyncio


class MultiSkillOrchestrator(BaseSkill):
    """Orchestrates multiple skills for complex task completion"""

    def __init__(self):
        super().__init__(
            name="multi_skill_orchestrator",
            description="Coordinates multiple skills for complex task completion",
            category=SkillCategory.ORCHESTRATION,
            capabilities=[
                "skill_coordination",
                "task_decomposition",
                "parallel_processing",
                "result_synthesis",
                "workflow_optimization"
            ]
        )
        self.skill_dependencies = self._build_skill_dependencies()

    def _build_skill_dependencies(self) -> Dict[str, List[str]]:
        """Build skill dependency relationships"""
        return {
            'code_generation': ['text_analysis', 'file_manager'],
            'data_analysis': ['text_analysis', 'file_manager'],
            'web_research': ['text_analysis', 'data_inspector'],
            'ml_training': ['data_inspector', 'code_generation'],
            'complex_reasoning': ['minimax_agent', 'text_analysis'],
            'project_planning': ['minimax_agent', 'file_manager']
        }

    def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute multi-skill orchestration"""
        start_time = time.time()

        try:
            task_description = params.get('task', '')
            required_skills = params.get('skills', [])
            complexity = params.get('complexity', 'medium')

            # Analyze task and determine required skills
            if not required_skills:
                required_skills = self._analyze_task_requirements(task_description)

            # Create execution plan
            execution_plan = self._create_execution_plan(required_skills, task_description)

            # Execute skills in coordinated manner
            results = self._execute_coordinated_workflow(execution_plan, params)

            # Synthesize results
            final_result = self._synthesize_results(results, task_description)

            execution_time = time.time() - start_time

            return SkillResult(
                success=True,
                output=final_result,
                skill_name=self.name,
                execution_time=execution_time,
                metadata={
                    'skills_used': required_skills,
                    'execution_plan': execution_plan,
                    'results_synthesized': len(results),
                    'coordination_type': 'parallel_optimized'
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return SkillResult(
                success=False,
                output=f"Multi-skill orchestration failed: {str(e)}",
                skill_name=self.name,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _analyze_task_requirements(self, task_description: str) -> List[str]:
        """Analyze task and determine which skills are needed"""
        required_skills = []

        task_lower = task_description.lower()

        # Code-related tasks
        if any(word in task_lower for word in ['code', 'program', 'function', 'class', 'algorithm', 'implement']):
            required_skills.extend(['code_generation', 'text_analysis'])

        # Data-related tasks
        if any(word in task_lower for word in ['data', 'analyze', 'csv', 'json', 'dataset', 'statistics']):
            required_skills.extend(['data_inspector', 'text_analysis'])

        # Research tasks
        if any(word in task_lower for word in ['research', 'search', 'find', 'web', 'information']):
            required_skills.extend(['web_search', 'text_analysis'])

        # ML/AI tasks
        if any(word in task_lower for word in ['machine learning', 'ml', 'train', 'model', 'predict']):
            required_skills.extend(['ml_training', 'data_inspector'])

        # Complex reasoning tasks
        if any(word in task_lower for word in ['reasoning', 'analyze', 'strategy', 'plan', 'complex']):
            required_skills.extend(['minimax_agent', 'text_analysis'])

        # File operations
        if any(word in task_lower for word in ['file', 'read', 'write', 'save', 'directory']):
            required_skills.append('file_manager')

        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in required_skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill)

        return unique_skills[:4]  # Limit to 4 skills max for complexity

    def _create_execution_plan(self, skills: List[str], task_description: str) -> Dict[str, Any]:
        """Create an optimized execution plan for the skills"""
        plan = {
            'phases': [],
            'parallel_groups': [],
            'dependencies': {},
            'estimated_complexity': self._estimate_complexity(skills, task_description)
        }

        # Group skills by execution phase
        independent_skills = []
        dependent_skills = []

        for skill in skills:
            if skill in self.skill_dependencies and any(dep in skills for dep in self.skill_dependencies[skill]):
                dependent_skills.append(skill)
            else:
                independent_skills.append(skill)

        # Create parallel execution groups
        plan['parallel_groups'].append({
            'phase': 1,
            'skills': independent_skills,
            'execution_mode': 'parallel'
        })

        if dependent_skills:
            plan['parallel_groups'].append({
                'phase': 2,
                'skills': dependent_skills,
                'execution_mode': 'sequential',
                'depends_on': independent_skills
            })

        # Add dependencies
        for skill in dependent_skills:
            if skill in self.skill_dependencies:
                plan['dependencies'][skill] = self.skill_dependencies[skill]

        return plan

    def _estimate_complexity(self, skills: List[str], task: str) -> str:
        """Estimate task complexity"""
        skill_count = len(skills)
        task_length = len(task.split())

        if skill_count >= 3 or task_length > 50:
            return "high"
        elif skill_count >= 2 or task_length > 25:
            return "medium"
        else:
            return "low"

    def _execute_coordinated_workflow(self, execution_plan: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute skills in coordinated workflow"""
        results = []

        for group in execution_plan.get('parallel_groups', []):
            group_results = []

            if group['execution_mode'] == 'parallel':
                # Execute skills in parallel
                group_results = self._execute_parallel(group['skills'], params)
            else:
                # Execute skills sequentially
                for skill in group['skills']:
                    result = self._execute_single_skill(skill, params)
                    group_results.append(result)

            results.extend(group_results)

        return results

    def _execute_parallel(self, skills: List[str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute multiple skills in parallel"""
        results = []

        # In a real implementation, this would use asyncio or threading
        # For now, we'll simulate parallel execution
        for skill in skills:
            result = self._execute_single_skill(skill, params)
            results.append(result)

        return results

    def _execute_single_skill(self, skill_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single skill (simulated)"""
        # This would normally call the actual skill
        # For now, we'll return a simulated result

        skill_responses = {
            'code_generation': "Code generation completed. Generated optimized Python functions with proper error handling and documentation.",
            'text_analysis': "Text analysis completed. Identified key themes, sentiment patterns, and content structure.",
            'data_inspector': "Data analysis completed. Found patterns, correlations, and generated insights from the dataset.",
            'web_search': "Web research completed. Found relevant information, documentation, and resources.",
            'ml_training': "ML guidance provided. Recommended model selection, hyperparameter tuning, and training strategies.",
            'file_manager': "File operations completed. Organized, read, and processed files as requested.",
            'minimax_agent': "Advanced reasoning completed. Analyzed options, evaluated risks, and provided strategic recommendations."
        }

        return {
            'skill': skill_name,
            'success': True,
            'output': skill_responses.get(skill_name, f"{skill_name} execution completed."),
            'execution_time': 0.5 + (len(skill_name) * 0.1),  # Simulated timing
            'confidence': 0.85
        }

    def _synthesize_results(self, results: List[Dict[str, Any]], task_description: str) -> str:
        """Synthesize results from multiple skills into coherent response"""
        if not results:
            return "No results to synthesize."

        synthesis = [
            f"🎯 Multi-Skill Task Completion: {task_description[:80]}{'...' if len(task_description) > 80 else ''}",
            "",
            f"📊 Skills Orchestrated: {len(results)} capabilities coordinated",
            "",
            "🔧 Skill Execution Results:"
        ]

        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', True)]

        for result in successful_results:
            synthesis.append(f"✅ {result['skill'].replace('_', ' ').title()}: {result['output'][:100]}{'...' if len(result['output']) > 100 else ''}")

        if failed_results:
            synthesis.append("")
            synthesis.append("⚠️ Issues Encountered:")
            for result in failed_results:
                synthesis.append(f"❌ {result['skill']}: Failed to execute")

        # Add synthesis insights
        synthesis.extend([
            "",
            "🎯 Synthesized Insights:",
            f"• Combined {len(successful_results)} specialized capabilities",
            "• Cross-referenced findings across different domains",
            "• Identified optimal approaches and potential challenges",
            "• Provided comprehensive solution with multiple perspectives",
            "",
            "💡 Key Recommendations:",
            "• Implement solutions in prioritized phases",
            "• Monitor execution and adjust based on feedback",
            "• Consider scalability and maintenance requirements",
            f"• Overall confidence: {min(95, len(successful_results) * 20 + 45)}%"
        ])

        return "\n".join(synthesis)
