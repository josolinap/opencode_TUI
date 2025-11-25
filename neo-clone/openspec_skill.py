"""
OpenSpec Skill for Neo-Clone
Integrates OpenSpec-NC workflow capabilities with Neo-Clone's skills framework
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Import OpenSpec engine
from openspec_neo_clone import (
    OpenSpecEngine, 
    Specification, 
    SpecChange, 
    SpecRequirement, 
    SpecDelta, 
    ChangeType,
    SpecStatus,
    ImplementationTask,
    TaskStatus
)


class OpenSpecSkill:
    """OpenSpec skill for Neo-Clone - provides spec-driven development capabilities"""
    
    def __init__(self):
        self.engine = OpenSpecEngine()
        self.name = "OpenSpec Skill"
        self.description = "Spec-driven development workflow for Neo-Clone"
        self.version = "1.0.0"
        
        # Skill capabilities
        self.capabilities = [
            "create_specification",
            "list_specifications", 
            "create_change",
            "apply_change",
            "generate_tasks",
            "validate_specification",
            "get_spec_stats",
            "export_specification",
            "import_specification"
        ]
    
    async def create_specification(self, 
                                 title: str,
                                 description: str,
                                 author: str = "",
                                 requirements: Optional[List[Dict[str, Any]]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new specification"""
        try:
            spec_id = f"spec-{uuid.uuid4().hex[:8]}"
            
            # Convert requirements to SpecRequirement objects
            req_objects = []
            if requirements:
                for req_data in requirements:
                    req = SpecRequirement(
                        id=req_data.get('id', f"{spec_id}-req-{len(req_objects)+1}"),
                        title=req_data.get('title', ''),
                        description=req_data.get('description', ''),
                        acceptance_criteria=req_data.get('acceptance_criteria', []),
                        priority=req_data.get('priority', 'medium'),
                        tags=req_data.get('tags', []),
                        scenarios=req_data.get('scenarios', [])
                    )
                    req_objects.append(req)
            
            spec = Specification(
                id=spec_id,
                title=title,
                description=description,
                author=author,
                requirements=req_objects,
                metadata=metadata or {}
            )
            
            # Validate specification
            errors = self.engine.validate_specification(spec)
            if errors:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "validation_errors": errors
                }
            
            # Create specification
            file_path = self.engine.create_specification(spec)
            
            return {
                "success": True,
                "specification_id": spec_id,
                "file_path": file_path,
                "specification": self._serialize_spec_for_response(spec)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_specifications(self, status: Optional[str] = None) -> Dict[str, Any]:
        """List all specifications, optionally filtered by status"""
        try:
            specs = self.engine.list_specifications()
            
            # Filter by status if provided
            if status:
                try:
                    status_enum = SpecStatus(status)
                    specs = [spec for spec in specs if spec.status == status_enum]
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Invalid status: {status}"
                    }
            
            # Serialize for response
            spec_list = []
            for spec in specs:
                spec_list.append({
                    "id": spec.id,
                    "title": spec.title,
                    "description": spec.description,
                    "version": spec.version,
                    "author": spec.author,
                    "status": spec.status.value,
                    "created_at": spec.created_at.isoformat(),
                    "updated_at": spec.updated_at.isoformat(),
                    "requirement_count": len(spec.requirements),
                    "metadata": spec.metadata
                })
            
            return {
                "success": True,
                "specifications": spec_list,
                "count": len(spec_list)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_change(self,
                          title: str,
                          description: str,
                          author: str = "",
                          deltas: Optional[List[Dict[str, Any]]] = None,
                          tasks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create a new change proposal"""
        try:
            change_id = f"change-{uuid.uuid4().hex[:8]}"
            
            # Convert deltas to SpecDelta objects
            delta_objects = []
            if deltas:
                for delta_data in deltas:
                    delta = SpecDelta(
                        type=ChangeType(delta_data['type']),
                        requirement_id=delta_data.get('requirement_id'),
                        requirement=self._deserialize_requirement(delta_data.get('requirement')),
                        old_requirement=self._deserialize_requirement(delta_data.get('old_requirement'))
                    )
                    delta_objects.append(delta)
            
            # Convert tasks to ImplementationTask objects
            task_objects = []
            if tasks:
                for task_data in tasks:
                    task = ImplementationTask(
                        id=task_data.get('id', f"task-{uuid.uuid4().hex[:8]}"),
                        title=task_data.get('title', ''),
                        description=task_data.get('description', ''),
                        requirement_ids=task_data.get('requirement_ids', []),
                        scenarios=task_data.get('scenarios', []),
                        status=TaskStatus(task_data.get('status', 'pending')),
                        dependencies=task_data.get('dependencies', []),
                        estimated_effort=task_data.get('estimated_effort')
                    )
                    task_objects.append(task)
            
            change = SpecChange(
                id=change_id,
                title=title,
                description=description,
                author=author,
                created_at=datetime.now(),
                deltas=delta_objects,
                tasks=task_objects
            )
            
            # Create change
            dir_path = self.engine.create_change(change)
            
            return {
                "success": True,
                "change_id": change_id,
                "directory_path": dir_path,
                "change": self._serialize_change_for_response(change)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def apply_change(self, change_id: str, author: str = "") -> Dict[str, Any]:
        """Apply a change to specifications"""
        try:
            success = self.engine.apply_change(change_id, author)
            
            if success:
                return {
                    "success": True,
                    "message": f"Change {change_id} applied successfully",
                    "change_id": change_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to apply change {change_id}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_tasks(self, spec_id: str) -> Dict[str, Any]:
        """Generate implementation tasks from specification"""
        try:
            tasks = self.engine.generate_tasks_from_spec(spec_id)
            
            # Serialize tasks for response
            task_list = []
            for task in tasks:
                task_list.append({
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "requirement_ids": task.requirement_ids,
                    "scenarios": task.scenarios,
                    "status": task.status.value,
                    "dependencies": task.dependencies,
                    "estimated_effort": task.estimated_effort
                })
            
            return {
                "success": True,
                "specification_id": spec_id,
                "tasks": task_list,
                "count": len(task_list)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def validate_specification(self, spec_id: str) -> Dict[str, Any]:
        """Validate a specification"""
        try:
            spec = self.engine.load_specification(spec_id)
            errors = self.engine.validate_specification(spec)
            
            return {
                "success": True,
                "specification_id": spec_id,
                "is_valid": len(errors) == 0,
                "validation_errors": errors,
                "error_count": len(errors)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_spec_stats(self, spec_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for specifications"""
        try:
            if spec_id:
                # Stats for specific spec
                spec = self.engine.load_specification(spec_id)
                stats = {
                    "specification_id": spec_id,
                    "title": spec.title,
                    "requirement_count": len(spec.requirements),
                    "status": spec.status.value,
                    "version": spec.version,
                    "created_at": spec.created_at.isoformat(),
                    "updated_at": spec.updated_at.isoformat(),
                    "priority_breakdown": self._get_priority_breakdown(spec.requirements),
                    "tag_breakdown": self._get_tag_breakdown(spec.requirements)
                }
            else:
                # Stats for all specs
                specs = self.engine.list_specifications()
                total_requirements = sum(len(spec.requirements) for spec in specs)
                status_breakdown = {}
                for spec in specs:
                    status = spec.status.value
                    status_breakdown[status] = status_breakdown.get(status, 0) + 1
                
                stats = {
                    "total_specifications": len(specs),
                    "total_requirements": total_requirements,
                    "status_breakdown": status_breakdown,
                    "average_requirements_per_spec": total_requirements / len(specs) if specs else 0
                }
            
            return {
                "success": True,
                "statistics": stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def export_specification(self, spec_id: str, format: str = "json") -> Dict[str, Any]:
        """Export specification in various formats"""
        try:
            spec = self.engine.load_specification(spec_id)
            
            if format.lower() == "json":
                content = json.dumps(self._serialize_spec_for_response(spec), indent=2)
                content_type = "application/json"
            elif format.lower() == "markdown":
                content = self._generate_spec_markdown(spec)
                content_type = "text/markdown"
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }
            
            return {
                "success": True,
                "specification_id": spec_id,
                "format": format,
                "content_type": content_type,
                "content": content
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def import_specification(self, content: str, format: str = "json") -> Dict[str, Any]:
        """Import specification from various formats"""
        try:
            if format.lower() == "json":
                spec_data = json.loads(content)
                spec = self._deserialize_spec_from_response(spec_data)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported import format: {format}"
                }
            
            # Validate imported spec
            errors = self.engine.validate_specification(spec)
            if errors:
                return {
                    "success": False,
                    "error": "Imported specification failed validation",
                    "validation_errors": errors
                }
            
            # Create specification
            file_path = self.engine.create_specification(spec)
            
            return {
                "success": True,
                "specification_id": spec.id,
                "file_path": file_path,
                "specification": self._serialize_spec_for_response(spec)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _serialize_spec_for_response(self, spec: Specification) -> Dict[str, Any]:
        """Serialize specification for API response"""
        return {
            "id": spec.id,
            "title": spec.title,
            "description": spec.description,
            "version": spec.version,
            "author": spec.author,
            "created_at": spec.created_at.isoformat(),
            "updated_at": spec.updated_at.isoformat(),
            "status": spec.status.value,
            "requirements": [
                {
                    "id": req.id,
                    "title": req.title,
                    "description": req.description,
                    "acceptance_criteria": req.acceptance_criteria,
                    "priority": req.priority,
                    "tags": req.tags,
                    "scenarios": req.scenarios
                }
                for req in spec.requirements
            ],
            "metadata": spec.metadata
        }
    
    def _serialize_change_for_response(self, change: SpecChange) -> Dict[str, Any]:
        """Serialize change for API response"""
        return {
            "id": change.id,
            "title": change.title,
            "description": change.description,
            "author": change.author,
            "created_at": change.created_at.isoformat(),
            "status": change.status.value,
            "deltas": [
                {
                    "type": delta.type.value,
                    "requirement_id": delta.requirement_id,
                    "requirement": self._serialize_requirement_for_response(delta.requirement) if delta.requirement else None,
                    "old_requirement": self._serialize_requirement_for_response(delta.old_requirement) if delta.old_requirement else None
                }
                for delta in change.deltas
            ],
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "requirement_ids": task.requirement_ids,
                    "scenarios": task.scenarios,
                    "status": task.status.value,
                    "dependencies": task.dependencies,
                    "estimated_effort": task.estimated_effort
                }
                for task in change.tasks
            ],
            "metadata": change.metadata
        }
    
    def _serialize_requirement_for_response(self, req: Optional[SpecRequirement]) -> Optional[Dict[str, Any]]:
        """Serialize requirement for API response"""
        if not req:
            return None
        return {
            "id": req.id,
            "title": req.title,
            "description": req.description,
            "acceptance_criteria": req.acceptance_criteria,
            "priority": req.priority,
            "tags": req.tags,
            "scenarios": req.scenarios
        }
    
    def _deserialize_requirement(self, req_data: Optional[Dict[str, Any]]) -> Optional[SpecRequirement]:
        """Deserialize requirement from dictionary"""
        if not req_data:
            return None
        return SpecRequirement(
            id=req_data["id"],
            title=req_data["title"],
            description=req_data["description"],
            acceptance_criteria=req_data.get("acceptance_criteria", []),
            priority=req_data.get("priority", "medium"),
            tags=req_data.get("tags", []),
            scenarios=req_data.get("scenarios", [])
        )
    
    def _deserialize_spec_from_response(self, spec_data: Dict[str, Any]) -> Specification:
        """Deserialize specification from API response format"""
        return Specification(
            id=spec_data["id"],
            title=spec_data["title"],
            description=spec_data["description"],
            version=spec_data.get("version", "1.0.0"),
            author=spec_data.get("author", ""),
            created_at=datetime.fromisoformat(spec_data["created_at"]),
            updated_at=datetime.fromisoformat(spec_data["updated_at"]),
            status=SpecStatus(spec_data.get("status", "draft")),
            requirements=[
                SpecRequirement(
                    id=req["id"],
                    title=req["title"],
                    description=req["description"],
                    acceptance_criteria=req.get("acceptance_criteria", []),
                    priority=req.get("priority", "medium"),
                    tags=req.get("tags", []),
                    scenarios=req.get("scenarios", [])
                )
                for req in spec_data.get("requirements", [])
            ],
            metadata=spec_data.get("metadata", {})
        )
    
    def _get_priority_breakdown(self, requirements: List[SpecRequirement]) -> Dict[str, int]:
        """Get breakdown of requirements by priority"""
        breakdown = {"high": 0, "medium": 0, "low": 0}
        for req in requirements:
            priority = req.priority.lower()
            if priority in breakdown:
                breakdown[priority] += 1
        return breakdown
    
    def _get_tag_breakdown(self, requirements: List[SpecRequirement]) -> Dict[str, int]:
        """Get breakdown of requirements by tags"""
        tag_counts = {}
        for req in requirements:
            for tag in req.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts
    
    def _generate_spec_markdown(self, spec: Specification) -> str:
        """Generate markdown representation of specification"""
        content = f"""# {spec.title}

**ID:** {spec.id}  
**Version:** {spec.version}  
**Author:** {spec.author}  
**Status:** {spec.status.value}  
**Created:** {spec.created_at.strftime('%Y-%m-%d %H:%M:%S')}  
**Updated:** {spec.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Description

{spec.description}

## Requirements

"""
        
        for req in spec.requirements:
            content += f"""### {req.title}

**ID:** {req.id}  
**Priority:** {req.priority}

{req.description}

"""
            if req.acceptance_criteria:
                content += "**Acceptance Criteria:**\n"
                for ac in req.acceptance_criteria:
                    content += f"- {ac}\n"
                content += "\n"
            
            if req.scenarios:
                content += "**Scenarios:**\n"
                for scenario in req.scenarios:
                    content += f"- {scenario}\n"
                content += "\n"
            
            if req.tags:
                content += f"**Tags:** {', '.join(req.tags)}\n\n"
        
        return content


# Neo-Clone skill interface functions
async def create_specification(title: str, description: str, author: str = "", requirements: Optional[List[Dict[str, Any]]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new specification"""
    skill = OpenSpecSkill()
    return await skill.create_specification(title, description, author, requirements, metadata)


async def list_specifications(status: Optional[str] = None) -> Dict[str, Any]:
    """List all specifications"""
    skill = OpenSpecSkill()
    return await skill.list_specifications(status)


async def create_change(title: str, description: str, author: str = "", deltas: Optional[List[Dict[str, Any]]] = None, tasks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Create a new change proposal"""
    skill = OpenSpecSkill()
    return await skill.create_change(title, description, author, deltas, tasks)


async def apply_change(change_id: str, author: str = "") -> Dict[str, Any]:
    """Apply a change to specifications"""
    skill = OpenSpecSkill()
    return await skill.apply_change(change_id, author)


async def generate_tasks(spec_id: str) -> Dict[str, Any]:
    """Generate implementation tasks from specification"""
    skill = OpenSpecSkill()
    return await skill.generate_tasks(spec_id)


async def validate_specification(spec_id: str) -> Dict[str, Any]:
    """Validate a specification"""
    skill = OpenSpecSkill()
    return await skill.validate_specification(spec_id)


async def get_spec_stats(spec_id: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics for specifications"""
    skill = OpenSpecSkill()
    return await skill.get_spec_stats(spec_id)


async def export_specification(spec_id: str, format: str = "json") -> Dict[str, Any]:
    """Export specification in various formats"""
    skill = OpenSpecSkill()
    return await skill.export_specification(spec_id, format)


async def import_specification(content: str, format: str = "json") -> Dict[str, Any]:
    """Import specification from various formats"""
    skill = OpenSpecSkill()
    return await skill.import_specification(content, format)


# Skill registration info
SKILL_INFO = {
    "name": "openspecskill",
    "description": "OpenSpec-NC: Spec-driven development workflow for Neo-Clone",
    "version": "1.0.0",
    "capabilities": [
        "create_specification",
        "list_specifications",
        "create_change", 
        "apply_change",
        "generate_tasks",
        "validate_specification",
        "get_spec_stats",
        "export_specification",
        "import_specification"
    ],
    "functions": [
        create_specification,
        list_specifications,
        create_change,
        apply_change,
        generate_tasks,
        validate_specification,
        get_spec_stats,
        export_specification,
        import_specification
    ]
}


if __name__ == "__main__":
    # Test the skill
    async def test_skill():
        skill = OpenSpecSkill()
        
        # Test creating a specification
        result = await skill.create_specification(
            title="Test Specification",
            description="A test specification for OpenSpec skill",
            author="Neo-Clone",
            requirements=[
                {
                    "title": "User Authentication",
                    "description": "Implement user authentication system",
                    "acceptance_criteria": [
                        "Users can register with email and password",
                        "Users can login with valid credentials",
                        "Password must be hashed"
                    ],
                    "priority": "high",
                    "tags": ["security", "authentication"]
                }
            ]
        )
        
        print("Create spec result:", result)
        
        if result["success"]:
            spec_id = result["specification_id"]
            
            # Test listing specifications
            specs = await skill.list_specifications()
            print("List specs result:", specs)
            
            # Test generating tasks
            tasks = await skill.generate_tasks(spec_id)
            print("Generate tasks result:", tasks)
    
    asyncio.run(test_skill())