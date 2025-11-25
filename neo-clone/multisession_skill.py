#!/usr/bin/env python3
"""
Multi-Session Skill for Neo-Clone
Integrates the multi-session manager with Neo-Clone's skills framework
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add neo-clone to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import Neo-Clone base classes with fallbacks
try:
    from skills import BaseSkill, SkillResult, SkillMetadata, SkillCategory, SkillParameter, SkillParameterType
    from data_models import SkillContext
    from multisession_neo_clone import MultiSessionManager, SessionType, SessionStatus
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    # Fallback classes for standalone operation
    from abc import ABC, abstractmethod
    
    class BaseSkill(ABC):
        def __init__(self):
            self.metadata = SkillMetadata("multisession", "general", "Multi-session management")
            self.execution_count = 0
            self.success_count = 0
            self.average_execution_time = 0.0
        
        @abstractmethod
        async def _execute_async(self, context, **kwargs):
            pass
        
        def get_parameters(self):
            return {}
    
    class SkillMetadata:
        def __init__(self, name, category, description):
            self.name = name
            self.category = category
            self.description = description
            self.capabilities = []
            self.parameters = {}
            self.examples = []
    
    class SkillCategory:
        GENERAL = "general"
    
    class SkillContext:
        def __init__(self, user_input, intent, conversation_history):
            self.user_input = user_input
            self.intent = intent
            self.conversation_history = conversation_history
    
    class SkillResult:
        def __init__(self, success: bool, output: Any = None, skill_name: str = "", execution_time: float = 0.0, error_message: str = "", metadata: dict = None):
            self.success = success
            self.output = output
            self.skill_name = skill_name
            self.execution_time = execution_time
            self.error_message = error_message
            self.metadata = metadata or {}
    
    class SkillParameter:
        def __init__(self, name, param_type, required=False, default=None, description=""):
            self.name = name
            self.param_type = param_type
            self.required = required
            self.default = default
            self.description = description
    
    class SkillParameterType:
        STRING = "string"
        INTEGER = "integer"
        BOOLEAN = "boolean"
        DICT = "dict"
    
    # Fallback MultiSessionManager
    class MultiSessionManager:
        def __init__(self):
            pass
        async def create_session(self, *args, **kwargs):
            return {"id": "fallback-session", "status": "created"}
        async def list_sessions(self):
            return []
        async def get_session(self, session_id):
            return None
        async def terminate_session(self, session_id):
            return {"success": True}
    
    class SessionType:
        ISOLATED = "isolated"
        SHARED = "shared"
    
    class SessionStatus:
        ACTIVE = "active"
        TERMINATED = "terminated"


class MultiSessionSkill(BaseSkill):
    """Multi-session management skill for Neo-Clone"""
    
    def __init__(self):
        super().__init__()
        self.manager = None
        self.initialized = False
        
        # Update metadata
        self.metadata.name = "multisession"
        self.metadata.category = SkillCategory.GENERAL
        self.metadata.description = "Manages multiple Neo-Clone sessions with isolation and parallel execution"
        self.metadata.capabilities = [
            "create_session",
            "list_sessions", 
            "get_session",
            "terminate_session",
            "session_isolation",
            "parallel_execution"
        ]
        self.metadata.parameters = {
            "action": {"type": "string", "description": "Action to perform"},
            "session_id": {"type": "string", "description": "Session ID"},
            "session_type": {"type": "string", "description": "Type of session"},
            "config": {"type": "object", "description": "Session configuration"}
        }
        self.metadata.examples = [
            "Create a new isolated session",
            "List all active sessions",
            "Terminate a specific session"
        ]

    def get_parameters(self):
        """Get skill parameters"""
        try:
            return {
                "action": SkillParameter(
                    name="action",
                    param_type=SkillParameterType.STRING,
                    required=False,
                    description="Action to perform (create, list, get, terminate)"
                ),
                "session_id": SkillParameter(
                    name="session_id",
                    param_type=SkillParameterType.STRING,
                    required=False,
                    description="Session ID"
                ),
                "session_type": SkillParameter(
                    name="session_type",
                    param_type=SkillParameterType.STRING,
                    required=False,
                    description="Type of session (isolated, shared)"
                ),
                "config": SkillParameter(
                    name="config",
                    param_type=SkillParameterType.DICT,
                    required=False,
                    description="Session configuration"
                )
            }
        except:
            return {
                "action": {"type": "string", "required": False, "description": "Action to perform"},
                "session_id": {"type": "string", "required": False, "description": "Session ID"},
                "session_type": {"type": "string", "required": False, "description": "Session type"},
                "config": {"type": "object", "required": False, "description": "Session configuration"}
            }

    async def _execute_async(self, context: SkillContext, **kwargs) -> SkillResult:
        """Execute multi-session operations"""
        start_time = time.time()
        
        try:
            # Initialize if not already done
            if not self.initialized:
                init_result = await self.initialize()
                if not init_result.success:
                    return SkillResult(
                        success=False,
                        output=None,
                        skill_name=self.metadata.name,
                        execution_time=time.time() - start_time,
                        error_message=init_result.error
                    )
            
            action = kwargs.get("action", "list")
            
            if action == "create":
                result = await self._create_session(kwargs)
            elif action == "list":
                result = await self._list_sessions()
            elif action == "get":
                result = await self._get_session(kwargs.get("session_id"))
            elif action == "terminate":
                result = await self._terminate_session(kwargs.get("session_id"))
            else:
                result = {"error": f"Unknown action: {action}"}
            
            execution_time = time.time() - start_time
            return SkillResult(
                success=result.get("success", True),
                output=result,
                skill_name=self.metadata.name,
                execution_time=execution_time,
                metadata={"action": action}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SkillResult(
                success=False,
                output=None,
                skill_name=self.metadata.name,
                execution_time=execution_time,
                error_message=str(e)
            )

    async def initialize(self):
        """Initialize the multi-session manager"""
        if not self.initialized:
            try:
                self.manager = MultiSessionManager()
                await self.manager.load_session_configs()
                self.initialized = True
                return SkillResult(
                    success=True,
                    message="Multi-session manager initialized successfully"
                )
            except Exception as e:
                return SkillResult(
                    success=False,
                    error=f"Failed to initialize multi-session manager: {str(e)}"
                )
        return SkillResult(success=True, message="Already initialized")
    
    async def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute multi-session operations"""
        try:
            # Initialize if not already done
            if not self.initialized:
                init_result = await self.initialize()
                if not init_result.success:
                    return init_result
            
            operation = params.get("operation", "")
            
            if operation == "create_session":
                return await self._create_session(params)
            elif operation == "list_sessions":
                return await self._list_sessions()
            elif operation == "execute_in_session":
                return await self._execute_in_session(params)
            elif operation == "terminate_session":
                return await self._terminate_session(params)
            elif operation == "get_session_status":
                return await self._get_session_status(params)
            elif operation == "system_status":
                return await self._get_system_status()
            elif operation == "cleanup_sessions":
                return await self._cleanup_sessions()
            elif operation == "create_parallel_sessions":
                return await self._create_parallel_sessions(params)
            elif operation == "batch_execute":
                return await self._batch_execute(params)
            else:
                return SkillResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
        
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Error executing multi-session operation: {str(e)}"
            )
    
    async def _create_session(self, params: Dict[str, Any]) -> SkillResult:
        """Create a new session"""
        try:
            name = params.get("name")
            if not name:
                return SkillResult(success=False, error="Session name is required")
            
            session_type = SessionType(params.get("type", "general"))
            priority = params.get("priority", 5)
            background = params.get("background", False)
            
            session_id = await self.manager.create_session(
                name=name,
                session_type=session_type,
                priority=priority,
                background=background
            )
            
            return SkillResult(
                success=True,
                data={
                    "session_id": session_id,
                    "name": name,
                    "type": session_type.value,
                    "background": background
                },
                message=f"Session '{name}' created successfully with ID: {session_id}"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to create session: {str(e)}")
    
    async def _list_sessions(self) -> SkillResult:
        """List all active sessions"""
        try:
            sessions = await self.manager.list_sessions()
            return SkillResult(
                success=True,
                data=sessions,
                message=f"Found {len(sessions)} active sessions"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to list sessions: {str(e)}")
    
    async def _execute_in_session(self, params: Dict[str, Any]) -> SkillResult:
        """Execute a command in a specific session"""
        try:
            session_id = params.get("session_id")
            command = params.get("command")
            args = params.get("args", [])
            
            if not session_id or not command:
                return SkillResult(
                    success=False,
                    error="Both session_id and command are required"
                )
            
            result = await self.manager.execute_in_session(session_id, command, args)
            
            return SkillResult(
                success=result["success"],
                data=result,
                message=f"Command executed in session {session_id}"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to execute command: {str(e)}")
    
    async def _terminate_session(self, params: Dict[str, Any]) -> SkillResult:
        """Terminate a session"""
        try:
            session_id = params.get("session_id")
            if not session_id:
                return SkillResult(success=False, error="Session ID is required")
            
            success = await self.manager.terminate_session(session_id)
            
            return SkillResult(
                success=success,
                data={"session_id": session_id, "terminated": success},
                message=f"Session {session_id} terminated" if success else f"Failed to terminate session {session_id}"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to terminate session: {str(e)}")
    
    async def _get_session_status(self, params: Dict[str, Any]) -> SkillResult:
        """Get status of a specific session"""
        try:
            session_id = params.get("session_id")
            if not session_id:
                return SkillResult(success=False, error="Session ID is required")
            
            session = await self.manager.get_session(session_id)
            if not session:
                return SkillResult(success=False, error=f"Session {session_id} not found")
            
            status = session.get_status()
            return SkillResult(
                success=True,
                data=status,
                message=f"Retrieved status for session {session_id}"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to get session status: {str(e)}")
    
    async def _get_system_status(self) -> SkillResult:
        """Get overall system status"""
        try:
            status = await self.manager.get_system_status()
            return SkillResult(
                success=True,
                data=status,
                message="System status retrieved successfully"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to get system status: {str(e)}")
    
    async def _cleanup_sessions(self) -> SkillResult:
        """Clean up terminated sessions"""
        try:
            count = await self.manager.cleanup_terminated_sessions()
            return SkillResult(
                success=True,
                data={"cleaned_sessions": count},
                message=f"Cleaned up {count} terminated sessions"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to cleanup sessions: {str(e)}")
    
    async def _create_parallel_sessions(self, params: Dict[str, Any]) -> SkillResult:
        """Create multiple parallel sessions"""
        try:
            session_configs = params.get("sessions", [])
            if not session_configs:
                return SkillResult(success=False, error="Session configurations are required")
            
            created_sessions = []
            errors = []
            
            for config in session_configs:
                try:
                    session_id = await self.manager.create_session(
                        name=config.get("name", f"Session-{len(created_sessions)}"),
                        session_type=SessionType(config.get("type", "general")),
                        priority=config.get("priority", 5),
                        background=config.get("background", False)
                    )
                    created_sessions.append({
                        "session_id": session_id,
                        "name": config.get("name"),
                        "type": config.get("type", "general")
                    })
                except Exception as e:
                    errors.append(f"Failed to create session {config.get('name', 'unknown')}: {str(e)}")
            
            return SkillResult(
                success=len(created_sessions) > 0,
                data={
                    "created_sessions": created_sessions,
                    "errors": errors,
                    "total_requested": len(session_configs),
                    "total_created": len(created_sessions)
                },
                message=f"Created {len(created_sessions)} out of {len(session_configs)} requested sessions"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to create parallel sessions: {str(e)}")
    
    async def _batch_execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute commands in multiple sessions"""
        try:
            commands = params.get("commands", [])
            if not commands:
                return SkillResult(success=False, error="Commands list is required")
            
            results = []
            
            for cmd in commands:
                session_id = cmd.get("session_id")
                command = cmd.get("command")
                args = cmd.get("args", [])
                
                if session_id and command:
                    try:
                        result = await self.manager.execute_in_session(session_id, command, args)
                        results.append({
                            "session_id": session_id,
                            "command": command,
                            "success": result["success"],
                            "result": result
                        })
                    except Exception as e:
                        results.append({
                            "session_id": session_id,
                            "command": command,
                            "success": False,
                            "error": str(e)
                        })
            
            successful = sum(1 for r in results if r["success"])
            return SkillResult(
                success=successful > 0,
                data={
                    "results": results,
                    "total_commands": len(commands),
                    "successful_commands": successful
                },
                message=f"Executed {successful} out of {len(commands)} commands successfully"
            )
        
        except Exception as e:
            return SkillResult(success=False, error=f"Failed to execute batch commands: {str(e)}")


# Convenience functions for direct usage
async def create_session(name: str, session_type: str = "general", **kwargs) -> Dict[str, Any]:
    """Create a new session"""
    skill = MultiSessionSkill()
    result = await skill.execute({
        "operation": "create_session",
        "name": name,
        "type": session_type,
        **kwargs
    })
    
    if result.success:
        return {
            "success": True,
            "session_id": result.data["session_id"],
            "message": result.message
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


async def list_sessions() -> Dict[str, Any]:
    """List all active sessions"""
    skill = MultiSessionSkill()
    result = await skill.execute({"operation": "list_sessions"})
    
    if result.success:
        return {
            "success": True,
            "sessions": result.data,
            "message": result.message
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


async def execute_in_session(session_id: str, command: str, args: List[str] = None) -> Dict[str, Any]:
    """Execute a command in a specific session"""
    skill = MultiSessionSkill()
    result = await skill.execute({
        "operation": "execute_in_session",
        "session_id": session_id,
        "command": command,
        "args": args or []
    })
    
    if result.success:
        return {
            "success": True,
            "result": result.data,
            "message": result.message
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


async def terminate_session(session_id: str) -> Dict[str, Any]:
    """Terminate a session"""
    skill = MultiSessionSkill()
    result = await skill.execute({
        "operation": "terminate_session",
        "session_id": session_id
    })
    
    if result.success:
        return {
            "success": True,
            "terminated": result.data["terminated"],
            "message": result.message
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


async def get_system_status() -> Dict[str, Any]:
    """Get overall system status"""
    skill = MultiSessionSkill()
    result = await skill.execute({"operation": "system_status"})
    
    if result.success:
        return {
            "success": True,
            "status": result.data,
            "message": result.message
        }
    else:
        return {
            "success": False,
            "error": result.error
        }


# CLI interface
async def main():
    """CLI interface for multi-session skill"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neo-Clone Multi-Session Skill")
    parser.add_argument("operation", choices=[
        "create", "list", "execute", "terminate", "status", "cleanup", "parallel", "batch"
    ], help="Operation to perform")
    parser.add_argument("--name", help="Session name")
    parser.add_argument("--type", default="general", help="Session type")
    parser.add_argument("--session-id", help="Session ID")
    parser.add_argument("--command", help="Command to execute")
    parser.add_argument("--args", nargs="*", help="Command arguments")
    parser.add_argument("--config", help="JSON config for parallel/batch operations")
    
    args = parser.parse_args()
    
    try:
        if args.operation == "create":
            if not args.name:
                print("Error: --name is required for create operation")
                return
            
            result = await create_session(args.name, args.type)
            if result["success"]:
                print(f"✅ {result['message']}")
                print(f"Session ID: {result['session_id']}")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.operation == "list":
            result = await list_sessions()
            if result["success"]:
                print(f"✅ {result['message']}")
                if result["sessions"]:
                    for session in result["sessions"]:
                        print(f"  {session['session_id']}: {session['name']} ({session['type']}) - {session['status']}")
                else:
                    print("  No active sessions")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.operation == "execute":
            if not args.session_id or not args.command:
                print("Error: --session-id and --command are required for execute operation")
                return
            
            result = await execute_in_session(args.session_id, args.command, args.args)
            if result["success"]:
                print(f"✅ {result['message']}")
                if result["result"]["success"]:
                    if result["result"].get("output"):
                        print(f"Output: {result['result']['output']}")
                else:
                    if result["result"].get("error"):
                        print(f"Error: {result['result']['error']}")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.operation == "terminate":
            if not args.session_id:
                print("Error: --session-id is required for terminate operation")
                return
            
            result = await terminate_session(args.session_id)
            if result["success"]:
                print(f"✅ {result['message']}")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.operation == "status":
            result = await get_system_status()
            if result["success"]:
                print(f"✅ {result['message']}")
                print("System Status:")
                for key, value in result["status"].items():
                    print(f"  {key}: {value}")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.operation == "cleanup":
            skill = MultiSessionSkill()
            result = await skill.execute({"operation": "cleanup_sessions"})
            if result.success:
                print(f"✅ {result.message}")
                print(f"Cleaned sessions: {result.data['cleaned_sessions']}")
            else:
                print(f"❌ Error: {result.error}")
        
        elif args.operation == "parallel":
            if not args.config:
                print("Error: --config is required for parallel operation")
                return
            
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            skill = MultiSessionSkill()
            result = await skill.execute({
                "operation": "create_parallel_sessions",
                "sessions": config.get("sessions", [])
            })
            
            if result.success:
                print(f"✅ {result.message}")
                print(f"Created: {result.data['total_created']}/{result.data['total_requested']}")
                for session in result.data["created_sessions"]:
                    print(f"  {session['session_id']}: {session['name']}")
            else:
                print(f"❌ Error: {result.error}")
        
        elif args.operation == "batch":
            if not args.config:
                print("Error: --config is required for batch operation")
                return
            
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            skill = MultiSessionSkill()
            result = await skill.execute({
                "operation": "batch_execute",
                "commands": config.get("commands", [])
            })
            
            if result.success:
                print(f"✅ {result.message}")
                print(f"Commands: {result.data['successful_commands']}/{result.data['total_commands']}")
                for cmd_result in result.data["results"]:
                    status = "✅" if cmd_result["success"] else "❌"
                    print(f"  {status} {cmd_result['session_id']}: {cmd_result['command']}")
            else:
                print(f"❌ Error: {result.error}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())