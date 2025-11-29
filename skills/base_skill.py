"""
Base classes for Neo-Clone Skills System
=========================================

Provides foundational classes for skill implementation and execution.
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories of skills for better organization"""
    CODE_ANALYSIS = "code_analysis"
    DATA_PROCESSING = "data_processing"
    WEB_OPERATIONS = "web_operations"
    FILE_MANAGEMENT = "file_management"
    MATHEMATICS = "mathematics"
    REASONING = "reasoning"
    CREATIVE = "creative"
    COMMUNICATION = "communication"
    SYSTEM_OPERATIONS = "system_operations"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    CONVERSATION = "conversation"
    GENERAL = "general"


@dataclass
class SkillMetadata:
    """Metadata for skill registration and discovery"""
    name: str
    description: str
    category: str = "general"
    version: str = "1.0.0"
    author: str = "Neo-Clone"
    tags: list = None
    dependencies: list = None
    parameters: Dict[str, str] = None
    capabilities: list = None
    example: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class SkillResult:
    """Result from skill execution"""
    success: bool
    message: str
    data: Any = None
    timestamp: str = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


class BaseSkill:
    """Base class for all Neo-Clone skills"""
    
    def __init__(self, name: str, description: str, example: str = ""):
        self.name = name
        self.description = description
        self.example = example
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._execution_count = 0
        self._success_count = 0
        self._total_execution_time = 0.0
        
    @property
    def parameters(self) -> Dict[str, str]:
        """Return parameter description for this skill"""
        return {}
    
    @property
    def capabilities(self) -> list:
        """Return list of capabilities for this skill"""
        return []
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return skill metadata"""
        return {
            'name': self.name,
            'description': self.description,
            'example': self.example,
            'parameters': self.parameters,
            'capabilities': self.capabilities,
            'execution_count': self._execution_count,
            'success_count': self._success_count,
            'success_rate': self._success_count / max(self._execution_count, 1),
            'average_execution_time': self._total_execution_time / max(self._execution_count, 1)
        }
    
    def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute the skill with given parameters.
        
        Args:
            params: Dictionary of parameters for skill execution
            
        Returns:
            SkillResult: Result of skill execution
        """
        start_time = datetime.now()
        self._execution_count += 1
        
        try:
            result = self._execute(params)
            self._success_count += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._total_execution_time += execution_time
            
            result.execution_time = execution_time
            result.success = True
            
            self.logger.info(f"Skill '{self.name}' executed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._total_execution_time += execution_time
            
            error_msg = f"Skill '{self.name}' failed: {str(e)}"
            self.logger.error(error_msg)
            
            return SkillResult(
                success=False,
                message=error_msg,
                data={'error': str(e), 'error_type': type(e).__name__},
                execution_time=execution_time
            )
    
    def _execute(self, params: Dict[str, Any]) -> SkillResult:
        """Internal execution method to be implemented by subclasses.
        
        Args:
            params: Dictionary of parameters for skill execution
            
        Returns:
            SkillResult: Result of skill execution (success should be True)
        """
        raise NotImplementedError("Subclasses must implement _execute method")
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters for skill execution.
        
        Args:
            params: Parameters to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        return True, ""
    
    def get_help(self) -> str:
        """Get help text for this skill"""
        help_text = f"""
# {self.name.title()} Skill

**Description**: {self.description}

**Example Usage**: {self.example}

## Parameters:
"""
        for param_name, param_desc in self.parameters.items():
            help_text += f"- **{param_name}**: {param_desc}\n"
        
        help_text += f"""
## Capabilities:
{', '.join(self.capabilities) if self.capabilities else 'General functionality'}

## Statistics:
- Executions: {self._execution_count}
- Success Rate: {self._success_count / max(self._execution_count, 1):.1%}
- Average Execution Time: {self._total_execution_time / max(self._execution_count, 1):.2f}s
"""
        return help_text
    
    def reset_stats(self):
        """Reset execution statistics"""
        self._execution_count = 0
        self._success_count = 0
        self._total_execution_time = 0.0
        self.logger.info(f"Statistics reset for skill '{self.name}'")


class SkillError(Exception):
    """Custom exception for skill-related errors"""
    pass


class SkillValidationError(SkillError):
    """Exception raised when skill parameters are invalid"""
    pass


class SkillExecutionError(SkillError):
    """Exception raised when skill execution fails"""
    pass