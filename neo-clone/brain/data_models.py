"""
Data Models for MiniMax Agent Architecture

This module defines all data structures, enums, and models used throughout
the brain system for message handling, memory management, and skill execution.

Author: MiniMax Agent
Version: 1.0
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import uuid


class MessageRole(Enum):
    """Message role types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class MemoryType(Enum):
    """Memory entry types"""
    EPISODIC = "episodic"      # Conversation history
    SEMANTIC = "semantic"      # Knowledge and facts
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"        # Temporary working memory
    EMOTIONAL = "emotional"    # User preferences and context


class IntentType(Enum):
    """Intent types for user input classification"""
    CODE = "code"
    DATA_ANALYSIS = "data_analysis"
    CONVERSATION = "conversation"
    PLANNING = "planning"
    REASONING = "reasoning"
    DEBUGGING = "debugging"
    LEARNING = "learning"
    SYSTEM = "system"


class SkillCategory(Enum):
    """Skill categories"""
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    REASONING = "reasoning"
    PLANNING = "planning"
    SYSTEM_OPERATIONS = "system_operations"
    GENERAL = "general"


class SkillExecutionStatus(Enum):
    """Skill execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Message:
    """Message data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    role: MessageRole = MessageRole.USER
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class ConversationHistory:
    """Conversation history structure"""
    session_id: str = ""
    messages: List[Message] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, content: str, role: MessageRole, metadata: Dict[str, Any] = None) -> str:
        """Add a message to the conversation"""
        message = Message(
            content=content,
            role=role,
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message.id
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages"""
        return self.messages[-count:] if self.messages else []
    
    def clear(self):
        """Clear conversation history"""
        self.messages.clear()


@dataclass
class MemoryEntry:
    """Memory entry structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()
        # Ensure importance is between 0 and 1
        self.importance = max(0.0, min(1.0, self.importance))


@dataclass
class SkillResult:
    """Skill execution result"""
    success: bool = True
    output: Any = None
    skill_name: str = ""
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class SkillContext:
    """Context for skill execution"""
    user_input: str = ""
    intent: IntentType = IntentType.CONVERSATION
    conversation_history: List[Message] = field(default_factory=list)
    memory_context: List[MemoryEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Individual reasoning step"""
    step_type: str = ""
    description: str = ""
    input_data: Any = None
    output_data: Any = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class MiniMaxReasoningTrace:
    """Complete reasoning trace"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[ReasoningStep] = field(default_factory=list)
    intent_analysis: Dict[str, Any] = field(default_factory=dict)
    final_confidence: float = 0.0
    total_execution_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        if not self.start_time:
            self.start_time = datetime.now()
    
    def add_step(self, step_type: str, description: str, input_data: Any = None, 
                 output_data: Any = None, confidence: float = 0.0) -> None:
        """Add a reasoning step"""
        step = ReasoningStep(
            step_type=step_type,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence
        )
        self.steps.append(step)
    
    def add_intent_analysis(self, intent: str, confidence: float, categories: List[str], 
                          metadata: Dict[str, Any] = None) -> None:
        """Add intent analysis results"""
        self.intent_analysis = {
            "intent": intent,
            "confidence": confidence,
            "categories": categories,
            "metadata": metadata or {}
        }
    
    def finalize(self, final_confidence: float = 0.0, execution_time: float = 0.0) -> None:
        """Finalize the reasoning trace"""
        self.final_confidence = final_confidence
        self.total_execution_time = execution_time
        self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get reasoning trace summary"""
        return {
            "trace_id": self.trace_id,
            "steps_count": len(self.steps),
            "final_confidence": self.final_confidence,
            "total_execution_time": self.total_execution_time,
            "intent": self.intent_analysis.get("intent", "unknown"),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    operation_name: str = ""
    execution_time: float = 0.0
    success: bool = True
    input_size: int = 0
    output_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class SkillParameter:
    """Skill parameter definition"""
    name: str = ""
    param_type: str = "string"
    required: bool = True
    default: Any = None
    description: str = ""
    choices: List[str] = None
    
    def __post_init__(self):
        if self.choices is None:
            self.choices = []


class SkillParameterType:
    """Skill parameter types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


class SkillStatus:
    """Skill execution status"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    """Result from skill execution"""
    success: bool = True
    message: str = ""
    data: Any = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SkillContext:
    """Context for skill execution"""
    user_input: str = ""
    session_id: str = ""
    conversation_history: List[Message] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VectorSearchResult:
    """Vector memory search result"""
    content: str = ""
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


# Utility functions

def create_user_message(content: str, metadata: Dict[str, Any] = None) -> Message:
    """Create a user message"""
    return Message(
        content=content,
        role=MessageRole.USER,
        metadata=metadata or {}
    )


def create_assistant_message(content: str, metadata: Dict[str, Any] = None) -> Message:
    """Create an assistant message"""
    return Message(
        content=content,
        role=MessageRole.ASSISTANT,
        metadata=metadata or {}
    )


def create_system_message(content: str, metadata: Dict[str, Any] = None) -> Message:
    """Create a system message"""
    return Message(
        content=content,
        role=MessageRole.SYSTEM,
        metadata=metadata or {}
    )


def create_memory_entry(content: str, memory_type: MemoryType, importance: float = 0.5, 
                       tags: List[str] = None, metadata: Dict[str, Any] = None) -> MemoryEntry:
    """Create a memory entry"""
    return MemoryEntry(
        content=content,
        memory_type=memory_type,
        importance=importance,
        tags=tags or [],
        metadata=metadata or {}
    )


def create_conversation(session_id: str, metadata: Dict[str, Any] = None) -> ConversationHistory:
    """Create a conversation history"""
    return ConversationHistory(
        session_id=session_id,
        metadata=metadata or {}
    )


def create_skill_result(success: bool, output: Any, skill_name: str, 
                       execution_time: float = 0.0, error_message: str = None) -> SkillResult:
    """Create a skill result"""
    return SkillResult(
        success=success,
        output=output,
        skill_name=skill_name,
        execution_time=execution_time,
        error_message=error_message
    )


def create_reasoning_step(step_type: str, description: str, input_data: Any = None,
                         output_data: Any = None, confidence: float = 0.0) -> ReasoningStep:
    """Create a reasoning step"""
    return ReasoningStep(
        step_type=step_type,
        description=description,
        input_data=input_data,
        output_data=output_data,
        confidence=confidence
    )


def create_performance_metrics(operation_name: str, execution_time: float, success: bool,
                              input_size: int = 0, output_size: int = 0, 
                              metadata: Dict[str, Any] = None) -> PerformanceMetrics:
    """Create performance metrics"""
    return PerformanceMetrics(
        operation_name=operation_name,
        execution_time=execution_time,
        success=success,
        input_size=input_size,
        output_size=output_size,
        metadata=metadata or {}
    )
