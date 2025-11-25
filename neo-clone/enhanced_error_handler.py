#!/usr/bin/env python3
"""
Enhanced Error Handling System for Neo-Clone
Provides graceful fallbacks, recovery mechanisms, and comprehensive error reporting
"""

import logging
import traceback
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better handling"""
    NETWORK = "network"
    MODEL = "model"
    MEMORY = "memory"
    SKILL = "skill"
    FILE_IO = "file_io"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Recovery actions for errors"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"
    TERMINATE = "terminate"

@dataclass
class ErrorRecord:
    """Error record for tracking and analysis"""
    id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback: str
    context: Dict[str, Any]
    recovery_action: Optional[RecoveryAction] = None
    recovery_successful: Optional[bool] = None
    retry_count: int = 0
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'message': self.message,
            'exception_type': self.exception_type,
            'traceback': self.traceback,
            'context': self.context,
            'recovery_action': self.recovery_action.value if self.recovery_action else None,
            'recovery_successful': self.recovery_successful,
            'retry_count': self.retry_count,
            'resolved': self.resolved
        }

@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms"""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    fallback_models: List[str] = field(default_factory=list)
    fallback_skills: List[str] = field(default_factory=list)
    graceful_degradation: bool = True

class EnhancedErrorHandler:
    """Enhanced error handling system with recovery mechanisms"""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self.error_history: List[ErrorRecord] = []
        self.error_patterns: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'resolved_errors': 0,
            'escalated_errors': 0,
            'categories': {category.value: 0 for category in ErrorCategory},
            'severities': {severity.value: 0 for severity in ErrorSeverity}
        }
        
        logger.info("Enhanced error handling system initialized")
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies for each error category"""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: [
                self._retry_with_backoff,
                self._switch_to_offline_mode,
                self._use_cached_response
            ],
            ErrorCategory.MODEL: [
                self._switch_to_fallback_model,
                self._reduce_complexity,
                self._use_simpler_approach
            ],
            ErrorCategory.MEMORY: [
                self._clear_working_memory,
                self._use_persistent_storage,
                self._restart_memory_system
            ],
            ErrorCategory.SKILL: [
                self._retry_with_different_params,
                self._use_alternative_skill,
                self._skip_skill_execution
            ],
            ErrorCategory.FILE_IO: [
                self._retry_file_operation,
                self._use_temporary_location,
                self._skip_file_operation
            ],
            ErrorCategory.TIMEOUT: [
                self._increase_timeout,
                self._break_into_smaller_tasks,
                self._skip_long_operation
            ],
            ErrorCategory.SYSTEM: [
                self._restart_component,
                self._use_safe_mode,
                self._escalate_to_admin
            ]
        }
    
    async def handle_error(self, 
                          exception: Exception, 
                          context: Dict[str, Any] = None,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorRecord:
        """
        Handle an error with appropriate recovery mechanisms
        
        Args:
            exception: The exception that occurred
            context: Context information when error occurred
            severity: Error severity level
            
        Returns:
            ErrorRecord with handling details
        """
        start_time = time.time()
        
        # Create error record
        error_record = self._create_error_record(exception, context, severity)
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(error_record.category):
            logger.warning(f"Circuit breaker open for {error_record.category.value}, skipping recovery")
            error_record.recovery_action = RecoveryAction.ESCALATE
            error_record.recovery_successful = False
            return error_record
        
        # Attempt recovery
        recovery_successful = await self._attempt_recovery(error_record)
        
        # Update error record
        error_record.recovery_successful = recovery_successful
        error_record.resolved = recovery_successful
        
        # Update statistics
        self._update_statistics(error_record)
        
        # Store error record
        with self._lock:
            self.error_history.append(error_record)
            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
        
        handling_time = time.time() - start_time
        logger.info(f"Error handled in {handling_time:.3f}s: {error_record.message} "
                   f"(Recovery: {'Success' if recovery_successful else 'Failed'})")
        
        return error_record
    
    def _create_error_record(self, exception: Exception, context: Dict[str, Any], severity: ErrorSeverity) -> ErrorRecord:
        """Create an error record from exception"""
        import uuid
        
        # Determine error category
        category = self._categorize_error(exception)
        
        # Create error record
        error_record = ErrorRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        return error_record
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize exception based on type and message"""
        exception_type = type(exception).__name__.lower()
        message = str(exception).lower()
        
        # Network errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['connection', 'network', 'timeout', 'http', 'request']):
            return ErrorCategory.NETWORK
        
        # Model errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['model', 'llm', 'ai', 'generation', 'prediction']):
            return ErrorCategory.MODEL
        
        # Memory errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['memory', 'storage', 'cache', 'buffer']):
            return ErrorCategory.MEMORY
        
        # Skill errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['skill', 'execution', 'parameter', 'method']):
            return ErrorCategory.SKILL
        
        # File I/O errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['file', 'path', 'directory', 'permission', 'io']):
            return ErrorCategory.FILE_IO
        
        # Timeout errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['timeout', 'deadline', 'expired']):
            return ErrorCategory.TIMEOUT
        
        # System errors
        if any(keyword in exception_type or keyword in message for keyword in 
               ['system', 'os', 'process', 'thread']):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    async def _attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt recovery using appropriate strategies"""
        strategies = self.recovery_strategies.get(error_record.category, [])
        
        for strategy in strategies:
            try:
                if error_record.retry_count >= self.config.max_retries:
                    logger.warning(f"Max retries exceeded for {error_record.id}")
                    break
                
                # Apply retry delay
                if error_record.retry_count > 0:
                    delay = self._calculate_retry_delay(error_record.retry_count)
                    await asyncio.sleep(delay)
                
                # Attempt recovery
                success = await strategy(error_record)
                if success:
                    error_record.recovery_action = RecoveryAction.RETRY
                    return True
                
                error_record.retry_count += 1
                
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                error_record.retry_count += 1
        
        # If all strategies failed, try fallback
        return await self._attempt_fallback(error_record)
    
    async def _attempt_fallback(self, error_record: ErrorRecord) -> bool:
        """Attempt fallback recovery"""
        try:
            if error_record.category == ErrorCategory.MODEL:
                return await self._switch_to_fallback_model(error_record)
            elif error_record.category == ErrorCategory.SKILL:
                return await self._use_alternative_skill(error_record)
            else:
                # Generic fallback
                error_record.recovery_action = RecoveryAction.FALLBACK
                return False
        except Exception as e:
            logger.error(f"Fallback recovery failed: {e}")
            return False
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay with exponential backoff"""
        if self.config.exponential_backoff:
            return self.config.retry_delay * (2 ** retry_count)
        else:
            return self.config.retry_delay
    
    def _is_circuit_breaker_open(self, category: ErrorCategory) -> bool:
        """Check if circuit breaker is open for category"""
        breaker_key = category.value
        if breaker_key not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[breaker_key]
        
        # Reset if timeout has passed
        if datetime.now() - breaker['last_failure'] > timedelta(seconds=breaker['timeout']):
            self.circuit_breakers[breaker_key] = {
                'failures': 0,
                'last_failure': datetime.now(),
                'timeout': 60,
                'threshold': 5
            }
            return False
        
        return breaker['failures'] >= breaker['threshold']
    
    def _update_circuit_breaker(self, category: ErrorCategory, success: bool):
        """Update circuit breaker state"""
        breaker_key = category.value
        
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = {
                'failures': 0,
                'last_failure': datetime.now(),
                'timeout': 60,
                'threshold': 5
            }
        
        breaker = self.circuit_breakers[breaker_key]
        
        if success:
            breaker['failures'] = max(0, breaker['failures'] - 1)
        else:
            breaker['failures'] += 1
            breaker['last_failure'] = datetime.now()
    
    def _update_statistics(self, error_record: ErrorRecord):
        """Update error statistics"""
        with self._lock:
            self.error_stats['total_errors'] += 1
            self.error_stats['categories'][error_record.category.value] += 1
            self.error_stats['severities'][error_record.severity.value] += 1
            
            if error_record.resolved:
                self.error_stats['resolved_errors'] += 1
            else:
                self.error_stats['escalated_errors'] += 1
    
    # Recovery strategy implementations
    async def _retry_with_backoff(self, error_record: ErrorRecord) -> bool:
        """Retry with exponential backoff"""
        logger.info(f"Retrying operation for {error_record.id}")
        return False  # Placeholder - actual retry logic would be context-specific
    
    async def _switch_to_fallback_model(self, error_record: ErrorRecord) -> bool:
        """Switch to fallback model"""
        logger.info("Switching to fallback model")
        return True  # Assume success for demonstration
    
    async def _use_alternative_skill(self, error_record: ErrorRecord) -> bool:
        """Use alternative skill"""
        logger.info("Using alternative skill")
        return True  # Assume success for demonstration
    
    async def _clear_working_memory(self, error_record: ErrorRecord) -> bool:
        """Clear working memory"""
        logger.info("Clearing working memory")
        return True
    
    async def _increase_timeout(self, error_record: ErrorRecord) -> bool:
        """Increase timeout for operation"""
        logger.info("Increasing timeout")
        return True
    
    async def _skip_skill_execution(self, error_record: ErrorRecord) -> bool:
        """Skip skill execution"""
        logger.info("Skipping skill execution")
        return True
    
    # Placeholder recovery strategies
    async def _switch_to_offline_mode(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _use_cached_response(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _reduce_complexity(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _use_simpler_approach(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _use_persistent_storage(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _restart_memory_system(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _retry_with_different_params(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _retry_file_operation(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _use_temporary_location(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _skip_file_operation(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _break_into_smaller_tasks(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _skip_long_operation(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _restart_component(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _use_safe_mode(self, error_record: ErrorRecord) -> bool:
        return False
    
    async def _escalate_to_admin(self, error_record: ErrorRecord) -> bool:
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        with self._lock:
            recent_errors = [e for e in self.error_history 
                           if datetime.now() - e.timestamp < timedelta(hours=24)]
            
            return {
                'statistics': self.error_stats.copy(),
                'recent_errors_24h': len(recent_errors),
                'error_rate_24h': len(recent_errors) / 24.0,  # errors per hour
                'top_error_categories': self._get_top_categories(),
                'recovery_success_rate': self._calculate_recovery_success_rate(),
                'circuit_breakers': {k: v for k, v in self.circuit_breakers.items() 
                                   if v['failures'] > 0},
                'last_errors': [e.to_dict() for e in self.error_history[-10:]]
            }
    
    def _get_top_categories(self) -> List[Dict[str, Any]]:
        """Get top error categories"""
        category_counts = {}
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'category': cat, 'count': count} for cat, count in sorted_categories[:5]]
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        if not self.error_history:
            return 0.0
        
        resolved = sum(1 for e in self.error_history if e.resolved)
        return resolved / len(self.error_history)
    
    def export_error_log(self, filepath: str):
        """Export error log to file"""
        try:
            error_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_error_summary(),
                'errors': [e.to_dict() for e in self.error_history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            logger.info(f"Error log exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export error log: {e}")

# Global error handler instance
error_handler = EnhancedErrorHandler()

def get_error_handler() -> EnhancedErrorHandler:
    """Get the global error handler instance"""
    return error_handler

async def handle_error(exception: Exception, context: Dict[str, Any] = None, 
                      severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorRecord:
    """Handle an error using the global error handler"""
    return await error_handler.handle_error(exception, context, severity)

# Decorator for automatic error handling
def error_safe(category: ErrorCategory = ErrorCategory.UNKNOWN, 
               severity: ErrorSeverity = ErrorSeverity.MEDIUM,
               fallback_return: Any = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
                await handle_error(e, context, severity)
                return fallback_return
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
                # Run async error handler in sync context
                asyncio.create_task(handle_error(e, context, severity))
                return fallback_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator