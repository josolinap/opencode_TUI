from functools import lru_cache
'\nUnified Brain System - Consolidated Neo-Clone Brain\n\nThis module consolidates all brain implementations into a single,\ncohesive system that integrates with the OpenCode CLI.\n'
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

@dataclass
class BrainResponse:
    """Standardized brain response"""
    success: bool
    content: str
    reasoning: str
    skill_used: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

class UnifiedBrain:
    """Unified brain system that consolidates all brain capabilities"""

    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.skills_manager = None
        self.model_analytics = None
        self.initialized = False
        self.capabilities = []
        self._initialize_components()

    def _get_default_config(self):
        """Get default configuration"""

        class DefaultConfig:

            def __init__(self):
                self.provider = 'ollama'
                self.model = 'llama2'
                self.endpoint = 'http://localhost:11434'
                self.reasoning_mode = 'enhanced'
                self.skill_discovery = True
        return DefaultConfig()

    @lru_cache(maxsize=128)
    def _initialize_components(self):
        """Initialize all brain components"""
        try:
            self.skills_manager = None
            try:
                from skills.opencode_skills_manager import OpenCodeSkillsManager
                self.skills_manager = OpenCodeSkillsManager()
                logger.info('Loaded SkillsManager from skills.opencode_skills_manager')
            except ImportError:
                try:
                    from skills import SkillsManager
                    self.skills_manager = SkillsManager()
                    logger.info('Loaded SkillsManager from skills module')
                except ImportError:
                    logger.warning('Could not load SkillsManager')
            if self.skills_manager and hasattr(self.skills_manager, 'initialize'):
                success = self.skills_manager.initialize()
                if success:
                    logger.info('Skills manager initialized successfully')
                    self.capabilities.extend(self._get_skill_capabilities())
                else:
                    logger.warning('Skills manager initialization failed')
            try:
                from model_analytics import ModelAnalytics
                self.model_analytics = ModelAnalytics()
                logger.info('Model analytics initialized')
            except ImportError:
                logger.warning('Model analytics not available')
            self._load_builtin_skills()
            self.initialized = True
            logger.info('Unified brain initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize unified brain: {e}')
            self.initialized = False

    def _get_skill_capabilities(self) -> List[str]:
        """Get capabilities from skills manager"""
        capabilities = []
        if self.skills_manager and hasattr(self.skills_manager, 'skills'):
            for (skill_name, skill_info) in self.skills_manager.skills.items():
                if hasattr(skill_info, 'capabilities'):
                    capabilities.extend(skill_info.capabilities)
                else:
                    capabilities.append(skill_name)
        return list(set(capabilities))

    def _load_builtin_skills(self):
        """Load built-in skills from neo-clone directory"""
        builtin_skills = ['code_generation', 'text_analysis', 'data_inspector', 'ml_training', 'file_manager', 'web_search', 'minimax_agent']
        neo_clone_path = Path(__file__).parent
        for skill_name in builtin_skills:
            skill_file = neo_clone_path / f'{skill_name}.py'
            if skill_file.exists():
                try:
                    spec = __import__('importlib.util').util.spec_from_file_location(skill_name, skill_file)
                    module = __import__('importlib.util').util.module_from_spec(spec)
                    sys.modules[skill_name] = module
                    spec.loader.exec_module(module)
                    if hasattr(module, 'main') or hasattr(module, skill_name):
                        self.capabilities.append(skill_name)
                        logger.debug(f'Loaded built-in skill: {skill_name}')
                except Exception as e:
                    logger.warning(f'Failed to load built-in skill {skill_name}: {e}')

    def process_request(self, message: str, context: Dict[str, Any]=None) -> BrainResponse:
        """Process a request using the unified brain"""
        if not self.initialized:
            return BrainResponse(success=False, content='Brain not initialized', reasoning='Unified brain failed to initialize properly')
        try:
            intent = self._analyze_intent(message)
            skill_result = self._execute_skill(intent, message, context)
            return BrainResponse(success=True, content=skill_result.get('content', ''), reasoning=skill_result.get('reasoning', f'Used {intent} skill'), skill_used=intent, confidence=skill_result.get('confidence', 0.8), metadata=skill_result.get('metadata', {}))
        except Exception as e:
            logger.error(f'Error processing request: {e}')
            return BrainResponse(success=False, content=f'Error processing request: {str(e)}', reasoning='An error occurred during processing')

    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent and select appropriate skill"""
        message_lower = message.lower()
        intent_keywords = {'code_generation': ['code', 'python', 'function', 'class', 'algorithm', 'programming'], 'text_analysis': ['text', 'sentiment', 'analysis', 'analyze text'], 'data_inspector': ['data', 'csv', 'json', 'analyze data', 'dataset'], 'ml_training': ['machine learning', 'ml', 'model', 'training', 'neural network'], 'file_manager': ['file', 'read', 'write', 'directory', 'folder'], 'web_search': ['search', 'web', 'google', 'find information', 'lookup'], 'minimax_agent': ['reasoning', 'complex', 'analyze', 'strategy']}
        intent_scores = {}
        for (intent, keywords) in intent_keywords.items():
            score = sum((1 for keyword in keywords if keyword in message_lower))
            if score > 0:
                intent_scores[intent] = score
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'minimax_agent'

    def _execute_skill(self, intent: str, message: str, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Execute the selected skill"""
        context = context or {}
        if self.skills_manager and hasattr(self.skills_manager, 'get_skill'):
            skill = self.skills_manager.get_skill(intent)
            if skill:
                try:
                    result = skill.process(message, context)
                    return {'content': result, 'reasoning': f'Executed {intent} skill via skills manager', 'confidence': 0.9}
                except Exception as e:
                    logger.warning(f'Skills manager execution failed: {e}')
        try:
            skill_module = sys.modules.get(intent)
            if skill_module:
                if hasattr(skill_module, 'main'):
                    result = skill_module.main(message)
                    return {'content': result, 'reasoning': f'Executed built-in {intent} skill', 'confidence': 0.8}
                elif hasattr(skill_module, intent):
                    skill_func = getattr(skill_module, intent)
                    result = skill_func(message)
                    return {'content': result, 'reasoning': f'Executed built-in {intent} function', 'confidence': 0.8}
        except Exception as e:
            logger.warning(f'Built-in skill execution failed: {e}')
        return {'content': f"I processed your request about '{message}' using {intent} intent, but the specific skill execution failed.", 'reasoning': f'Intent detected as {intent} but skill execution failed', 'confidence': 0.3}

    def get_capabilities(self) -> List[str]:
        """Get list of available capabilities"""
        return self.capabilities.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get brain status"""
        return {'initialized': self.initialized, 'capabilities_count': len(self.capabilities), 'capabilities': self.capabilities, 'skills_manager_loaded': self.skills_manager is not None, 'model_analytics_loaded': self.model_analytics is not None, 'config': {'provider': self.config.provider, 'model': self.config.model, 'endpoint': self.config.endpoint}}
_unified_brain = None

def get_unified_brain(config=None) -> UnifiedBrain:
    """Get the global unified brain instance"""
    global _unified_brain
    if _unified_brain is None:
        _unified_brain = UnifiedBrain(config)
    return _unified_brain

def reset_unified_brain():
    """Reset the global unified brain instance"""
    global _unified_brain
    _unified_brain = None
__all__ = ['UnifiedBrain', 'BrainResponse', 'get_unified_brain', 'reset_unified_brain']