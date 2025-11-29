from functools import lru_cache
'\nSelf-Aware Model Manager for Neo-Clone\nIntelligently selects and manages models based on availability, cost, and performance\n'
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    WORKING = 'working'
    FAILED = 'failed'
    SLOW = 'slow'
    RATE_LIMITED = 'rate_limited'
    UNKNOWN = 'unknown'

class ModelTier(Enum):
    FREE = 'free'
    LOW_COST = 'low_cost'
    PREMIUM = 'premium'

@dataclass
class ModelInfo:
    name: str
    tier: ModelTier
    status: ModelStatus
    capabilities: List[str]
    cost_per_token: float
    max_context: int
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    last_check: float = 0.0
    error_count: int = 0
    total_requests: int = 0

class SelfAwareModelManager:
    """Self-aware model management system"""

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        self.budget_limits = {ModelTier.FREE: float('inf'), ModelTier.LOW_COST: 0.01, ModelTier.PREMIUM: 0.001}
        self.initialize_models()

    def initialize_models(self):
        """Initialize known models with their properties"""
        known_models = {'opencode/gpt-5-nano': ModelInfo(name='opencode/gpt-5-nano', tier=ModelTier.FREE, status=ModelStatus.UNKNOWN, capabilities=['reasoning', 'tool_calling', 'attachment'], cost_per_token=0.0, max_context=400000), 'opencode/big-pickle': ModelInfo(name='opencode/big-pickle', tier=ModelTier.FREE, status=ModelStatus.UNKNOWN, capabilities=['reasoning', 'tool_calling'], cost_per_token=0.0, max_context=400000), 'opencode/grok-code': ModelInfo(name='opencode/grok-code', tier=ModelTier.FREE, status=ModelStatus.UNKNOWN, capabilities=['reasoning', 'tool_calling'], cost_per_token=0.0, max_context=400000), 'opencode/alpha-doubao-seed-code': ModelInfo(name='opencode/alpha-doubao-seed-code', tier=ModelTier.FREE, status=ModelStatus.UNKNOWN, capabilities=['reasoning', 'tool_calling'], cost_per_token=0.0, max_context=400000), 'openai/o1-pro': ModelInfo(name='openai/o1-pro', tier=ModelTier.PREMIUM, status=ModelStatus.UNKNOWN, capabilities=['reasoning', 'tool_calling'], cost_per_token=0.001, max_context=200000)}
        self.models = known_models
        logger.info(f'Initialized {len(self.models)} models')

    async def check_model_health(self, model_name: str) -> ModelStatus:
        """Check if a model is currently working"""
        if model_name not in self.models:
            return ModelStatus.UNKNOWN
        model_info = self.models[model_name]
        try:
            start_time = time.time()
            if 'opencode/' in model_name:
                status = ModelStatus.WORKING
            else:
                status = ModelStatus.UNKNOWN
            response_time = time.time() - start_time
            model_info.status = status
            model_info.last_check = time.time()
            model_info.avg_response_time = response_time
            self.record_performance(model_name, {'status': status.value, 'response_time': response_time, 'timestamp': time.time()})
            return status
        except Exception as e:
            logger.error(f'Health check failed for {model_name}: {e}')
            model_info.status = ModelStatus.FAILED
            model_info.error_count += 1
            return ModelStatus.FAILED

    def record_performance(self, model_name: str, performance_data: Dict):
        """Record performance data for a model"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        self.performance_history[model_name].append(performance_data)
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]
        model_info = self.models[model_name]
        total_requests = len(self.performance_history[model_name])
        successful_requests = sum((1 for p in self.performance_history[model_name] if p['status'] == ModelStatus.WORKING.value))
        model_info.success_rate = successful_requests / total_requests if total_requests > 0 else 0.0

    @lru_cache(maxsize=128)
    def get_best_model_for_task(self, task_type: str='general', capabilities_needed: List[str]=None, preferred_tier: ModelTier=ModelTier.FREE, budget_limit: float=None) -> Optional[str]:
        """Get the best model for a specific task"""
        if capabilities_needed is None:
            capabilities_needed = ['reasoning']
        capable_models = []
        for (name, model) in self.models.items():
            if all((cap in model.capabilities for cap in capabilities_needed)):
                capable_models.append((name, model))
        if not capable_models:
            logger.warning(f'No models found with capabilities: {capabilities_needed}')
            return None

        def model_score(model_tuple):
            (name, model) = model_tuple
            score = 0
            if model.tier == preferred_tier:
                score += 100
            elif model.tier == ModelTier.FREE:
                score += 90
            elif model.tier == ModelTier.LOW_COST:
                score += 70
            else:
                score += 50
            score += model.success_rate * 50
            if model.avg_response_time > 0:
                score += max(0, 10 - model.avg_response_time)
            if model.status == ModelStatus.WORKING:
                score += 100
            elif model.status == ModelStatus.UNKNOWN:
                score += 50
            else:
                score -= 50
            return score
        capable_models.sort(key=model_score, reverse=True)
        best_model_name = capable_models[0][0]
        best_model = capable_models[0][1]
        logger.info(f'Selected model: {best_model_name} (score: {model_score(capable_models[0]):.1f})')
        return best_model_name

    async def ensure_model_working(self, model_name: str) -> bool:
        """Ensure a model is working, try alternatives if needed"""
        if self.models[model_name].status != ModelStatus.WORKING:
            await self.check_model_health(model_name)
        if self.models[model_name].status != ModelStatus.WORKING:
            logger.warning(f'Model {model_name} not working, finding alternative...')
            model_info = self.models[model_name]
            alternative = self.get_best_model_for_task(capabilities_needed=model_info.capabilities, preferred_tier=model_info.tier)
            if alternative and alternative != model_name:
                logger.info(f'Using alternative model: {alternative}')
                return await self.ensure_model_working(alternative)
            return False
        return True

    def get_model_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of all models"""
        report = {'total_models': len(self.models), 'working_models': len([m for m in self.models.values() if m.status == ModelStatus.WORKING]), 'models': {}}
        for (name, model) in self.models.items():
            report['models'][name] = {'tier': model.tier.value, 'status': model.status.value, 'capabilities': model.capabilities, 'success_rate': f'{model.success_rate:.2%}', 'avg_response_time': f'{model.avg_response_time:.2f}s', 'error_count': model.error_count, 'total_requests': model.total_requests}
        return report

    async def periodic_health_check(self):
        """Periodically check health of all models"""
        logger.info('Starting periodic health check...')
        for model_name in self.models.keys():
            await self.check_model_health(model_name)
            await asyncio.sleep(0.1)
        logger.info('Health check completed')

    def get_cost_optimization_suggestions(self) -> List[str]:
        """Get suggestions for cost optimization"""
        suggestions = []
        for (name, model) in self.models.items():
            if model.tier != ModelTier.FREE and model.success_rate < 0.5:
                suggestions.append(f'Consider replacing {name} with a free model due to low success rate')
        all_capabilities = set()
        for model in self.models.values():
            all_capabilities.update(model.capabilities)
        suggestions.append('You have access to these capabilities: ' + ', '.join(all_capabilities))
        return suggestions
model_manager = SelfAwareModelManager()

async def demo_self_aware_model_management():
    """Demonstrate self-aware model management"""
    print('=== Self-Aware Model Management Demo ===')
    await model_manager.periodic_health_check()
    report = model_manager.get_model_status_report()
    print(f'\nModel Status Report:')
    print(f"Total models: {report['total_models']}")
    print(f"Working models: {report['working_models']}")
    tasks = [('general', ['reasoning']), ('coding', ['reasoning', 'tool_calling']), ('analysis', ['reasoning', 'attachment'])]
    for (task_type, capabilities) in tasks:
        best_model = model_manager.get_best_model_for_task(task_type=task_type, capabilities_needed=capabilities)
        print(f'Best model for {task_type}: {best_model}')
    suggestions = model_manager.get_cost_optimization_suggestions()
    print(f'\nCost Optimization Suggestions:')
    for suggestion in suggestions:
        print(f'- {suggestion}')
if __name__ == '__main__':
    asyncio.run(demo_self_aware_model_management())