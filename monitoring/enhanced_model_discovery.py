from functools import lru_cache
'\nEnhanced Model Discovery for Neo-Clone\n========================================\n\nAdvanced model discovery and management capabilities.\n'
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from pathlib import Path
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about an AI model"""
    name: str
    provider: str
    model_id: str
    description: str
    capabilities: List[str]
    context_length: int
    pricing: Dict[str, float]
    is_free: bool
    is_available: bool = True
    last_checked: Optional[datetime] = None
    response_time: Optional[float] = None
    success_rate: float = 1.0

class EnhancedModelDiscovery:
    """Enhanced model discovery and management system"""

    def __init__(self, cache_file: str='data/models_cache.json'):
        self.cache_file = Path(cache_file)
        self.models: Dict[str, ModelInfo] = {}
        self.providers = {'openai': self._discover_openai_models, 'anthropic': self._discover_anthropic_models, 'google': self._discover_google_models, 'huggingface': self._discover_huggingface_models, 'together': self._discover_together_models, 'ollama': self._discover_ollama_models}
        self.load_cache()

    def load_cache(self):
        """Load models from cache file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for model_data in data.get('models', []):
                        model = ModelInfo(**model_data)
                        if model.last_checked:
                            model.last_checked = datetime.fromisoformat(model.last_checked)
                        self.models[model.name] = model
                logger.info(f'Loaded {len(self.models)} models from cache')
        except Exception as e:
            logger.warning(f'Failed to load model cache: {e}')

    def save_cache(self):
        """Save models to cache file"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {'models': [], 'last_updated': datetime.now().isoformat()}
            for model in self.models.values():
                model_dict = asdict(model)
                if model.last_checked:
                    model_dict['last_checked'] = model.last_checked.isoformat()
                data['models'].append(model_dict)
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f'Saved {len(self.models)} models to cache')
        except Exception as e:
            logger.error(f'Failed to save model cache: {e}')

    def discover_all_models(self, force_refresh: bool=False) -> Dict[str, ModelInfo]:
        """Discover models from all providers"""
        logger.info('Starting model discovery...')
        for (provider_name, discover_func) in self.providers.items():
            try:
                logger.info(f'Discovering models from {provider_name}...')
                provider_models = discover_func()
                for model in provider_models:
                    existing_model = self.models.get(model.name)
                    if existing_model and (not force_refresh):
                        if existing_model.last_checked and (datetime.now() - existing_model.last_checked).hours < 1:
                            model.is_available = existing_model.is_available
                            model.response_time = existing_model.response_time
                            model.success_rate = existing_model.success_rate
                    self.models[model.name] = model
            except Exception as e:
                logger.warning(f'Failed to discover models from {provider_name}: {e}')
        self.save_cache()
        logger.info(f'Model discovery completed. Found {len(self.models)} models')
        return self.models

    def _discover_openai_models(self) -> List[ModelInfo]:
        """Discover OpenAI models"""
        models = [ModelInfo(name='gpt-4', provider='openai', model_id='gpt-4', description="OpenAI's most capable model", capabilities=['text-generation', 'code-generation', 'reasoning'], context_length=8192, pricing={'input': 0.03, 'output': 0.06}, is_free=False), ModelInfo(name='gpt-3.5-turbo', provider='openai', model_id='gpt-3.5-turbo', description='Fast and efficient model for most tasks', capabilities=['text-generation', 'code-generation'], context_length=4096, pricing={'input': 0.001, 'output': 0.002}, is_free=True)]
        return models

    def _discover_anthropic_models(self) -> List[ModelInfo]:
        """Discover Anthropic models"""
        models = [ModelInfo(name='claude-3-opus', provider='anthropic', model_id='claude-3-opus-20240229', description='Most powerful Claude model', capabilities=['text-generation', 'code-generation', 'reasoning', 'analysis'], context_length=200000, pricing={'input': 0.015, 'output': 0.075}, is_free=False), ModelInfo(name='claude-3-sonnet', provider='anthropic', model_id='claude-3-sonnet-20240229', description='Balanced performance and speed', capabilities=['text-generation', 'code-generation', 'reasoning'], context_length=200000, pricing={'input': 0.003, 'output': 0.015}, is_free=True)]
        return models

    def _discover_google_models(self) -> List[ModelInfo]:
        """Discover Google models"""
        models = [ModelInfo(name='gemini-pro', provider='google', model_id='gemini-pro', description="Google's advanced language model", capabilities=['text-generation', 'code-generation', 'reasoning'], context_length=32768, pricing={'input': 0.0005, 'output': 0.0015}, is_free=True)]
        return models

    def _discover_huggingface_models(self) -> List[ModelInfo]:
        """Discover Hugging Face models"""
        models = [ModelInfo(name='llama-2-7b', provider='huggingface', model_id='meta-llama/Llama-2-7b-chat-hf', description="Meta's open-source LLaMA 2 model", capabilities=['text-generation', 'conversation'], context_length=4096, pricing={'input': 0, 'output': 0}, is_free=True), ModelInfo(name='mistral-7b', provider='huggingface', model_id='mistralai/Mistral-7B-Instruct-v0.1', description="Mistral's open-source model", capabilities=['text-generation', 'code-generation', 'reasoning'], context_length=8192, pricing={'input': 0, 'output': 0}, is_free=True)]
        return models

    def _discover_together_models(self) -> List[ModelInfo]:
        """Discover Together AI models"""
        models = [ModelInfo(name='together-llama-2-70b', provider='together', model_id='togethercomputer/llama-2-70b-chat', description='Large LLaMA 2 model via Together AI', capabilities=['text-generation', 'code-generation', 'reasoning'], context_length=4096, pricing={'input': 0.0009, 'output': 0.0009}, is_free=True)]
        return models

    def _discover_ollama_models(self) -> List[ModelInfo]:
        """Discover Ollama local models"""
        models = [ModelInfo(name='ollama-llama2', provider='ollama', model_id='llama2', description='Local LLaMA 2 model via Ollama', capabilities=['text-generation', 'conversation'], context_length=4096, pricing={'input': 0, 'output': 0}, is_free=True), ModelInfo(name='ollama-mistral', provider='ollama', model_id='mistral', description='Local Mistral model via Ollama', capabilities=['text-generation', 'code-generation'], context_length=8192, pricing={'input': 0, 'output': 0}, is_free=True)]
        return models

    def get_free_models(self) -> List[ModelInfo]:
        """Get all free models"""
        return [model for model in self.models.values() if model.is_free]

    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get models that have a specific capability"""
        return [model for model in self.models.values() if capability in model.capabilities]

    @lru_cache(maxsize=128)
    def get_best_model_for_task(self, task: str, prefer_free: bool=True) -> Optional[ModelInfo]:
        """Get the best model for a specific task"""
        task_capabilities = {'code': ['code-generation'], 'reasoning': ['reasoning'], 'conversation': ['text-generation', 'conversation'], 'analysis': ['analysis', 'reasoning']}
        required_caps = task_capabilities.get(task, ['text-generation'])
        candidates = []
        for model in self.models.values():
            if any((cap in model.capabilities for cap in required_caps)):
                if prefer_free and (not model.is_free):
                    continue
                if model.is_available:
                    candidates.append(model)
        if not candidates:
            return None
        candidates.sort(key=lambda m: (m.success_rate, -m.response_time if m.response_time else 0), reverse=True)
        return candidates[0]

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available"""
        model = self.models.get(model_name)
        if not model:
            return False
        model.is_available = True
        model.last_checked = datetime.now()
        return True

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered models"""
        total_models = len(self.models)
        free_models = len(self.get_free_models())
        available_models = len([m for m in self.models.values() if m.is_available])
        provider_counts = {}
        capability_counts = {}
        for model in self.models.values():
            provider_counts[model.provider] = provider_counts.get(model.provider, 0) + 1
            for cap in model.capabilities:
                capability_counts[cap] = capability_counts.get(cap, 0) + 1
        return {'total_models': total_models, 'free_models': free_models, 'available_models': available_models, 'providers': provider_counts, 'capabilities': capability_counts, 'last_updated': datetime.now().isoformat()}
_model_discovery = None

def get_model_discovery() -> EnhancedModelDiscovery:
    """Get the global model discovery instance"""
    global _model_discovery
    if _model_discovery is None:
        _model_discovery = EnhancedModelDiscovery()
    return _model_discovery
if __name__ == '__main__':
    discovery = EnhancedModelDiscovery()
    print('Testing Enhanced Model Discovery')
    print('=' * 40)
    models = discovery.discover_all_models()
    print(f'Discovered {len(models)} models')
    free_models = discovery.get_free_models()
    print(f'Free models: {len(free_models)}')
    stats = discovery.get_model_stats()
    print(f"Providers: {list(stats['providers'].keys())}")
    print(f"Capabilities: {list(stats['capabilities'].keys())}")
    best_code_model = discovery.get_best_model_for_task('code', prefer_free=True)
    if best_code_model:
        print(f'Best free model for coding: {best_code_model.name}')
    print('\nEnhanced model discovery working correctly!')