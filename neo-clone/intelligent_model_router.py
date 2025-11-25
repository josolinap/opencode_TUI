"""
Intelligent Model Selection and Routing System for Neo-Clone
Optimizes model selection based on task requirements, cost, and performance
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks for model routing"""
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    DATA_ANALYSIS = "data_analysis"
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    COMPLEX_PROBLEM_SOLVING = "complex_problem_solving"
    CREATIVE = "creative"
    MATHEMATICAL = "mathematical"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

class ModelTier(Enum):
    """Model performance tiers"""
    ULTRA_HIGH = "ultra_high"      # GPT-4, Claude-3.5
    HIGH = "high"                   # GPT-3.5-Turbo, Gemini Pro
    MEDIUM = "medium"               # Good local models
    FAST = "fast"                   # Lightweight models
    SPECIALIZED = "specialized"       # Task-specific models

@dataclass
class ModelCapability:
    """Model capability definition"""
    name: str
    tier: ModelTier
    cost_per_token: float
    max_tokens: int
    strengths: List[TaskType]
    weaknesses: List[TaskType]
    average_response_time: float  # seconds
    reliability_score: float  # 0-1
    supports_functions: bool = False
    supports_vision: bool = False
    supports_code_execution: bool = False
    context_window: int = 4096
    provider: str = "unknown"

@dataclass
class RoutingDecision:
    """Model routing decision"""
    selected_model: str
    confidence: float  # 0-1
    reasoning: str
    alternatives: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_time: float = 0.0
    fallback_available: bool = True

class IntelligentModelRouter:
    """Intelligent model selection and routing system"""
    
    def __init__(self):
        self.models: Dict[str, ModelCapability] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.cost_tracker: Dict[str, float] = {}
        self.last_update = None
        
        # Initialize model registry
        self._initialize_models()
        
        # Performance tracking
        self.routing_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, float] = {}
        
        # Adaptive learning
        self.learning_enabled = True
        self.adaptation_threshold = 10  # Minimum samples before adaptation
        
    def _initialize_models(self):
        """Initialize available models with capabilities"""
        
        # Ultra High Tier Models
        self.models["claude-3.5-sonnet"] = ModelCapability(
            name="claude-3.5-sonnet",
            tier=ModelTier.ULTRA_HIGH,
            cost_per_token=0.000015,
            max_tokens=200000,
            strengths=[TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.COMPLEX_PROBLEM_SOLVING],
            weaknesses=[TaskType.MATHEMATICAL],
            average_response_time=2.5,
            reliability_score=0.95,
            supports_functions=True,
            context_window=200000,
            provider="anthropic"
        )
        
        self.models["gpt-4-turbo"] = ModelCapability(
            name="gpt-4-turbo",
            tier=ModelTier.ULTRA_HIGH,
            cost_per_token=0.00001,
            max_tokens=128000,
            strengths=[TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.COMPLEX_PROBLEM_SOLVING, TaskType.MATHEMATICAL],
            weaknesses=[],
            average_response_time=3.0,
            reliability_score=0.94,
            supports_functions=True,
            supports_vision=True,
            context_window=128000,
            provider="openai"
        )
        
        # High Tier Models
        self.models["gpt-3.5-turbo"] = ModelCapability(
            name="gpt-3.5-turbo",
            tier=ModelTier.HIGH,
            cost_per_token=0.000002,
            max_tokens=16384,
            strengths=[TaskType.CODE_GENERATION, TaskType.CONVERSATION, TaskType.TRANSLATION],
            weaknesses=[TaskType.COMPLEX_PROBLEM_SOLVING],
            average_response_time=1.2,
            reliability_score=0.92,
            supports_functions=True,
            context_window=16384,
            provider="openai"
        )
        
        self.models["gemini-pro"] = ModelCapability(
            name="gemini-pro",
            tier=ModelTier.HIGH,
            cost_per_token=0.0000005,
            max_tokens=32768,
            strengths=[TaskType.REASONING, TaskType.DATA_ANALYSIS, TaskType.TRANSLATION],
            weaknesses=[TaskType.CODE_GENERATION],
            average_response_time=1.8,
            reliability_score=0.90,
            supports_functions=True,
            context_window=32768,
            provider="google"
        )
        
        # Medium Tier Models (Good Local Models)
        self.models["llama-3.1-8b"] = ModelCapability(
            name="llama-3.1-8b",
            tier=ModelTier.MEDIUM,
            cost_per_token=0.0,  # Free local model
            max_tokens=8192,
            strengths=[TaskType.CONVERSATION, TaskType.TEXT_ANALYSIS],
            weaknesses=[TaskType.COMPLEX_PROBLEM_SOLVING, TaskType.MATHEMATICAL],
            average_response_time=0.8,
            reliability_score=0.85,
            context_window=8192,
            provider="local"
        )
        
        self.models["mixtral-8x7b"] = ModelCapability(
            name="mixtral-8x7b",
            tier=ModelTier.MEDIUM,
            cost_per_token=0.0,
            max_tokens=32768,
            strengths=[TaskType.REASONING, TaskType.CODE_GENERATION],
            weaknesses=[TaskType.MATHEMATICAL],
            average_response_time=1.5,
            reliability_score=0.88,
            context_window=32768,
            provider="local"
        )
        
        # Fast Tier Models
        self.models["phi-3-mini"] = ModelCapability(
            name="phi-3-mini",
            tier=ModelTier.FAST,
            cost_per_token=0.0,
            max_tokens=4096,
            strengths=[TaskType.CONVERSATION, TaskType.SUMMARIZATION],
            weaknesses=[TaskType.COMPLEX_PROBLEM_SOLVING],
            average_response_time=0.3,
            reliability_score=0.80,
            context_window=4096,
            provider="local"
        )
        
        # Initialize usage stats
        for model_name in self.models:
            self.usage_stats[model_name] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_cost': 0.0,
                'total_time': 0.0,
                'last_used': None
            }
            self.performance_history[model_name] = []
            self.cost_tracker[model_name] = 0.0
            
        self.last_update = datetime.now()
        logger.info(f"Initialized {len(self.models)} models for intelligent routing")
        
    def route_request(self, 
                     task_description: str,
                     task_type: Optional[TaskType] = None,
                     user_preferences: Optional[Dict[str, Any]] = None,
                     budget_constraint: Optional[float] = None,
                     time_constraint: Optional[float] = None) -> RoutingDecision:
        """
        Route request to optimal model
        
        Args:
            task_description: Description of the task
            task_type: Type of task (optional, will be inferred)
            user_preferences: User preferences for model selection
            budget_constraint: Maximum cost constraint
            time_constraint: Maximum time constraint
            
        Returns:
            RoutingDecision with selected model and reasoning
        """
        
        start_time = time.time()
        
        # Infer task type if not provided
        if task_type is None:
            task_type = self._infer_task_type(task_description)
            
        logger.info(f"Routing request: {task_type.value} - {task_description[:100]}...")
        
        # Filter models by constraints
        candidate_models = self._filter_models(task_type, budget_constraint, time_constraint)
        
        if not candidate_models:
            # Fallback to any available model
            candidate_models = list(self.models.keys())
            logger.warning("No models match constraints, using fallback")
            
        # Score and rank models
        scored_models = self._score_models(candidate_models, task_type, task_description, user_preferences)
        
        # Select best model
        if not scored_models:
            return RoutingDecision(
                selected_model="claude-3.5-sonnet",  # Default fallback
                confidence=0.1,
                reasoning="No suitable models found, using default fallback",
                estimated_cost=0.0,
                estimated_time=0.0,
                fallback_available=True
            )
            
        best_model_name, best_score = scored_models[0]
        best_model = self.models[best_model_name]
        
        # Calculate estimates
        estimated_tokens = self._estimate_tokens(task_description)
        estimated_cost = estimated_tokens * best_model.cost_per_token
        estimated_time = best_model.average_response_time
        
        # Get alternatives
        alternatives = [name for name, score in scored_models[1:4] if score > 0.3]
        
        # Create decision
        decision = RoutingDecision(
            selected_model=best_model_name,
            confidence=min(best_score, 1.0),
            reasoning=self._generate_reasoning(best_model, task_type, best_score),
            alternatives=alternatives,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            fallback_available=len(alternatives) > 0
        )
        
        # Record routing decision
        self._record_routing_decision(task_description, task_type, decision)
        
        routing_time = time.time() - start_time
        logger.info(f"Routed to {best_model_name} (confidence: {decision.confidence:.2f}) in {routing_time:.3f}s")
        
        return decision
        
    def _infer_task_type(self, task_description: str) -> TaskType:
        """Infer task type from description"""
        description_lower = task_description.lower()
        
        # Keywords for each task type
        task_keywords = {
            TaskType.CODE_GENERATION: ['code', 'function', 'class', 'program', 'script', 'algorithm', 'debug', 'implement'],
            TaskType.TEXT_ANALYSIS: ['analyze', 'sentiment', 'extract', 'parse', 'text', 'language'],
            TaskType.DATA_ANALYSIS: ['data', 'analyze', 'statistics', 'dataset', 'csv', 'json', 'metrics'],
            TaskType.REASONING: ['reason', 'logic', 'solve', 'problem', 'think', 'explain'],
            TaskType.CONVERSATION: ['chat', 'talk', 'discuss', 'conversation', 'hello', 'hi'],
            TaskType.COMPLEX_PROBLEM_SOLVING: ['complex', 'challenge', 'difficult', 'advanced', 'expert'],
            TaskType.CREATIVE: ['creative', 'write', 'story', 'poem', 'imagine', 'design'],
            TaskType.MATHEMATICAL: ['math', 'calculate', 'equation', 'formula', 'number', 'statistics'],
            TaskType.TRANSLATION: ['translate', 'language', 'convert', 'from', 'to'],
            TaskType.SUMMARIZATION: ['summarize', 'summary', 'brief', 'condense', 'shorten']
        }
        
        # Score each task type
        scores = {}
        for task_type, keywords in task_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            scores[task_type] = score
            
        # Return highest scoring task type
        if scores:
            return max(scores, key=scores.get)
            
        # Default to conversation
        return TaskType.CONVERSATION
        
    def _filter_models(self, 
                     task_type: TaskType, 
                     budget_constraint: Optional[float], 
                     time_constraint: Optional[float]) -> List[str]:
        """Filter models based on constraints"""
        
        candidates = []
        
        for model_name, model in self.models.items():
            # Check if model is good at this task type
            if task_type in model.weaknesses:
                continue  # Skip models weak at this task
                
            # Check budget constraint
            if budget_constraint is not None:
                estimated_cost = self._estimate_tokens("average task") * model.cost_per_token
                if estimated_cost > budget_constraint:
                    continue
                    
            # Check time constraint
            if time_constraint is not None:
                if model.average_response_time > time_constraint:
                    continue
                    
            candidates.append(model_name)
            
        return candidates
        
    def _score_models(self, 
                     candidate_models: List[str], 
                     task_type: TaskType, 
                     task_description: str,
                     user_preferences: Optional[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Score and rank candidate models"""
        
        scores = []
        
        for model_name in candidate_models:
            model = self.models[model_name]
            score = 0.0
            
            # Task compatibility score (0-0.4)
            if task_type in model.strengths:
                score += 0.4
            elif task_type not in model.weaknesses:
                score += 0.2
                
            # Performance score (0-0.2)
            score += model.reliability_score * 0.2
            
            # Speed score (0-0.15)
            # Faster models get higher scores for time-sensitive tasks
            speed_score = max(0, 0.15 - (model.average_response_time / 20))
            score += speed_score
            
            # Cost efficiency score (0-0.15)
            if model.cost_per_token == 0:
                score += 0.15  # Free models get bonus
            else:
                cost_efficiency = max(0, 0.15 - (model.cost_per_token * 1000))
                score += cost_efficiency
                
            # Historical performance score (0-0.1)
            hist_score = self.success_rates.get(model_name, 0.5)
            score += hist_score * 0.1
            
            # User preferences (0-0.1)
            if user_preferences:
                if user_preferences.get('prefer_free', False) and model.cost_per_token == 0:
                    score += 0.1
                if user_preferences.get('prefer_fast', False) and model.average_response_time < 1.0:
                    score += 0.1
                if user_preferences.get('prefer_high_quality', False) and model.tier == ModelTier.ULTRA_HIGH:
                    score += 0.1
                    
            scores.append((model_name, score))
            
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4
        
    def _generate_reasoning(self, 
                          model: ModelCapability, 
                          task_type: TaskType, 
                          score: float) -> str:
        """Generate reasoning for model selection"""
        
        reasons = []
        
        if task_type in model.strengths:
            reasons.append(f"Excels at {task_type.value} tasks")
            
        if model.tier == ModelTier.ULTRA_HIGH:
            reasons.append("Highest quality and reasoning capabilities")
        elif model.tier == ModelTier.HIGH:
            reasons.append("Good balance of quality and speed")
        elif model.tier == ModelTier.FAST:
            reasons.append("Fast response time")
            
        if model.cost_per_token == 0:
            reasons.append("No cost (free model)")
        elif model.cost_per_token < 0.000001:
            reasons.append("Very cost effective")
            
        if model.reliability_score > 0.9:
            reasons.append("High reliability")
            
        if score > 0.7:
            return f"Selected for strong match: {', '.join(reasons[:3])}"
        elif score > 0.5:
            return f"Good match: {', '.join(reasons[:2])}"
        else:
            return f"Adequate match: {reasons[0] if reasons else 'Default selection'}"
            
    def _record_routing_decision(self, 
                             task_description: str, 
                             task_type: TaskType, 
                             decision: RoutingDecision):
        """Record routing decision for learning"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type.value,
            'task_description': task_description[:200],
            'selected_model': decision.selected_model,
            'confidence': decision.confidence,
            'estimated_cost': decision.estimated_cost,
            'alternatives': decision.alternatives
        }
        
        self.routing_history.append(record)
        
        # Keep history manageable
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]
            
    def record_model_performance(self, 
                              model_name: str, 
                              success: bool, 
                              actual_time: float, 
                              actual_cost: float,
                              user_satisfaction: Optional[float] = None):
        """Record actual model performance for learning"""
        
        if model_name not in self.models:
            return
            
        # Update usage stats
        stats = self.usage_stats[model_name]
        stats['total_requests'] += 1
        stats['total_time'] += actual_time
        stats['total_cost'] += actual_cost
        stats['last_used'] = datetime.now()
        
        if success:
            stats['successful_requests'] += 1
            
        # Update performance history
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'response_time': actual_time,
            'cost': actual_cost,
            'user_satisfaction': user_satisfaction
        }
        
        self.performance_history[model_name].append(performance_record)
        
        # Keep history manageable
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-50:]
            
        # Update success rate
        if stats['total_requests'] >= self.adaptation_threshold:
            self.success_rates[model_name] = stats['successful_requests'] / stats['total_requests']
            
        # Adaptive learning: update model capabilities based on performance
        if self.learning_enabled and len(self.performance_history[model_name]) >= 20:
            self._adaptive_learning_update(model_name)
            
        logger.debug(f"Recorded performance for {model_name}: success={success}, time={actual_time:.2f}s")
        
    def _adaptive_learning_update(self, model_name: str):
        """Update model capabilities based on observed performance"""
        
        if model_name not in self.performance_history:
            return
            
        recent_performance = self.performance_history[model_name][-20:]
        
        # Calculate actual vs expected performance
        actual_avg_time = sum(p['response_time'] for p in recent_performance) / len(recent_performance)
        expected_time = self.models[model_name].average_response_time
        
        # Update response time if consistently different
        if actual_avg_time > expected_time * 1.5:
            # Model is slower than expected
            self.models[model_name].average_response_time = actual_avg_time
            logger.info(f"Updated {model_name} response time: {expected_time:.2f}s -> {actual_avg_time:.2f}s")
        elif actual_avg_time < expected_time * 0.7:
            # Model is faster than expected
            self.models[model_name].average_response_time = actual_avg_time
            logger.info(f"Updated {model_name} response time: {expected_time:.2f}s -> {actual_avg_time:.2f}s")
            
        # Update reliability based on success rate
        success_rate = sum(1 for p in recent_performance if p['success']) / len(recent_performance)
        if abs(success_rate - self.models[model_name].reliability_score) > 0.1:
            self.models[model_name].reliability_score = success_rate
            logger.info(f"Updated {model_name} reliability: {self.models[model_name].reliability_score:.2f}")
            
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        total_requests = sum(stats['total_requests'] for stats in self.usage_stats.values())
        total_cost = sum(stats['total_cost'] for stats in self.usage_stats.values())
        
        most_used_model = max(self.usage_stats.items(), key=lambda x: x[1]['total_requests']) if self.usage_stats else None
        
        return {
            'total_requests': total_requests,
            'total_cost': total_cost,
            'models_available': len(self.models),
            'most_used_model': most_used_model[0] if most_used_model else None,
            'average_confidence': sum(r['confidence'] for r in self.routing_history[-100:]) / len(self.routing_history[-100:]) if self.routing_history else 0.0,
            'routing_accuracy': self._calculate_routing_accuracy(),
            'cost_savings': self._calculate_cost_savings(),
            'performance_improvements': self._calculate_performance_improvements()
        }
        
    def _calculate_routing_accuracy(self) -> float:
        """Calculate routing accuracy based on user satisfaction"""
        
        satisfied_requests = 0
        total_requests = 0
        
        for model_name, history in self.performance_history.items():
            for record in history:
                if record.get('user_satisfaction') is not None:
                    total_requests += 1
                    if record['user_satisfaction'] > 0.7:
                        satisfied_requests += 1
                        
        return satisfied_requests / max(total_requests, 1)
        
    def _calculate_cost_savings(self) -> float:
        """Calculate cost savings from intelligent routing"""
        
        # Compare actual costs vs if we always used the most expensive model
        most_expensive_cost = max(model.cost_per_token for model in self.models.values())
        actual_total_cost = sum(stats['total_cost'] for stats in self.usage_stats.values())
        
        # Estimate what it would have cost with most expensive model
        estimated_requests = sum(stats['total_requests'] for stats in self.usage_stats.values())
        estimated_tokens_per_request = 100  # Rough estimate
        would_have_cost = estimated_requests * estimated_tokens_per_request * most_expensive_cost
        
        return would_have_cost - actual_total_cost
        
    def _calculate_performance_improvements(self) -> Dict[str, float]:
        """Calculate performance improvements from routing"""
        
        improvements = {}
        
        for model_name, model in self.models.items():
            if model_name in self.performance_history and len(self.performance_history[model_name]) >= 10:
                recent_performance = self.performance_history[model_name][-10:]
                actual_avg_time = sum(p['response_time'] for p in recent_performance) / len(recent_performance)
                
                if actual_avg_time < model.average_response_time:
                    improvement = (model.average_response_time - actual_avg_time) / model.average_response_time
                    improvements[model_name] = improvement
                    
        return improvements
        
    def get_model_recommendations(self, task_type: TaskType) -> List[Dict[str, Any]]:
        """Get model recommendations for specific task type"""
        
        recommendations = []
        
        for model_name, model in self.models.items():
            if task_type in model.strengths:
                score = 0.8  # High base score for strengths
            elif task_type not in model.weaknesses:
                score = 0.5  # Medium score for neutral
            else:
                score = 0.2  # Low score for weaknesses
                
            recommendations.append({
                'model': model_name,
                'score': score,
                'tier': model.tier.value,
                'cost_per_token': model.cost_per_token,
                'average_response_time': model.average_response_time,
                'reliability': model.reliability_score,
                'reasons': self._generate_reasoning(model, task_type, score)
            })
            
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]  # Top 5 recommendations

# Global router instance
router = IntelligentModelRouter()

def get_router() -> IntelligentModelRouter:
    """Get the global model router instance"""
    return router

def route_request(task_description: str, 
                 task_type: Optional[TaskType] = None,
                 user_preferences: Optional[Dict[str, Any]] = None,
                 budget_constraint: Optional[float] = None,
                 time_constraint: Optional[float] = None) -> RoutingDecision:
    """Route a request to the optimal model"""
    return router.route_request(task_description, task_type, user_preferences, budget_constraint, time_constraint)