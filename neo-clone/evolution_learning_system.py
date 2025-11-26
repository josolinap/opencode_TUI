"""
Evolution Learning System - Advanced learning from evolution outcomes

This system analyzes the success/failure patterns of evolution attempts,
learns from outcomes, and improves future evolution decisions.
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class EvolutionOutcome:
    """Represents the outcome of an evolution attempt"""
    opportunity_id: str
    category: str
    priority: str
    success: bool
    implementation_time: float
    validation_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningPattern:
    """Represents a learned pattern from evolution outcomes"""
    pattern_id: str
    category: str
    success_rate: float
    avg_implementation_time: float
    common_characteristics: Dict[str, Any]
    recommended_actions: List[str]
    confidence_score: float
    sample_size: int
    last_updated: datetime = field(default_factory=datetime.now)

class EvolutionLearningSystem:
    """Advanced learning system for autonomous evolution"""

    def __init__(self, learning_file: str = 'evolution_learning_data.pkl'):
        self.learning_file = learning_file
        self.outcomes: List[EvolutionOutcome] = []
        self.patterns: Dict[str, LearningPattern] = {}
        self.category_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.learning_enabled = True

        # Load existing learning data
        self._load_learning_data()

    def record_outcome(self, outcome: EvolutionOutcome):
        """Record an evolution outcome for learning"""
        self.outcomes.append(outcome)

        # Update category statistics
        self._update_category_stats(outcome)

        # Learn new patterns
        self._learn_patterns()

        # Save learning data
        self._save_learning_data()

        logger.info(f"Recorded evolution outcome: {outcome.opportunity_id} ({'success' if outcome.success else 'failure'})")

    def get_category_success_rate(self, category: str) -> float:
        """Get success rate for a category"""
        stats = self.category_stats.get(category, {})
        total = stats.get('total_attempts', 0)
        if total == 0:
            return 0.5  # Default neutral rate

        successful = stats.get('successful_attempts', 0)
        return successful / total

    def predict_success_probability(self, category: str, priority: str, context: Dict[str, Any] = None) -> float:
        """Predict success probability for an opportunity"""
        base_rate = self.get_category_success_rate(category)

        # Adjust based on priority
        priority_multipliers = {'critical': 1.2, 'high': 1.1, 'medium': 1.0, 'low': 0.9}
        priority_multiplier = priority_multipliers.get(priority, 1.0)

        # Adjust based on context and learned patterns
        context_multiplier = self._calculate_context_multiplier(category, context or {})

        predicted_rate = base_rate * priority_multiplier * context_multiplier
        return max(0.0, min(1.0, predicted_rate))

    def get_recommended_priority(self, category: str, context: Dict[str, Any] = None) -> str:
        """Get recommended priority based on learning"""
        success_rate = self.get_category_success_rate(category)

        # Higher success rates can be lower priority (more routine)
        if success_rate > 0.8:
            return 'medium'
        elif success_rate > 0.6:
            return 'high'
        else:
            return 'critical'

    def get_implementation_time_estimate(self, category: str, priority: str) -> float:
        """Estimate implementation time based on learning"""
        stats = self.category_stats.get(category, {})
        avg_time = stats.get('avg_implementation_time', 30.0)  # Default 30 seconds

        # Adjust based on priority
        priority_multipliers = {'critical': 1.5, 'high': 1.2, 'medium': 1.0, 'low': 0.8}
        priority_multiplier = priority_multipliers.get(priority, 1.0)

        return avg_time * priority_multiplier

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights"""
        insights = {
            'total_outcomes': len(self.outcomes),
            'success_rate_overall': self._calculate_overall_success_rate(),
            'category_performance': dict(self.category_stats),
            'learned_patterns': len(self.patterns),
            'most_successful_categories': self._get_most_successful_categories(),
            'most_problematic_categories': self._get_most_problematic_categories(),
            'time_based_trends': self._analyze_time_based_trends(),
            'recommendations': self._generate_recommendations()
        }

        return insights

    def _update_category_stats(self, outcome: EvolutionOutcome):
        """Update statistics for a category"""
        category = outcome.category
        stats = self.category_stats[category]

        # Update counters
        stats['total_attempts'] = stats.get('total_attempts', 0) + 1
        if outcome.success:
            stats['successful_attempts'] = stats.get('successful_attempts', 0) + 1

        # Update timing
        if 'implementation_times' not in stats:
            stats['implementation_times'] = []
        stats['implementation_times'].append(outcome.implementation_time)

        # Keep only last 100 times for memory efficiency
        if len(stats['implementation_times']) > 100:
            stats['implementation_times'] = stats['implementation_times'][-100:]

        # Recalculate averages
        stats['avg_implementation_time'] = sum(stats['implementation_times']) / len(stats['implementation_times'])
        stats['success_rate'] = stats['successful_attempts'] / stats['total_attempts']

        # Track recent performance (last 10 attempts)
        if 'recent_outcomes' not in stats:
            stats['recent_outcomes'] = []
        stats['recent_outcomes'].append(outcome.success)

        if len(stats['recent_outcomes']) > 10:
            stats['recent_outcomes'] = stats['recent_outcomes'][-10:]

        stats['recent_success_rate'] = sum(stats['recent_outcomes']) / len(stats['recent_outcomes'])

    def _learn_patterns(self):
        """Learn patterns from outcomes"""
        if len(self.outcomes) < 5:
            return  # Need minimum data for pattern learning

        # Group outcomes by category
        category_outcomes = defaultdict(list)
        for outcome in self.outcomes:
            category_outcomes[outcome.category].append(outcome)

        # Learn patterns for each category
        for category, outcomes in category_outcomes.items():
            if len(outcomes) >= 3:  # Need at least 3 outcomes for pattern
                pattern = self._extract_pattern(category, outcomes)
                if pattern:
                    self.patterns[f"{category}_pattern"] = pattern

    def _extract_pattern(self, category: str, outcomes: List[EvolutionOutcome]) -> Optional[LearningPattern]:
        """Extract a learning pattern from outcomes"""
        successful_outcomes = [o for o in outcomes if o.success]
        failed_outcomes = [o for o in outcomes if not o.success]

        if not successful_outcomes:
            return None

        success_rate = len(successful_outcomes) / len(outcomes)
        avg_time = sum(o.implementation_time for o in successful_outcomes) / len(successful_outcomes)

        # Find common characteristics of successful outcomes
        common_chars = self._find_common_characteristics(successful_outcomes)

        # Generate recommendations based on pattern
        recommendations = self._generate_pattern_recommendations(category, success_rate, common_chars)

        # Calculate confidence based on sample size and consistency
        sample_size = len(outcomes)
        confidence = min(1.0, sample_size / 20.0) * (success_rate if success_rate > 0.5 else 0.5)

        pattern = LearningPattern(
            pattern_id=f"{category}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
            category=category,
            success_rate=success_rate,
            avg_implementation_time=avg_time,
            common_characteristics=common_chars,
            recommended_actions=recommendations,
            confidence_score=confidence,
            sample_size=sample_size
        )

        return pattern

    def _find_common_characteristics(self, outcomes: List[EvolutionOutcome]) -> Dict[str, Any]:
        """Find common characteristics among successful outcomes"""
        if not outcomes:
            return {}

        characteristics = {}

        # Analyze priorities
        priorities = [o.priority for o in outcomes]
        if priorities:
            priority_counts = Counter(priorities)
            characteristics['common_priority'] = priority_counts.most_common(1)[0][0]

        # Analyze implementation times
        times = [o.implementation_time for o in outcomes]
        if times:
            characteristics['avg_implementation_time'] = sum(times) / len(times)
            characteristics['fastest_time'] = min(times)
            characteristics['slowest_time'] = max(times)

        # Analyze context patterns
        context_keys = set()
        for outcome in outcomes:
            context_keys.update(outcome.context.keys())

        common_context = {}
        for key in context_keys:
            values = [o.context.get(key) for o in outcomes if key in o.context]
            if values and len(values) > len(outcomes) * 0.5:  # Present in >50% of outcomes
                value_counts = Counter(values)
                if value_counts:
                    common_context[key] = value_counts.most_common(1)[0][0]

        characteristics['common_context'] = common_context

        return characteristics

    def _generate_pattern_recommendations(self, category: str, success_rate: float, characteristics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on learned patterns"""
        recommendations = []

        if success_rate > 0.8:
            recommendations.append(f"High success rate ({success_rate:.1f}) for {category} - prioritize these opportunities")
        elif success_rate < 0.4:
            recommendations.append(f"Low success rate ({success_rate:.1f}) for {category} - review implementation approach")

        # Priority recommendations
        common_priority = characteristics.get('common_priority')
        if common_priority:
            recommendations.append(f"Most successful {category} opportunities have {common_priority} priority")

        # Time-based recommendations
        avg_time = characteristics.get('avg_implementation_time')
        if avg_time:
            if avg_time < 10:
                recommendations.append(f"{category} implementations are typically fast ({avg_time:.1f}s) - good for automation")
            elif avg_time > 60:
                recommendations.append(f"{category} implementations are typically slow ({avg_time:.1f}s) - allocate more time")

        return recommendations

    def _calculate_context_multiplier(self, category: str, context: Dict[str, Any]) -> float:
        """Calculate success multiplier based on context"""
        pattern = self.patterns.get(f"{category}_pattern")
        if not pattern:
            return 1.0

        multiplier = 1.0

        # Check common context
        common_context = pattern.common_characteristics.get('common_context', {})
        for key, expected_value in common_context.items():
            actual_value = context.get(key)
            if actual_value == expected_value:
                multiplier *= 1.1  # Slight boost for matching context
            elif actual_value is not None:
                multiplier *= 0.95  # Slight penalty for non-matching context

        return multiplier

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all outcomes"""
        if not self.outcomes:
            return 0.0

        successful = sum(1 for o in self.outcomes if o.success)
        return successful / len(self.outcomes)

    def _get_most_successful_categories(self) -> List[Tuple[str, float]]:
        """Get categories sorted by success rate"""
        category_rates = []
        for category, stats in self.category_stats.items():
            rate = stats.get('success_rate', 0)
            category_rates.append((category, rate))

        return sorted(category_rates, key=lambda x: x[1], reverse=True)[:5]

    def _get_most_problematic_categories(self) -> List[Tuple[str, float]]:
        """Get categories with lowest success rates"""
        category_rates = []
        for category, stats in self.category_stats.items():
            rate = stats.get('success_rate', 0)
            if stats.get('total_attempts', 0) >= 3:  # Only include categories with enough data
                category_rates.append((category, rate))

        return sorted(category_rates, key=lambda x: x[1])[:5]

    def _analyze_time_based_trends(self) -> Dict[str, Any]:
        """Analyze trends over time"""
        if len(self.outcomes) < 10:
            return {'insufficient_data': True}

        # Group by time periods
        recent_outcomes = [o for o in self.outcomes if (datetime.now() - o.timestamp).days <= 1]
        older_outcomes = [o for o in self.outcomes if (datetime.now() - o.timestamp).days > 1]

        recent_success = sum(1 for o in recent_outcomes if o.success) / len(recent_outcomes) if recent_outcomes else 0
        older_success = sum(1 for o in older_outcomes if o.success) / len(older_outcomes) if older_outcomes else 0

        trend = 'improving' if recent_success > older_success else 'declining' if recent_success < older_success else 'stable'

        return {
            'trend': trend,
            'recent_success_rate': recent_success,
            'older_success_rate': older_success,
            'improvement': recent_success - older_success
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate overall recommendations based on learning"""
        recommendations = []

        # Overall success rate
        overall_rate = self._calculate_overall_success_rate()
        if overall_rate > 0.8:
            recommendations.append("Evolution system performing excellently - consider expanding scope")
        elif overall_rate < 0.5:
            recommendations.append("Evolution system needs improvement - focus on implementation quality")

        # Category-specific recommendations
        problematic = self._get_most_problematic_categories()
        if problematic:
            worst_category = problematic[0][0]
            recommendations.append(f"Focus improvement efforts on {worst_category} category")

        # Time-based recommendations
        trends = self._analyze_time_based_trends()
        if trends.get('trend') == 'improving':
            recommendations.append("Evolution performance is improving over time")
        elif trends.get('trend') == 'declining':
            recommendations.append("Evolution performance is declining - review recent changes")

        return recommendations

    def _load_learning_data(self):
        """Load learning data from file"""
        try:
            with open(self.learning_file, 'rb') as f:
                data = pickle.load(f)
                self.outcomes = data.get('outcomes', [])
                self.patterns = data.get('patterns', {})
                self.category_stats = data.get('category_stats', defaultdict(dict))
                logger.info(f"Loaded learning data: {len(self.outcomes)} outcomes, {len(self.patterns)} patterns")
        except FileNotFoundError:
            logger.info("No existing learning data found - starting fresh")
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")

    def _save_learning_data(self):
        """Save learning data to file"""
        try:
            data = {
                'outcomes': self.outcomes[-1000:],  # Keep last 1000 outcomes
                'patterns': self.patterns,
                'category_stats': dict(self.category_stats),
                'last_updated': datetime.now()
            }

            with open(self.learning_file, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")

# Global learning system instance
evolution_learning = EvolutionLearningSystem()

def record_evolution_outcome(opportunity_id: str, category: str, priority: str,
                           success: bool, implementation_time: float,
                           validation_result: Optional[Dict[str, Any]] = None,
                           error_message: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None):
    """Record an evolution outcome for learning"""
    outcome = EvolutionOutcome(
        opportunity_id=opportunity_id,
        category=category,
        priority=priority,
        success=success,
        implementation_time=implementation_time,
        validation_result=validation_result,
        error_message=error_message,
        context=context or {}
    )

    evolution_learning.record_outcome(outcome)

def get_learning_insights():
    """Get current learning insights"""
    return evolution_learning.get_learning_insights()

def predict_opportunity_success(category: str, priority: str, context: Dict[str, Any] = None) -> float:
    """Predict success probability for an opportunity"""
    return evolution_learning.predict_success_probability(category, priority, context)

if __name__ == "__main__":
    print("Neo-Clone Evolution Learning System")
    print("Testing learning capabilities...")

    # Add some test outcomes
    test_outcomes = [
        EvolutionOutcome("test_1", "internal_improvement", "high", True, 15.5, context={"file_type": "python"}),
        EvolutionOutcome("test_2", "skill_enhancement", "medium", True, 8.2, context={"skill_type": "api"}),
        EvolutionOutcome("test_3", "internal_improvement", "high", False, 45.1, context={"file_type": "python"}),
        EvolutionOutcome("test_4", "libraries", "medium", True, 120.0, context={"language": "python"}),
        EvolutionOutcome("test_5", "internal_improvement", "high", True, 12.3, context={"file_type": "python"}),
    ]

    for outcome in test_outcomes:
        evolution_learning.record_outcome(outcome)

    # Get insights
    insights = evolution_learning.get_learning_insights()
    print(json.dumps(insights, indent=2, default=str))

    # Test predictions
    prediction = evolution_learning.predict_success_probability("internal_improvement", "high", {"file_type": "python"})
    print(f"Predicted success for internal_improvement (high priority, python file): {prediction:.2f}")