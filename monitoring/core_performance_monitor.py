from functools import lru_cache
'\nCore Performance Monitor for Neo-Clone\n=======================================\n\nProvides performance monitoring capabilities for the Neo-Clone system.\n'
import logging
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    unit: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    name: str
    count: int = 0
    total: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    average: float = 0.0
    recent_average: float = 0.0
    unit: str = ''

    def update(self, value: float):
        """Update statistics with new value"""
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.average = self.total / self.count

class CorePerformanceMonitor:
    """Core performance monitoring system for Neo-Clone"""

    def __init__(self, max_history: int=10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda : deque(maxlen=max_history))
        self.stats: Dict[str, PerformanceStats] = {}
        self.lock = threading.RLock()
        self.start_time = datetime.now()
        self.categories = {'skill_execution': [], 'brain_operations': [], 'memory_operations': [], 'model_calls': [], 'system_resources': []}

    def record_metric(self, name: str, value: float, unit: str='', category: str='general', metadata: Dict[str, Any]=None):
        """Record a performance metric"""
        with self.lock:
            timestamp = datetime.now()
            metric = PerformanceMetric(name=name, value=value, timestamp=timestamp, unit=unit, metadata=metadata or {})
            self.metrics[name].append(metric)
            if name not in self.stats:
                self.stats[name] = PerformanceStats(name=name, unit=unit)
            self.stats[name].update(value)
            if category in self.categories:
                self.categories[category].append(metric)
            logger.debug(f'Recorded metric: {name} = {value} {unit}')

    def record_execution_time(self, operation: str, execution_time: float, metadata: Dict[str, Any]=None):
        """Record execution time for an operation"""
        self.record_metric(name=f'{operation}_execution_time', value=execution_time, unit='seconds', category='skill_execution', metadata=metadata)

    def record_memory_usage(self, component: str, memory_mb: float):
        """Record memory usage for a component"""
        self.record_metric(name=f'{component}_memory_usage', value=memory_mb, unit='MB', category='system_resources', metadata={'component': component})

    def record_model_call(self, model_name: str, response_time: float, tokens: int=None, success: bool=True):
        """Record model API call metrics"""
        metadata = {'model': model_name, 'success': success}
        if tokens:
            metadata['tokens'] = tokens
        self.record_metric(name=f'{model_name}_response_time', value=response_time, unit='seconds', category='model_calls', metadata=metadata)

    def get_statistics(self, name: str) -> Optional[PerformanceStats]:
        """Get statistics for a specific metric"""
        with self.lock:
            return self.stats.get(name)

    def get_recent_metrics(self, name: str, count: int=100) -> List[PerformanceMetric]:
        """Get recent metrics for a specific name"""
        with self.lock:
            metrics = list(self.metrics.get(name, []))
            return metrics[-count:] if metrics else []

    def get_average_execution_time(self, operation: str, recent_only: bool=False) -> Optional[float]:
        """Get average execution time for an operation"""
        stats = self.get_statistics(f'{operation}_execution_time')
        if stats:
            return stats.recent_average if recent_only else stats.average
        return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            uptime = datetime.now() - self.start_time
            summary = {'uptime_seconds': uptime.total_seconds(), 'uptime_formatted': str(uptime).split('.')[0], 'total_metrics': sum((len(metrics) for metrics in self.metrics.values())), 'metric_types': len(self.metrics), 'categories': {}}
            for (category, metrics) in self.categories.items():
                if metrics:
                    recent_metrics = metrics[-100:]
                    summary['categories'][category] = {'total_count': len(metrics), 'recent_count': len(recent_metrics), 'latest_metric': recent_metrics[-1].timestamp.isoformat() if recent_metrics else None}
            top_metrics = sorted([(name, len(metrics)) for (name, metrics) in self.metrics.items()], key=lambda x: x[1], reverse=True)[:10]
            summary['top_metrics'] = [{'name': name, 'count': count} for (name, count) in top_metrics]
            summary['performance_stats'] = {}
            for (name, stats) in self.stats.items():
                if stats.count > 0:
                    summary['performance_stats'][name] = {'count': stats.count, 'average': stats.average, 'min': stats.min_value, 'max': stats.max_value, 'unit': stats.unit}
            return summary

    def export_metrics(self, filename: str, format: str='json'):
        """Export metrics to file"""
        with self.lock:
            data = {'export_timestamp': datetime.now().isoformat(), 'summary': self.get_performance_summary(), 'metrics': {}}
            for (name, metric_deque) in self.metrics.items():
                data['metrics'][name] = [{'value': m.value, 'timestamp': m.timestamp.isoformat(), 'unit': m.unit, 'metadata': m.metadata} for m in metric_deque]
            with open(filename, 'w') as f:
                if format.lower() == 'json':
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(f'Unsupported export format: {format}')
            logger.info(f'Metrics exported to {filename}')

    def clear_metrics(self, name: str=None):
        """Clear metrics for a specific name or all metrics"""
        with self.lock:
            if name:
                if name in self.metrics:
                    del self.metrics[name]
                if name in self.stats:
                    del self.stats[name]
                logger.info(f'Cleared metrics for {name}')
            else:
                self.metrics.clear()
                self.stats.clear()
                for category in self.categories:
                    self.categories[category] = []
                logger.info('Cleared all metrics')

    def get_metrics_by_time_range(self, name: str, start_time: datetime, end_time: datetime) -> List[PerformanceMetric]:
        """Get metrics within a time range"""
        with self.lock:
            metrics = self.metrics.get(name, [])
            return [m for m in metrics if start_time <= m.timestamp <= end_time]

    def calculate_percentile(self, name: str, percentile: float=95) -> Optional[float]:
        """Calculate percentile value for a metric"""
        with self.lock:
            metrics = list(self.metrics.get(name, []))
            if not metrics:
                return None
            values = [m.value for m in metrics]
            values.sort()
            index = int(len(values) * percentile / 100)
            return values[min(index, len(values) - 1)]
_performance_monitor = None

def get_performance_monitor() -> CorePerformanceMonitor:
    """Get the global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = CorePerformanceMonitor()
    return _performance_monitor

def record_metric(name: str, value: float, unit: str='', category: str='general', metadata: Dict[str, Any]=None):
    """Convenience function to record a metric"""
    monitor = get_performance_monitor()
    monitor.record_metric(name, value, unit, category, metadata)

def record_execution_time(operation: str, execution_time: float, metadata: Dict[str, Any]=None):
    """Convenience function to record execution time"""
    monitor = get_performance_monitor()
    monitor.record_execution_time(operation, execution_time, metadata)

@lru_cache(maxsize=128)
def get_performance_summary() -> Dict[str, Any]:
    """Convenience function to get performance summary"""
    monitor = get_performance_monitor()
    return monitor.get_performance_summary()
if __name__ == '__main__':
    monitor = CorePerformanceMonitor()
    print('Testing Core Performance Monitor')
    print('=' * 40)
    for i in range(10):
        monitor.record_execution_time('test_operation', 0.1 + i * 0.01)
        monitor.record_metric('test_metric', i * 10, 'units')
    summary = monitor.get_performance_summary()
    print(f"Total metrics: {summary['total_metrics']}")
    print(f"Metric types: {summary['metric_types']}")
    print(f"Uptime: {summary['uptime_formatted']}")
    stats = monitor.get_statistics('test_operation_execution_time')
    if stats:
        print(f'Test operation stats:')
        print(f'  Count: {stats.count}')
        print(f'  Average: {stats.average:.3f}s')
        print(f'  Min: {stats.min_value:.3f}s')
        print(f'  Max: {stats.max_value:.3f}s')
    print('\nPerformance monitor working correctly!')