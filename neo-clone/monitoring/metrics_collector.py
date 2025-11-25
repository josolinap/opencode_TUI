"""
Metrics Collection System for Neo-Clone

This module provides comprehensive metrics collection and monitoring capabilities
for the Neo-Clone system, including performance metrics, resource usage,
and custom business metrics.

Features:
- Real-time metrics collection
- Time-series data aggregation
- Custom metric definitions
- Alerting and threshold monitoring
- Export capabilities for external monitoring systems
"""

import asyncio
import time
import json
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"           # Incrementing counter
    GAUGE = "gauge"              # Current value
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Rate per time unit


class MetricUnit(Enum):
    """Units for metrics"""
    COUNT = "count"
    MILLISECONDS = "milliseconds"
    SECONDS = "seconds"
    BYTES = "bytes"
    PERCENTAGE = "percentage"
    REQUESTS_PER_SECOND = "requests_per_second"
    OPERATIONS_PER_SECOND = "operations_per_second"


@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    unit: MetricUnit
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation_method: Optional[str] = None  # sum, avg, min, max, p95, p99
    retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=dict)  # warning, critical


@dataclass
class MetricSnapshot:
    """Snapshot of metric at a point in time"""
    metric_name: str
    timestamp: datetime
    value: Union[int, float, List[float]]
    count: int = 1
    sum_value: float = 0.0
    min_value: float = float('inf')
    max_value: float = 0.0
    avg_value: float = 0.0
    p95_value: float = 0.0
    p99_value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "count": self.count,
            "sum_value": self.sum_value,
            "min_value": self.min_value if self.min_value != float('inf') else 0.0,
            "max_value": self.max_value,
            "avg_value": self.avg_value,
            "p95_value": self.p95_value,
            "p99_value": self.p99_value,
            "tags": self.tags
        }


class Alert:
    """Alert definition and state"""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: str,  # gt, lt, eq, gte, lte
        threshold: float,
        severity: str = "warning",  # info, warning, critical
        duration_seconds: int = 300,  # How long condition must persist
        tags: Dict[str, str] = None
    ):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.duration_seconds = duration_seconds
        self.tags = tags or {}
        
        self.triggered_at: Optional[datetime] = None
        self.resolved_at: Optional[datetime] = None
        self.is_active = False
        self.notification_count = 0
        self.last_notification: Optional[datetime] = None
    
    def check_condition(self, value: float) -> bool:
        """Check if alert condition is met"""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity,
            "duration_seconds": self.duration_seconds,
            "tags": self.tags,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "is_active": self.is_active,
            "notification_count": self.notification_count,
            "last_notification": self.last_notification.isoformat() if self.last_notification else None
        }


class MetricsCollector:
    """
    Comprehensive metrics collection system for Neo-Clone
    
    Collects, aggregates, and manages various types of metrics with
    support for real-time monitoring and alerting.
    """
    
    def __init__(
        self,
        max_data_points: int = 10000,
        cleanup_interval_seconds: int = 3600,
        aggregation_interval_seconds: int = 60
    ):
        """
        Initialize metrics collector
        
        Args:
            max_data_points: Maximum data points to keep in memory
            cleanup_interval_seconds: Interval for cleanup operations
            aggregation_interval_seconds: Interval for metric aggregation
        """
        self.max_data_points = max_data_points
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.aggregation_interval_seconds = aggregation_interval_seconds
        
        # Metric storage
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_data_points))
        self.aggregated_metrics: Dict[str, MetricSnapshot] = {}
        
        # Alert management
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info("Metrics collector initialized")
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default metric definitions"""
        default_metrics = [
            MetricDefinition(
                name="brain_processing_duration",
                metric_type=MetricType.HISTOGRAM,
                unit=MetricUnit.MILLISECONDS,
                description="Duration of brain processing operations",
                aggregation_method="avg"
            ),
            MetricDefinition(
                name="skill_execution_count",
                metric_type=MetricType.COUNTER,
                unit=MetricUnit.COUNT,
                description="Number of skill executions"
            ),
            MetricDefinition(
                name="memory_access_duration",
                metric_type=MetricType.HISTOGRAM,
                unit=MetricUnit.MILLISECONDS,
                description="Duration of memory access operations"
            ),
            MetricDefinition(
                name="active_sessions",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.COUNT,
                description="Number of active sessions"
            ),
            MetricDefinition(
                name="error_rate",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENTAGE,
                description="Error rate percentage"
            ),
            MetricDefinition(
                name="request_rate",
                metric_type=MetricType.RATE,
                unit=MetricUnit.REQUESTS_PER_SECOND,
                description="Requests per second"
            ),
            MetricDefinition(
                name="cache_hit_rate",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENTAGE,
                description="Cache hit rate percentage"
            ),
            MetricDefinition(
                name="cpu_usage",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENTAGE,
                description="CPU usage percentage"
            ),
            MetricDefinition(
                name="memory_usage",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.BYTES,
                description="Memory usage in bytes"
            )
        ]
        
        for metric_def in default_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition) -> None:
        """Register a new metric definition"""
        with self._lock:
            self.metric_definitions[metric_def.name] = metric_def
            logger.debug(f"Registered metric: {metric_def.name}")
    
    def record_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Additional tags for the metric
        """
        if metric_name not in self.metric_definitions:
            logger.warning(f"Metric {metric_name} not registered, auto-registering as gauge")
            self.register_metric(MetricDefinition(
                name=metric_name,
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.COUNT,
                description=f"Auto-registered metric: {metric_name}"
            ))
        
        metric_value = MetricValue(
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self._lock:
            self.metric_data[metric_name].append(metric_value)
        
        # Check alerts
        self._check_alerts(metric_name, value)
    
    def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric"""
        self.record_metric(metric_name, value, tags)
    
    def set_gauge(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value"""
        self.record_metric(metric_name, value, tags)
    
    def record_timer(
        self,
        metric_name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timer/duration metric"""
        self.record_metric(metric_name, duration_ms, tags)
    
    def record_histogram(
        self,
        metric_name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram metric value"""
        self.record_metric(metric_name, value, tags)
    
    def get_metric_data(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MetricValue]:
        """
        Get metric data for a specific metric
        
        Args:
            metric_name: Name of the metric
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of data points
            
        Returns:
            List of metric values
        """
        with self._lock:
            data = list(self.metric_data.get(metric_name, []))
        
        # Apply time filters
        if start_time:
            data = [d for d in data if d.timestamp >= start_time]
        if end_time:
            data = [d for d in data if d.timestamp <= end_time]
        
        # Apply limit
        return data[-limit:] if len(data) > limit else data
    
    def get_aggregated_metric(self, metric_name: str) -> Optional[MetricSnapshot]:
        """Get aggregated metric snapshot"""
        with self._lock:
            return self.aggregated_metrics.get(metric_name)
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "total_metrics": len(self.metric_definitions),
            "total_data_points": sum(len(data) for data in self.metric_data.values()),
            "active_alerts": len(self.active_alerts),
            "metrics": {}
        }
        
        for metric_name, metric_def in self.metric_definitions.items():
            data = self.metric_data.get(metric_name, [])
            if data:
                values = [d.value for d in data if isinstance(d.value, (int, float))]
                if values:
                    summary["metrics"][metric_name] = {
                        "type": metric_def.metric_type.value,
                        "unit": metric_def.unit.value,
                        "data_points": len(data),
                        "latest_value": values[-1] if values else None,
                        "min_value": min(values),
                        "max_value": max(values),
                        "avg_value": statistics.mean(values),
                        "last_updated": data[-1].timestamp.isoformat() if data else None
                    }
        
        return summary
    
    def create_alert(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
        duration_seconds: int = 300,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Create a new alert"""
        alert = Alert(
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            duration_seconds=duration_seconds,
            tags=tags
        )
        
        with self._lock:
            self.alerts[name] = alert
        
        logger.info(f"Created alert: {name} for metric {metric_name}")
    
    def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check if any alerts should be triggered"""
        with self._lock:
            for alert in self.alerts.values():
                if alert.metric_name == metric_name:
                    condition_met = alert.check_condition(value)
                    
                    if condition_met and not alert.is_active:
                        # Alert triggered
                        alert.triggered_at = datetime.now()
                        alert.is_active = True
                        self.active_alerts[alert.name] = alert
                        self._trigger_alert_notification(alert, value)
                    
                    elif not condition_met and alert.is_active:
                        # Alert resolved
                        alert.resolved_at = datetime.now()
                        alert.is_active = False
                        self.active_alerts.pop(alert.name, None)
                        self._resolve_alert_notification(alert, value)
    
    def _trigger_alert_notification(self, alert: Alert, value: float) -> None:
        """Trigger alert notification"""
        alert.notification_count += 1
        alert.last_notification = datetime.now()
        
        logger.warning(
            f"ALERT TRIGGERED: {alert.name} - {alert.metric_name} = {value} "
            f"(threshold: {alert.threshold} {alert.condition})"
        )
        
        # Here you could integrate with external notification systems
        # (Slack, email, PagerDuty, etc.)
    
    def _resolve_alert_notification(self, alert: Alert, value: float) -> None:
        """Resolve alert notification"""
        logger.info(
            f"ALERT RESOLVED: {alert.name} - {alert.metric_name} = {value} "
            f"(threshold: {alert.threshold} {alert.condition})"
        )
    
    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics periodically"""
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval_seconds)
                
                with self._lock:
                    for metric_name, data_points in self.metric_data.items():
                        if not data_points:
                            continue
                        
                        metric_def = self.metric_definitions.get(metric_name)
                        if not metric_def:
                            continue
                        
                        # Get recent data points (last aggregation interval)
                        cutoff_time = datetime.now() - timedelta(seconds=self.aggregation_interval_seconds)
                        recent_points = [
                            d for d in data_points 
                            if d.timestamp >= cutoff_time and isinstance(d.value, (int, float))
                        ]
                        
                        if not recent_points:
                            continue
                        
                        values = [d.value for d in recent_points]
                        
                        # Calculate aggregations
                        snapshot = MetricSnapshot(
                            metric_name=metric_name,
                            timestamp=datetime.now(),
                            value=values[-1] if metric_def.metric_type == MetricType.GAUGE else len(values),
                            count=len(values),
                            sum_value=sum(values),
                            min_value=min(values),
                            max_value=max(values),
                            avg_value=statistics.mean(values),
                            tags=recent_points[0].tags if recent_points else {}
                        )
                        
                        # Calculate percentiles for histograms
                        if metric_def.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                            sorted_values = sorted(values)
                            n = len(sorted_values)
                            if n > 0:
                                snapshot.p95_value = sorted_values[int(0.95 * n)] if n > 20 else values[-1]
                                snapshot.p99_value = sorted_values[int(0.99 * n)] if n > 100 else values[-1]
                        
                        self.aggregated_metrics[metric_name] = snapshot
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metric data"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                
                cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
                
                with self._lock:
                    for metric_name, data_points in self.metric_data.items():
                        metric_def = self.metric_definitions.get(metric_name)
                        if not metric_def:
                            continue
                        
                        retention_cutoff = datetime.now() - timedelta(hours=metric_def.retention_hours)
                        
                        # Remove old data points
                        original_count = len(data_points)
                        while data_points and data_points[0].timestamp < retention_cutoff:
                            data_points.popleft()
                        
                        removed_count = original_count - len(data_points)
                        if removed_count > 0:
                            logger.debug(f"Cleaned {removed_count} old data points from {metric_name}")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def start_background_tasks(self) -> None:
        """Start background tasks for aggregation and cleanup"""
        try:
            if not self._aggregation_task or self._aggregation_task.done():
                self._aggregation_task = asyncio.create_task(self._aggregate_metrics())
            
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_old_data())
            
            logger.info("Background tasks started")
    
    def _start_fallback_tasks(self) -> None:
        """Start fallback sync tasks for environments without asyncio"""
        try:
            import threading
            import time
            
            def aggregation_worker():
                while True:
                    try:
                        time.sleep(self.aggregation_interval_seconds)
                        # Sync version of aggregation logic
                        self._sync_aggregate_metrics()
                    except Exception as e:
                        logger.error(f"Error in fallback aggregation: {e}")
            
            def cleanup_worker():
                while True:
                    try:
                        time.sleep(self.cleanup_interval_seconds)
                        # Sync version of cleanup logic
                        self._sync_cleanup_old_data()
                    except Exception as e:
                        logger.error(f"Error in fallback cleanup: {e}")
            
            # Start daemon threads
            aggregation_thread = threading.Thread(target=aggregation_worker, daemon=True)
            cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            
            aggregation_thread.start()
            cleanup_thread.start()
            
            logger.info("Fallback background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start fallback tasks: {e}")
    
    def _sync_aggregate_metrics(self) -> None:
        """Synchronous version of metrics aggregation"""
        try:
            with self._lock:
                for metric_name, data_points in self.metric_data.items():
                    if not data_points:
                        continue
                    
                    metric_def = self.metric_definitions.get(metric_name)
                    if not metric_def:
                        continue
                    
                    # Get recent data points
                    cutoff_time = datetime.now() - timedelta(seconds=self.aggregation_interval_seconds)
                    recent_points = [
                        d for d in data_points 
                        if d.timestamp >= cutoff_time and isinstance(d.value, (int, float))
                    ]
                    
                    if not recent_points:
                        continue
                    
                    values = [d.value for d in recent_points]
                    snapshot = MetricAggregatedSnapshot(
                        metric_name=metric_name,
                        timestamp=datetime.now(),
                        count=len(values),
                        sum_value=sum(values),
                        avg_value=statistics.mean(values),
                        min_value=min(values),
                        max_value=max(values)
                    )
                    
                    # Calculate percentiles
                    if len(values) > 1:
                        sorted_values = sorted(values)
                        n = len(values)
                        snapshot.p50_value = sorted_values[n // 2]
                        snapshot.p95_value = sorted_values[int(0.95 * n)] if n > 20 else values[-1]
                        snapshot.p99_value = sorted_values[int(0.99 * n)] if n > 100 else values[-1]
                    
                    self.aggregated_metrics[metric_name] = snapshot
                    
        except Exception as e:
            logger.error(f"Error in sync metrics aggregation: {e}")
    
    def _sync_cleanup_old_data(self) -> None:
        """Synchronous version of data cleanup"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with self._lock:
                for metric_name, data_points in self.metric_data.items():
                    metric_def = self.metric_definitions.get(metric_name)
                    if not metric_def:
                        continue
                    
                    retention_cutoff = datetime.now() - timedelta(hours=metric_def.retention_hours)
                    
                    # Remove old data points
                    original_count = len(data_points)
                    while data_points and data_points[0].timestamp < retention_cutoff:
                        data_points.popleft()
                    
                    removed_count = original_count - len(data_points)
                    if removed_count > 0:
                        logger.debug(f"Cleaned {removed_count} old data points from {metric_name}")
                        
        except Exception as e:
            logger.error(f"Error in sync cleanup: {e}")
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            # Create fallback sync tasks if async fails
            self._start_fallback_tasks()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format.lower() == "json":
            return self._export_json()
        elif format.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for metric_name, snapshot in self.aggregated_metrics.items():
            export_data["metrics"][metric_name] = snapshot.to_dict()
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for metric_name, snapshot in self.aggregated_metrics.items():
            metric_def = self.metric_definitions.get(metric_name)
            if not metric_def:
                continue
            
            # Base metric name
            base_name = f"neo_clone_{metric_name}"
            
            # Add tags as labels
            tags_str = ""
            if snapshot.tags:
                tags_list = [f'{k}="{v}"' for k, v in snapshot.tags.items()]
                tags_str = "{" + ",".join(tags_list) + "}"
            
            # Export based on metric type
            if metric_def.metric_type == MetricType.GAUGE:
                lines.append(f"{base_name}{tags_str} {snapshot.value}")
            elif metric_def.metric_type == MetricType.COUNTER:
                lines.append(f"{base_name}_total{tags_str} {snapshot.sum_value}")
            elif metric_def.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                lines.append(f"{base_name}_count{tags_str} {snapshot.count}")
                lines.append(f"{base_name}_sum{tags_str} {snapshot.sum_value}")
                lines.append(f"{base_name}_avg{tags_str} {snapshot.avg_value}")
                lines.append(f"{base_name}_p95{tags_str} {snapshot.p95_value}")
                lines.append(f"{base_name}_p99{tags_str} {snapshot.p99_value}")
        
        return "\n".join(lines)
    
    def shutdown(self) -> None:
        """Shutdown metrics collector"""
        try:
            # Cancel background tasks
            if self._aggregation_task and not self._aggregation_task.done():
                self._aggregation_task.cancel()
            
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            
            logger.info("Metrics collector shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during metrics collector shutdown: {e}")


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector(**kwargs) -> MetricsCollector:
    """Get or create global metrics collector instance"""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        with _collector_lock:
            if _global_metrics_collector is None:
                _global_metrics_collector = MetricsCollector(**kwargs)
    
    return _global_metrics_collector


# Convenience functions for common metrics
def increment_brain_processing_count(tags: Optional[Dict[str, str]] = None) -> None:
    """Increment brain processing count"""
    collector = get_metrics_collector()
    collector.increment_counter("brain_processing_count", tags=tags)


def record_brain_processing_duration(duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record brain processing duration"""
    collector = get_metrics_collector()
    collector.record_timer("brain_processing_duration", duration_ms, tags)


def increment_skill_execution_count(skill_name: str, success: bool = True, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment skill execution count"""
    collector = get_metrics_collector()
    full_tags = {"skill_name": skill_name, "success": str(success)}
    if tags:
        full_tags.update(tags)
    collector.increment_counter("skill_execution_count", tags=full_tags)


def record_skill_execution_duration(skill_name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record skill execution duration"""
    collector = get_metrics_collector()
    full_tags = {"skill_name": skill_name}
    if tags:
        full_tags.update(tags)
    collector.record_timer("skill_execution_duration", duration_ms, tags=full_tags)


def record_memory_access_duration(operation: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record memory access duration"""
    collector = get_metrics_collector()
    full_tags = {"operation": operation}
    if tags:
        full_tags.update(tags)
    collector.record_timer("memory_access_duration", duration_ms, tags=full_tags)


def set_active_sessions_count(count: int, tags: Optional[Dict[str, str]] = None) -> None:
    """Set active sessions count"""
    collector = get_metrics_collector()
    collector.set_gauge("active_sessions", count, tags)


def set_error_rate(rate: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set error rate"""
    collector = get_metrics_collector()
    collector.set_gauge("error_rate", rate, tags)


def set_cache_hit_rate(rate: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Set cache hit rate"""
    collector = get_metrics_collector()
    collector.set_gauge("cache_hit_rate", rate, tags)


def record_request_rate(rate: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record request rate"""
    collector = get_metrics_collector()
    collector.set_gauge("request_rate", rate, tags)