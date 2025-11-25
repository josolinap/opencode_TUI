"""
Unified Performance Dashboard for Neo-Clone
Consolidates all monitoring into single, actionable interface
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Unified system metrics"""
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_skills: int = 0
    total_skills: int = 0
    mcp_tools_available: int = 0
    evolution_improvements: int = 0
    error_rate: float = 0.0
    response_time_avg: float = 0.0
    requests_per_minute: float = 0.0

@dataclass
class Alert:
    """System alert"""
    id: str
    severity: str  # info, warning, critical
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    category: str = "general"

class UnifiedDashboard:
    """Unified monitoring dashboard"""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[Alert] = []
        self.is_running = False
        self.update_interval = 30  # seconds
        self.max_history_size = 1000
        
        # System state
        self.current_metrics = SystemMetrics(timestamp=datetime.now())
        self.skill_registry = {}
        self.mcp_tools = {}
        self.evolution_status = {}
        
        # Performance tracking
        self.request_times = []
        self.error_count = 0
        self.total_requests = 0
        self.last_cleanup = time.time()
        
    def start(self):
        """Start the dashboard"""
        if self.is_running:
            logger.warning("Dashboard already running")
            return
            
        self.is_running = True
        logger.info("📊 Starting Unified Performance Dashboard")
        
        # Start background update thread
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()
        
        logger.info("✅ Unified Dashboard started")
        
    def _update_loop(self):
        """Background update loop"""
        while self.is_running:
            try:
                self._collect_metrics()
                self._check_alerts()
                self._cleanup_old_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                time.sleep(5)
                
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # Basic system metrics (simplified)
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
        except ImportError:
            # Fallback if psutil not available
            cpu_usage = 0.0
            memory_usage = 0.0
            
        # Skill metrics
        active_skills = len([s for s in self.skill_registry.values() if getattr(s, 'status', None) == 'running'])
        total_skills = len(self.skill_registry)
        
        # MCP tools
        mcp_tools_available = len(self.mcp_tools)
        
        # Evolution metrics
        evolution_improvements = self.evolution_status.get('improvements_made', 0)
        
        # Performance metrics
        response_time_avg = sum(self.request_times[-100:]) / len(self.request_times[-100:]) if self.request_times else 0.0
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        
        # Calculate requests per minute
        now = time.time()
        recent_requests = len([t for t in self.request_times if now - t < 60])
        requests_per_minute = recent_requests
        
        # Create metrics snapshot
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_skills=active_skills,
            total_skills=total_skills,
            mcp_tools_available=mcp_tools_available,
            evolution_improvements=evolution_improvements,
            error_rate=error_rate,
            response_time_avg=response_time_avg,
            requests_per_minute=requests_per_minute
        )
        
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Keep history manageable
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size//2:]
            
    def _check_alerts(self):
        """Check for alert conditions"""
        alerts = []
        
        # High CPU usage
        if self.current_metrics.cpu_usage > 80:
            alerts.append(Alert(
                id=f"cpu_high_{int(time.time())}",
                severity="warning",
                title="High CPU Usage",
                description=f"CPU usage is {self.current_metrics.cpu_usage:.1f}%",
                timestamp=datetime.now(),
                category="performance"
            ))
            
        # High memory usage
        if self.current_metrics.memory_usage > 85:
            alerts.append(Alert(
                id=f"memory_high_{int(time.time())}",
                severity="warning", 
                title="High Memory Usage",
                description=f"Memory usage is {self.current_metrics.memory_usage:.1f}%",
                timestamp=datetime.now(),
                category="performance"
            ))
            
        # High error rate
        if self.current_metrics.error_rate > 10:
            alerts.append(Alert(
                id=f"error_rate_high_{int(time.time())}",
                severity="critical",
                title="High Error Rate", 
                description=f"Error rate is {self.current_metrics.error_rate:.1f}%",
                timestamp=datetime.now(),
                category="reliability"
            ))
            
        # Slow response times
        if self.current_metrics.response_time_avg > 5.0:
            alerts.append(Alert(
                id=f"response_slow_{int(time.time())}",
                severity="warning",
                title="Slow Response Times",
                description=f"Average response time is {self.current_metrics.response_time_avg:.2f}s",
                timestamp=datetime.now(),
                category="performance"
            ))
            
        # Add new alerts (avoid duplicates)
        for alert in alerts:
            if not any(a.id == alert.id for a in self.alerts):
                self.alerts.append(alert)
                logger.warning(f"🚨 ALERT: {alert.title} - {alert.description}")
                
    def _cleanup_old_data(self):
        """Clean up old data"""
        now = time.time()
        
        # Clean old alerts (older than 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time or not a.resolved]
        
        # Clean old metrics (older than 1 hour)
        if len(self.metrics_history) > 100:
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for performance tracking"""
        self.request_times.append(time.time())
        if not success:
            self.error_count += 1
        self.total_requests += 1
        
    def update_skill_registry(self, skills: Dict[str, Any]):
        """Update skill registry information"""
        self.skill_registry = skills
        
    def update_mcp_tools(self, tools: Dict[str, Any]):
        """Update MCP tools information"""
        self.mcp_tools = tools
        
    def update_evolution_status(self, status: Dict[str, Any]):
        """Update evolution engine status"""
        self.evolution_status = status
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            'current_metrics': {
                'timestamp': self.current_metrics.timestamp.isoformat(),
                'cpu_usage': self.current_metrics.cpu_usage,
                'memory_usage': self.current_metrics.memory_usage,
                'active_skills': self.current_metrics.active_skills,
                'total_skills': self.current_metrics.total_skills,
                'mcp_tools_available': self.current_metrics.mcp_tools_available,
                'evolution_improvements': self.current_metrics.evolution_improvements,
                'error_rate': self.current_metrics.error_rate,
                'response_time_avg': self.current_metrics.response_time_avg,
                'requests_per_minute': self.current_metrics.requests_per_minute
            },
            'alerts': [
                {
                    'id': alert.id,
                    'severity': alert.severity,
                    'title': alert.title,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat(),
                    'category': alert.category,
                    'resolved': alert.resolved
                }
                for alert in sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:20]
            ],
            'system_health': self._calculate_health_score(),
            'trends': self._calculate_trends(),
            'summary': {
                'total_alerts': len(self.alerts),
                'critical_alerts': len([a for a in self.alerts if a.severity == 'critical']),
                'warning_alerts': len([a for a in self.alerts if a.severity == 'warning']),
                'uptime_percentage': 99.5,  # Placeholder
                'last_update': datetime.now().isoformat()
            }
        }
        
    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        score = 100
        
        # Deduct for high resource usage
        if self.current_metrics.cpu_usage > 80:
            score -= 15
        elif self.current_metrics.cpu_usage > 60:
            score -= 5
            
        if self.current_metrics.memory_usage > 85:
            score -= 15
        elif self.current_metrics.memory_usage > 70:
            score -= 5
            
        # Deduct for high error rate
        if self.current_metrics.error_rate > 10:
            score -= 25
        elif self.current_metrics.error_rate > 5:
            score -= 10
            
        # Deduct for slow response times
        if self.current_metrics.response_time_avg > 5.0:
            score -= 15
        elif self.current_metrics.response_time_avg > 2.0:
            score -= 5
            
        score = max(0, score)
        
        # Determine status
        if score >= 90:
            status = "excellent"
        elif score >= 75:
            status = "good"
        elif score >= 60:
            status = "fair"
        else:
            status = "poor"
            
        return {
            'score': score,
            'status': status,
            'factors': {
                'cpu_impact': -15 if self.current_metrics.cpu_usage > 80 else (-5 if self.current_metrics.cpu_usage > 60 else 0),
                'memory_impact': -15 if self.current_metrics.memory_usage > 85 else (-5 if self.current_metrics.memory_usage > 70 else 0),
                'error_impact': -25 if self.current_metrics.error_rate > 10 else (-10 if self.current_metrics.error_rate > 5 else 0),
                'performance_impact': -15 if self.current_metrics.response_time_avg > 5.0 else (-5 if self.current_metrics.response_time_avg > 2.0 else 0)
            }
        }
        
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(self.metrics_history) < 10:
            return {'status': 'insufficient_data'}
            
        recent = self.metrics_history[-10:]
        older = self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else []
        
        trends = {}
        
        if older:
            # CPU trend
            recent_cpu = sum(m.cpu_usage for m in recent) / len(recent)
            older_cpu = sum(m.cpu_usage for m in older) / len(older)
            trends['cpu'] = 'increasing' if recent_cpu > older_cpu + 5 else 'decreasing' if recent_cpu < older_cpu - 5 else 'stable'
            
            # Memory trend
            recent_mem = sum(m.memory_usage for m in recent) / len(recent)
            older_mem = sum(m.memory_usage for m in older) / len(older)
            trends['memory'] = 'increasing' if recent_mem > older_mem + 5 else 'decreasing' if recent_mem < older_mem - 5 else 'stable'
            
            # Response time trend
            recent_rt = sum(m.response_time_avg for m in recent) / len(recent)
            older_rt = sum(m.response_time_avg for m in older) / len(older)
            trends['response_time'] = 'increasing' if recent_rt > older_rt + 0.5 else 'decreasing' if recent_rt < older_rt - 0.5 else 'stable'
        else:
            trends = {'status': 'calculating'}
            
        return trends
        
    def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        logger.info("📊 Unified Dashboard stopped")

# Global dashboard instance
dashboard = UnifiedDashboard()

def start_dashboard():
    """Start the unified dashboard"""
    dashboard.start()
    return dashboard

def get_dashboard():
    """Get the dashboard instance"""
    return dashboard