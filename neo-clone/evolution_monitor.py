"""
Evolution Monitor Dashboard - Real-time monitoring of Neo-Clone's autonomous evolution

This module provides a live dashboard to monitor the evolution engine's activities,
performance metrics, and improvement progress.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class EvolutionMonitor:
    """Real-time monitoring dashboard for the autonomous evolution engine"""

    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.dashboard_data = {
            'status': 'stopped',
            'uptime': 0,
            'last_update': None,
            'metrics': {},
            'recent_activities': [],
            'active_opportunities': [],
            'performance_trends': [],
            'system_health': {},
            'evolution_goals': []
        }
        self.update_interval = 5  # seconds

    def start_monitoring(self):
        """Start the monitoring dashboard"""
        if self.is_monitoring:
            logger.warning("Evolution monitor already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Evolution monitor dashboard started")

    def stop_monitoring(self):
        """Stop the monitoring dashboard"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Evolution monitor dashboard stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        start_time = datetime.now()

        while self.is_monitoring:
            try:
                # Update dashboard data
                self._update_dashboard_data()
                self._analyze_trends()
                self._check_system_health()
                self._update_goals_progress()

                # Save dashboard snapshot
                self._save_dashboard_snapshot()

                # Log summary every minute
                if int(time.time()) % 60 == 0:
                    self._log_monitoring_summary()

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)  # Longer pause on error

    def _update_dashboard_data(self):
        """Update dashboard data from evolution engine"""
        try:
            # Import here to avoid circular imports
            from autonomous_evolution_engine import get_evolution_status

            status = get_evolution_status()

            self.dashboard_data.update({
                'status': 'running' if status.get('is_running', False) else 'stopped',
                'last_update': datetime.now().isoformat(),
                'metrics': status.get('metrics', {}),
                'queue_size': status.get('queue_size', 0),
                'internet_scanning': status.get('internet_scanning', {}),
                'llm_independence': status.get('llm_independence', {}),
                'performance': status.get('performance', {})
            })

            # Update uptime
            if self.dashboard_data['status'] == 'running':
                # This is approximate - would need better tracking
                self.dashboard_data['uptime'] = (datetime.now() - datetime.fromisoformat(self.dashboard_data.get('start_time', datetime.now().isoformat()))).total_seconds()

        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")

    def _analyze_trends(self):
        """Analyze performance trends and patterns"""
        try:
            metrics = self.dashboard_data.get('metrics', {})

            # Calculate trends
            trends = {
                'opportunities_discovered_trend': self._calculate_trend('opportunities_discovered'),
                'implementation_success_trend': self._calculate_trend('opportunities_implemented'),
                'scan_performance_trend': self._calculate_trend('scan_duration'),
                'timestamp': datetime.now().isoformat()
            }

            # Store recent trends (keep last 20)
            self.dashboard_data['performance_trends'].append(trends)
            if len(self.dashboard_data['performance_trends']) > 20:
                self.dashboard_data['performance_trends'] = self.dashboard_data['performance_trends'][-20:]

            # Generate insights
            insights = self._generate_performance_insights(trends)
            self.dashboard_data['performance_insights'] = insights

        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")

    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric (increasing, decreasing, stable)"""
        trends = self.dashboard_data.get('performance_trends', [])
        if len(trends) < 3:
            return 'insufficient_data'

        recent_values = [t.get(metric_name, 0) for t in trends[-3:]]
        if len(recent_values) < 3:
            return 'insufficient_data'

        # Simple trend calculation
        if recent_values[2] > recent_values[1] > recent_values[0]:
            return 'increasing'
        elif recent_values[2] < recent_values[1] < recent_values[0]:
            return 'decreasing'
        else:
            return 'stable'

    def _generate_performance_insights(self, trends: Dict[str, Any]) -> List[str]:
        """Generate human-readable performance insights"""
        insights = []

        opp_trend = trends.get('opportunities_discovered_trend', 'stable')
        if opp_trend == 'increasing':
            insights.append("📈 Opportunity discovery is increasing - evolution is finding more improvement opportunities")
        elif opp_trend == 'decreasing':
            insights.append("📉 Opportunity discovery is decreasing - may need to expand scanning scope")

        success_trend = trends.get('implementation_success_trend', 'stable')
        if success_trend == 'increasing':
            insights.append("✅ Implementation success is improving - evolution is getting better at applying changes")
        elif success_trend == 'decreasing':
            insights.append("❌ Implementation success is declining - may need to review change validation")

        scan_trend = trends.get('scan_performance_trend', 'stable')
        if scan_trend == 'increasing':
            insights.append("⏱️ Scan times are increasing - consider performance optimizations")
        elif scan_trend == 'decreasing':
            insights.append("⚡ Scan times are improving - performance optimizations are working")

        return insights

    def _check_system_health(self):
        """Check overall system health"""
        try:
            health_data = {
                'evolution_engine': self._check_evolution_engine_health(),
                'file_system': self._check_file_system_health(),
                'memory_usage': self._check_memory_usage(),
                'network_connectivity': self._check_network_connectivity(),
                'last_check': datetime.now().isoformat()
            }

            self.dashboard_data['system_health'] = health_data

            # Calculate overall health score
            health_score = self._calculate_health_score(health_data)
            self.dashboard_data['overall_health_score'] = health_score

        except Exception as e:
            logger.error(f"Failed to check system health: {e}")

    def _check_evolution_engine_health(self) -> Dict[str, Any]:
        """Check evolution engine health"""
        status = self.dashboard_data.get('status', 'unknown')

        if status == 'running':
            # Check if metrics are updating
            last_update = self.dashboard_data.get('last_update')
            if last_update:
                last_update_time = datetime.fromisoformat(last_update)
                time_since_update = (datetime.now() - last_update_time).total_seconds()

                if time_since_update > 60:  # No update for more than 1 minute
                    return {'status': 'warning', 'message': f'No updates for {time_since_update:.0f} seconds'}
                else:
                    return {'status': 'healthy', 'message': 'Evolution engine running normally'}
            else:
                return {'status': 'warning', 'message': 'No update timestamp available'}
        else:
            return {'status': 'error', 'message': 'Evolution engine not running'}

    def _check_file_system_health(self) -> Dict[str, Any]:
        """Check file system health"""
        try:
            import os

            # Check critical directories exist
            critical_dirs = ['skills', 'backups', 'repository_explorations']
            missing_dirs = [d for d in critical_dirs if not os.path.exists(d)]

            if missing_dirs:
                return {'status': 'warning', 'message': f'Missing directories: {missing_dirs}'}

            # Check disk space (simplified)
            try:
                stat = os.statvfs('.')
                free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                if free_space_gb < 1:
                    return {'status': 'error', 'message': f'Low disk space: {free_space_gb:.1f} GB free'}
            except:
                pass  # statvfs not available on Windows

            return {'status': 'healthy', 'message': 'File system healthy'}

        except Exception as e:
            return {'status': 'error', 'message': f'File system check failed: {e}'}

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 90:
                return {'status': 'error', 'message': f'High memory usage: {memory_percent}%'}
            elif memory_percent > 75:
                return {'status': 'warning', 'message': f'Elevated memory usage: {memory_percent}%'}
            else:
                return {'status': 'healthy', 'message': f'Memory usage: {memory_percent}%'}

        except ImportError:
            return {'status': 'unknown', 'message': 'Memory monitoring not available (psutil not installed)'}
        except Exception as e:
            return {'status': 'error', 'message': f'Memory check failed: {e}'}

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            import requests

            # Quick connectivity test
            response = requests.get('https://httpbin.org/status/200', timeout=5)
            if response.status_code == 200:
                return {'status': 'healthy', 'message': 'Network connectivity good'}
            else:
                return {'status': 'warning', 'message': f'Unexpected response: {response.status_code}'}

        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': f'Network connectivity issues: {e}'}
        except ImportError:
            return {'status': 'unknown', 'message': 'Network monitoring not available (requests not installed)'}

    def _calculate_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        penalties = {
            'error': 25,
            'warning': 10,
            'unknown': 5
        }

        for component, data in health_data.items():
            if isinstance(data, dict) and 'status' in data:
                status = data['status']
                if status in penalties:
                    score -= penalties[status]

        return max(0.0, score)

    def _update_goals_progress(self):
        """Update progress on evolution goals"""
        try:
            # This would integrate with the goal tracking in the evolution engine
            # For now, create sample goals based on current metrics
            metrics = self.dashboard_data.get('metrics', {})

            goals = [
                {
                    'name': 'Opportunity Discovery',
                    'current': metrics.get('opportunities_discovered', 0),
                    'target': 1000,
                    'progress': min(100, (metrics.get('opportunities_discovered', 0) / 1000) * 100)
                },
                {
                    'name': 'Implementation Success',
                    'current': metrics.get('opportunities_implemented', 0),
                    'target': metrics.get('opportunities_discovered', 0),
                    'progress': (metrics.get('opportunities_implemented', 0) / max(1, metrics.get('opportunities_discovered', 0))) * 100 if metrics.get('opportunities_discovered', 0) > 0 else 0
                }
            ]

            self.dashboard_data['evolution_goals'] = goals

        except Exception as e:
            logger.error(f"Failed to update goals progress: {e}")

    def _save_dashboard_snapshot(self):
        """Save dashboard snapshot to file"""
        try:
            snapshot_file = 'evolution_dashboard_snapshot.json'

            # Create a clean snapshot (remove very large data)
            snapshot = self.dashboard_data.copy()
            snapshot['performance_trends'] = snapshot['performance_trends'][-5:] if 'performance_trends' in snapshot else []

            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save dashboard snapshot: {e}")

    def _log_monitoring_summary(self):
        """Log a summary of monitoring data"""
        try:
            status = self.dashboard_data.get('status', 'unknown')
            metrics = self.dashboard_data.get('metrics', {})
            health_score = self.dashboard_data.get('overall_health_score', 0)

            logger.info(f"📊 Evolution Monitor Summary - Status: {status}, "
                       f"Opportunities: {metrics.get('opportunities_discovered', 0)} discovered / "
                       f"{metrics.get('opportunities_implemented', 0)} implemented, "
                       f"Health Score: {health_score:.1f}/100")

            # Log insights
            insights = self.dashboard_data.get('performance_insights', [])
            for insight in insights:
                logger.info(f"🔍 {insight}")

        except Exception as e:
            logger.error(f"Failed to log monitoring summary: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()

    def generate_report(self) -> str:
        """Generate a comprehensive monitoring report"""
        try:
            data = self.get_dashboard_data()

            report = f"""
# Neo-Clone Evolution Monitor Report
Generated: {datetime.now().isoformat()}

## System Status
- **Status**: {data.get('status', 'unknown').upper()}
- **Uptime**: {data.get('uptime', 0):.0f} seconds
- **Last Update**: {data.get('last_update', 'never')}
- **Overall Health Score**: {data.get('overall_health_score', 0):.1f}/100

## Evolution Metrics
- **Opportunities Discovered**: {data.get('metrics', {}).get('opportunities_discovered', 0)}
- **Opportunities Implemented**: {data.get('metrics', {}).get('opportunities_implemented', 0)}
- **Performance Gains**: {data.get('metrics', {}).get('performance_gains', 0)}
- **Features Added**: {data.get('metrics', {}).get('features_added', 0)}
- **Improvements Made**: {data.get('metrics', {}).get('improvements_made', 0)}
- **Bugs Fixed**: {data.get('metrics', {}).get('bugs_fixed', 0)}

## Queue Status
- **Active Opportunities**: {data.get('queue_size', 0)}

## Performance Insights
"""

            insights = data.get('performance_insights', [])
            for insight in insights:
                report += f"- {insight}\n"

            report += "\n## System Health\n"

            health = data.get('system_health', {})
            for component, status_data in health.items():
                if isinstance(status_data, dict):
                    status = status_data.get('status', 'unknown')
                    message = status_data.get('message', 'No details')
                    report += f"- **{component.title()}**: {status.upper()} - {message}\n"

            report += "\n## Evolution Goals\n"

            goals = data.get('evolution_goals', [])
            for goal in goals:
                progress = goal.get('progress', 0)
                report += f"- **{goal['name']}**: {goal.get('current', 0)}/{goal.get('target', 0)} ({progress:.1f}%)\n"

            report += "\n## LLM Independence
"
            llm_data = data.get('llm_independence', {})
            report += f"- **Core Functionality LLM-Free**: {llm_data.get('core_functionality_llm_free', False)}\n"
            report += f"- **LLM Enhancements Available**: {llm_data.get('llm_enhancements_available', False)}\n"
            report += f"- **Currently Available**: {llm_data.get('llm_currently_available', False)}\n"

            return report

        except Exception as e:
            return f"Error generating report: {e}"

# Global monitor instance
evolution_monitor = EvolutionMonitor()

def start_evolution_monitor():
    """Start the evolution monitor dashboard"""
    evolution_monitor.start_monitoring()

def stop_evolution_monitor():
    """Stop the evolution monitor dashboard"""
    evolution_monitor.stop_monitoring()

def get_monitor_report():
    """Get the current monitoring report"""
    return evolution_monitor.generate_report()

if __name__ == "__main__":
    print("Neo-Clone Evolution Monitor Dashboard")
    print("Starting monitoring...")

    start_evolution_monitor()

    try:
        while True:
            time.sleep(10)
            # Print summary every 10 seconds
            data = evolution_monitor.get_dashboard_data()
            status = data.get('status', 'unknown')
            metrics = data.get('metrics', {})
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status} | "
                  f"Discovered: {metrics.get('opportunities_discovered', 0)} | "
                  f"Implemented: {metrics.get('opportunities_implemented', 0)} | "
                  f"Health: {data.get('overall_health_score', 0):.1f}/100")

    except KeyboardInterrupt:
        print("\nStopping evolution monitor...")
        stop_evolution_monitor()
        print("Evolution monitor stopped.")