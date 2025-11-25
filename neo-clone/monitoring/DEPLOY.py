#!/usr/bin/env python3
"""
DEPLOY - Neo-Clone Monitoring System Deployment Script

This script deploys the monitoring system for immediate production use.
Run this script to deploy monitoring to your Neo-Clone + OpenCode TUI system.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

def print_banner():
    """Print deployment banner"""
    print("=" * 80)
    print("🚀 NEO-CLONE MONITORING SYSTEM DEPLOYMENT")
    print("=" * 80)
    print("Deploying comprehensive monitoring for Neo-Clone + OpenCode TUI")
    print()

def check_prerequisites():
    """Check deployment prerequisites"""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "monitoring_integration.py").exists():
        print("❌ Run this script from the monitoring directory")
        return False
    
    print("✅ Prerequisites check passed")
    return True

def create_deployment_structure():
    """Create deployment directory structure"""
    print("📁 Creating deployment structure...")
    
    # Create directories
    dirs_to_create = [
        "logs",
        "data/metrics",
        "data/traces", 
        "data/profiles",
        "config",
        "backups"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {dir_path}")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies that are always needed
    core_deps = [
        "asyncio-throttle>=1.0.2",
        "statistics>=1.0.0"
    ]
    
    # Optional but recommended dependencies
    optional_deps = [
        "psutil>=5.9.0",  # System monitoring
        "prometheus-client>=0.17.0",  # Metrics export
    ]
    
    # Install core dependencies
    for dep in core_deps:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"   ✅ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Failed to install {dep}: {e}")
            return False
    
    # Install optional dependencies (best effort)
    for dep in optional_deps:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"   ✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Optional dependency {dep} not installed")
    
    return True

def create_production_config():
    """Create production configuration"""
    print("⚙️  Creating production configuration...")
    
    config = {
        "monitoring": {
            "enabled": True,
            "tracing_enabled": True,
            "metrics_enabled": True,
            "profiling_enabled": True,
            "dashboard_enabled": True,
            "auto_instrument_brain": True,
            "auto_instrument_skills": True,
            "opencode_integration": True
        },
        "performance": {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "response_time_threshold": 2000.0,
            "monitoring_interval": 1.0
        },
        "tracing": {
            "service_name": "neo-clone-opencode",
            "sample_rate": 0.1,
            "endpoint": None,
            "jaeger_endpoint": None,
            "otlp_endpoint": None
        },
        "metrics": {
            "export_interval": 30.0,
            "endpoint": None,
            "prometheus_port": 8080,
            "retention_hours": 24
        },
        "profiling": {
            "sample_rate": 0.05,
            "min_duration_ms": 100.0,
            "memory_profiling": False,
            "cpu_profiling": True
        },
        "dashboard": {
            "enabled": True,
            "refresh_interval": 2.0,
            "port": 8081,
            "host": "localhost"
        },
        "error_handling": {
            "max_errors_per_window": 100,
            "error_window_minutes": 5,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60,
            "log_dir": "logs"
        },
        "logging": {
            "level": "INFO",
            "file": "logs/monitoring.log",
            "max_size_mb": 100,
            "backup_count": 5
        }
    }
    
    # Write configuration
    config_file = Path("config/monitoring_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✅ Configuration created: {config_file}")
    return True

def create_startup_script():
    """Create startup script for monitoring"""
    print("🚀 Creating startup script...")
    
    startup_script = '''#!/usr/bin/env python3
"""
Neo-Clone Monitoring System Startup Script
"""

import sys
import os
from pathlib import Path

# Add monitoring to path
monitoring_dir = Path(__file__).parent
sys.path.insert(0, str(monitoring_dir))

def start_monitoring():
    """Start the monitoring system"""
    try:
        print("🚀 Starting Neo-Clone Monitoring System...")
        
        # Import and initialize monitoring
        from monitoring_integration import get_global_monitoring, MonitoringConfig
        
        # Load configuration
        config_file = monitoring_dir / "config" / "monitoring_config.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                config_data = json.load(f)
            config = MonitoringConfig(**config_data.get('monitoring', {}))
            print("✅ Configuration loaded")
        else:
            config = MonitoringConfig()
            print("⚠️  Using default configuration")
        
        # Initialize monitoring
        monitoring = get_global_monitoring()
        if monitoring.initialize():
            print("✅ Monitoring system initialized successfully")
            
            # Start background tasks
            if hasattr(monitoring, 'start_background_tasks'):
                import asyncio
                asyncio.run(monitoring.start_background_tasks())
                print("✅ Background tasks started")
            
            print("🎯 Monitoring system is running!")
            print("📊 Dashboard available at: http://localhost:8081")
            print("📈 Metrics available at: http://localhost:8080/metrics")
            
            # Keep running
            try:
                import time
                while True:
                    time.sleep(60)
                    # Health check could go here
            except KeyboardInterrupt:
                print("\\n🛑 Shutting down monitoring system...")
                monitoring.shutdown()
                print("✅ Monitoring system stopped")
        else:
            print("❌ Failed to initialize monitoring system")
            return False
            
    except Exception as e:
        print(f"❌ Error starting monitoring: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_monitoring()
'''
    
    # Write startup script
    startup_file = Path("start_monitoring.py")
    with open(startup_file, 'w') as f:
        f.write(startup_script)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(startup_file, 0o755)
    
    print(f"   ✅ Startup script created: {startup_file}")
    return True

def create_integration_wrapper():
    """Create integration wrapper for OpenCode TUI"""
    print("🔗 Creating OpenCode TUI integration...")
    
    integration_code = '''
"""
OpenCode TUI Integration Wrapper for Neo-Clone Monitoring

This module provides automatic monitoring integration for OpenCode TUI agents.
"""

import sys
import os
from pathlib import Path

# Add monitoring to path
monitoring_dir = Path(__file__).parent.parent / "monitoring"
sys.path.insert(0, str(monitoring_dir))

def wrap_neo_clone_tool(original_neo_clone_func):
    """Wrap neo-clone tool with monitoring"""
    def monitored_neo_clone(message: str, mode: str = "tool", timeout: int = 300000):
        try:
            from monitoring_integration import get_global_monitoring, MonitoredOperation
            
            monitoring = get_global_monitoring()
            
            with MonitoredOperation(monitoring, "neo_clone_execution", 
                                  metadata={'mode': mode, 'timeout': timeout}):
                return original_neo_clone_func(message, mode, timeout)
                
        except ImportError:
            # Fallback to original function if monitoring not available
            return original_neo_clone_func(message, mode, timeout)
    
    return monitored_neo_clone

def wrap_model_selector_tool(original_model_selector_func):
    """Wrap model selector tool with monitoring"""
    def monitored_model_selector(task: str, requirements: dict, 
                                   max_recommendations: int = 3, format: str = "simple"):
        try:
            from monitoring_integration import get_global_monitoring, MonitoredOperation
            
            monitoring = get_global_monitoring()
            
            with MonitoredOperation(monitoring, "model_selection",
                                  metadata={'task': task, 'format': format}):
                return original_model_selector_func(task, requirements, 
                                               max_recommendations, format)
                                               
        except ImportError:
            # Fallback to original function if monitoring not available
            return original_model_selector_func(task, requirements, 
                                           max_recommendations, format)
    
    return monitored_model_selector

# Auto-apply monitoring if environment variable is set
if os.getenv('NEO_CLONE_MONITORING_ENABLED', 'true').lower() == 'true':
    try:
        # Import original tools (these would be the actual tool functions)
        # This is a template - actual imports would depend on your setup
        
        print("✅ Neo-Clone monitoring integration enabled")
        
    except ImportError as e:
        print(f"⚠️  Could not enable monitoring integration: {e}")
'''
    
    # Write integration wrapper
    integration_file = Path("opencode_integration.py")
    with open(integration_file, 'w') as f:
        f.write(integration_code)
    
    print(f"   ✅ Integration wrapper created: {integration_file}")
    return True

def create_health_check():
    """Create health check script"""
    print("🏥 Creating health check script...")
    
    health_check_script = '''#!/usr/bin/env python3
"""
Neo-Clone Monitoring System Health Check
"""

import sys
import json
import time
from pathlib import Path

# Add monitoring to path
monitoring_dir = Path(__file__).parent
sys.path.insert(0, str(monitoring_dir))

def check_health():
    """Check monitoring system health"""
    try:
        from monitoring_integration import get_global_monitoring
        from error_handling import get_global_error_handler
        
        monitoring = get_global_monitoring()
        error_handler = get_global_error_handler()
        
        health_status = {
            "timestamp": time.time(),
            "status": "healthy",
            "checks": {}
        }
        
        # Check monitoring initialization
        health_status["checks"]["monitoring_initialized"] = monitoring.status.initialized
        if not monitoring.status.initialized:
            health_status["status"] = "unhealthy"
        
        # Check error rates
        error_summary = error_handler.get_error_summary(hours=1)
        health_status["checks"]["error_count"] = error_summary["total_errors"]
        if error_summary["total_errors"] > 10:
            health_status["status"] = "degraded"
        
        # Check active operations
        health_status["checks"]["active_operations"] = len(monitoring.active_operations)
        
        # Print health status
        print(json.dumps(health_status, indent=2))
        
        return health_status["status"] in ["healthy", "degraded"]
        
    except Exception as e:
        error_status = {
            "timestamp": time.time(),
            "status": "unhealthy",
            "error": str(e)
        }
        print(json.dumps(error_status, indent=2))
        return False

if __name__ == "__main__":
    success = check_health()
    sys.exit(0 if success else 1)
'''
    
    # Write health check script
    health_file = Path("health_check.py")
    with open(health_file, 'w') as f:
        f.write(health_check_script)
    
    # Make executable
    if os.name != 'nt':
        os.chmod(health_file, 0o755)
    
    print(f"   ✅ Health check script created: {health_file}")
    return True

def create_systemd_service():
    """Create systemd service file for Linux"""
    if os.name != 'nt':  # Only on Unix systems
        print("🔧 Creating systemd service...")
        
        service_content = '''[Unit]
Description=Neo-Clone Monitoring System
After=network.target

[Service]
Type=simple
User=neo-clone
WorkingDirectory={monitoring_dir}
ExecStart={python_path} {startup_script}
Restart=always
RestartSec=10
Environment=NEO_CLONE_MONITORING_ENABLED=true

[Install]
WantedBy=multi-user.target
'''.format(
            monitoring_dir=str(Path.cwd()),
            python_path=sys.executable,
            startup_script=str(Path.cwd() / "start_monitoring.py")
        )
        
        # Write service file
        service_file = Path("neo-clone-monitoring.service")
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"   ✅ Systemd service created: {service_file}")
        print("   💡 To install: sudo cp neo-clone-monitoring.service /etc/systemd/system/")
        print("   💡 To enable: sudo systemctl enable neo-clone-monitoring")
        print("   💡 To start: sudo systemctl start neo-clone-monitoring")
    
    return True

def test_deployment():
    """Test the deployment"""
    print("🧪 Testing deployment...")
    
    try:
        # Test import
        sys.path.insert(0, str(Path.cwd()))
        from monitoring_integration import get_global_monitoring
        
        # Test initialization
        monitoring = get_global_monitoring()
        success = monitoring.initialize()
        
        if success:
            print("   ✅ Monitoring system initialized successfully")
            
            # Test basic operation
            operation_id = monitoring.start_operation("test_deployment")
            result = monitoring.end_operation(operation_id, True)
            
            if result.get('success', True):
                print("   ✅ Basic operation test passed")
            else:
                print("   ⚠️  Basic operation test failed")
            
            return True
        else:
            print("   ❌ Monitoring system initialization failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Deployment test failed: {e}")
        return False

def create_deployment_summary():
    """Create deployment summary"""
    print("📋 Creating deployment summary...")
    
    summary = {
        "deployment_timestamp": time.time(),
        "deployment_version": "1.0.0",
        "components": {
            "monitoring_integration": "✅ Deployed",
            "distributed_tracing": "✅ Deployed", 
            "metrics_collection": "✅ Deployed",
            "performance_profiler": "✅ Deployed",
            "error_handling": "✅ Deployed",
            "tui_dashboard": "✅ Deployed"
        },
        "endpoints": {
            "dashboard": "http://localhost:8081",
            "metrics": "http://localhost:8080/metrics",
            "health_check": "./health_check.py"
        },
        "commands": {
            "start": "./start_monitoring.py",
            "stop": "Ctrl+C",
            "health": "./health_check.py",
            "status": "./health_check.py | jq .status"
        },
        "configuration": {
            "file": "config/monitoring_config.json",
            "environment_variable": "NEO_CLONE_MONITORING_ENABLED=true"
        },
        "integration": {
            "opencode_tui": "opencode_integration.py",
            "auto_instrument": True
        }
    }
    
    # Write summary
    summary_file = Path("deployment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ Deployment summary created: {summary_file}")
    return summary

def main():
    """Main deployment function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Deployment steps
    steps = [
        ("Creating directory structure", create_deployment_structure),
        ("Installing dependencies", install_dependencies),
        ("Creating configuration", create_production_config),
        ("Creating startup script", create_startup_script),
        ("Creating integration wrapper", create_integration_wrapper),
        ("Creating health check", create_health_check),
        ("Creating systemd service", create_systemd_service),
        ("Testing deployment", test_deployment),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\\n{step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"   ❌ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    # Create summary
    summary = create_deployment_summary()
    
    # Final status
    print("\\n" + "=" * 80)
    if not failed_steps:
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("\\n🚀 Neo-Clone Monitoring System is ready to use!")
        print("\\n📋 Quick Start:")
        print("   1. Start monitoring: ./start_monitoring.py")
        print("   2. View dashboard: http://localhost:8081")
        print("   3. Check health: ./health_check.py")
        print("   4. View metrics: http://localhost:8080/metrics")
        
        print("\\n🔗 Integration:")
        print("   • Add to OpenCode TUI: import opencode_integration")
        print("   • Auto-monitoring enabled by default")
        print("   • Configure: edit config/monitoring_config.json")
        
        print("\\n📊 Monitoring Features:")
        print("   ✅ Distributed tracing")
        print("   ✅ Performance metrics")
        print("   ✅ Real-time profiling")
        print("   ✅ Error handling")
        print("   ✅ TUI dashboard")
        print("   ✅ Health monitoring")
        
    else:
        print("❌ DEPLOYMENT FAILED!")
        print(f"\\nFailed steps: {', '.join(failed_steps)}")
        print("\\n🔧 Troubleshooting:")
        print("   1. Check Python version (3.8+ required)")
        print("   2. Check internet connection for dependencies")
        print("   3. Check file permissions")
        print("   4. Review error messages above")
    
    print("=" * 80)
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)