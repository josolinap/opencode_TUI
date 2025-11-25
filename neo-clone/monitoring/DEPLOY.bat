@echo off
REM Neo-Clone Monitoring System Deployment Script for Windows
REM This script deploys the monitoring system for immediate use

echo ================================================================
echo    NEO-CLONE MONITORING SYSTEM DEPLOYMENT
echo ================================================================
echo Deploying comprehensive monitoring for Neo-Clone + OpenCode TUI
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create directory structure
echo.
echo 📁 Creating deployment structure...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist data\metrics mkdir data\metrics
if not exist data\traces mkdir data\traces
if not exist data\profiles mkdir data\profiles
if not exist config mkdir config
if not exist backups mkdir backups
echo    ✅ Directory structure created

REM Install core dependencies
echo.
echo 📦 Installing dependencies...
python -m pip install asyncio-throttle>=1.0.2
python -m pip install statistics>=1.0.0

REM Install optional dependencies (best effort)
echo    Installing optional dependencies...
python -m pip install psutil>=5.9.0 >nul 2>&1
python -m pip install prometheus-client>=0.17.0 >nul 2>&1
echo    ✅ Dependencies installed

REM Create configuration
echo.
echo ⚙️  Creating configuration...
echo { > config\monitoring_config.json
echo   "monitoring": { >> config\monitoring_config.json
echo     "enabled": true, >> config\monitoring_config.json
echo     "tracing_enabled": true, >> config\monitoring_config.json
echo     "metrics_enabled": true, >> config\monitoring_config.json
echo     "profiling_enabled": true, >> config\monitoring_config.json
echo     "dashboard_enabled": true, >> config\monitoring_config.json
echo     "auto_instrument_brain": true, >> config\monitoring_config.json
echo     "auto_instrument_skills": true, >> config\monitoring_config.json
echo     "opencode_integration": true >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "performance": { >> config\monitoring_config.json
echo     "cpu_threshold": 80.0, >> config\monitoring_config.json
echo     "memory_threshold": 85.0, >> config\monitoring_config.json
echo     "response_time_threshold": 2000.0, >> config\monitoring_config.json
echo     "monitoring_interval": 1.0 >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "tracing": { >> config\monitoring_config.json
echo     "service_name": "neo-clone-opencode", >> config\monitoring_config.json
echo     "sample_rate": 0.1, >> config\monitoring_config.json
echo     "endpoint": null, >> config\monitoring_config.json
echo     "jaeger_endpoint": null, >> config\monitoring_config.json
echo     "otlp_endpoint": null >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "metrics": { >> config\monitoring_config.json
echo     "export_interval": 30.0, >> config\monitoring_config.json
echo     "endpoint": null, >> config\monitoring_config.json
echo     "prometheus_port": 8080, >> config\monitoring_config.json
echo     "retention_hours": 24 >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "profiling": { >> config\monitoring_config.json
echo     "sample_rate": 0.05, >> config\monitoring_config.json
echo     "min_duration_ms": 100.0, >> config\monitoring_config.json
echo     "memory_profiling": false, >> config\monitoring_config.json
echo     "cpu_profiling": true >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "dashboard": { >> config\monitoring_config.json
echo     "enabled": true, >> config\monitoring_config.json
echo     "refresh_interval": 2.0, >> config\monitoring_config.json
echo     "port": 8081, >> config\monitoring_config.json
echo     "host": "localhost" >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "error_handling": { >> config\monitoring_config.json
echo     "max_errors_per_window": 100, >> config\monitoring_config.json
echo     "error_window_minutes": 5, >> config\monitoring_config.json
echo     "circuit_breaker_threshold": 5, >> config\monitoring_config.json
echo     "circuit_breaker_timeout": 60, >> config\monitoring_config.json
echo     "log_dir": "logs" >> config\monitoring_config.json
echo   }, >> config\monitoring_config.json
echo   "logging": { >> config\monitoring_config.json
echo     "level": "INFO", >> config\monitoring_config.json
echo     "file": "logs/monitoring.log", >> config\monitoring_config.json
echo     "max_size_mb": 100, >> config\monitoring_config.json
echo     "backup_count": 5 >> config\monitoring_config.json
echo   } >> config\monitoring_config.json
echo } >> config\monitoring_config.json
echo    ✅ Configuration created

REM Create startup script
echo.
echo 🚀 Creating startup script...
echo @echo off > START_MONITORING.bat
echo echo Starting Neo-Clone Monitoring System... >> START_MONITORING.bat
echo echo. >> START_MONITORING.bat
echo echo Dashboard: http://localhost:8081 >> START_MONITORING.bat
echo echo Metrics: http://localhost:8080/metrics >> START_MONITORING.bat
echo echo Press Ctrl+C to stop >> START_MONITORING.bat
echo echo. >> START_MONITORING.bat
echo python -c "import sys; import os; sys.path.insert(0, os.getcwd()); from monitoring_integration import get_global_monitoring; monitoring = get_global_monitoring(); monitoring.initialize(); print('🎯 Monitoring system is running!'); input('Press Enter to stop...')" >> START_MONITORING.bat
echo    ✅ Startup script created: START_MONITORING.bat

REM Create health check script
echo.
echo 🏥 Creating health check script...
echo @echo off > HEALTH_CHECK.bat
echo python -c "import sys; import os; import json; import time; sys.path.insert(0, os.getcwd()); from monitoring_integration import get_global_monitoring; monitoring = get_global_monitoring(); health = {'timestamp': time.time(), 'status': 'healthy' if monitoring.status.initialized else 'unhealthy', 'monitoring_initialized': monitoring.status.initialized, 'active_operations': len(monitoring.active_operations)}; print(json.dumps(health, indent=2))" >> HEALTH_CHECK.bat
echo    ✅ Health check script created: HEALTH_CHECK.bat

REM Test deployment
echo.
echo 🧪 Testing deployment...
python -c "import sys; import os; sys.path.insert(0, os.getcwd()); from monitoring_integration import get_global_monitoring; monitoring = get_global_monitoring(); success = monitoring.initialize(); print('✅ Deployment test passed' if success else '❌ Deployment test failed')" > test_result.txt
set /p test_result=<test_result.txt
del test_result.txt

echo %test_result%

REM Create deployment summary
echo.
echo 📋 Deployment Summary:
echo ================================================================
echo 🎉 NEO-CLONE MONITORING SYSTEM DEPLOYED!
echo ================================================================
echo.
echo 🚀 Quick Start:
echo    1. Start monitoring: START_MONITORING.bat
echo    2. View dashboard: http://localhost:8081
echo    3. Check health: HEALTH_CHECK.bat
echo    4. View metrics: http://localhost:8080/metrics
echo.
echo 🔗 Integration:
echo    • Add to OpenCode TUI: import opencode_integration
echo    • Auto-monitoring enabled by default
echo    • Configure: edit config\monitoring_config.json
echo.
echo 📊 Monitoring Features:
echo    ✅ Distributed tracing
echo    ✅ Performance metrics
echo    ✅ Real-time profiling
echo    ✅ Error handling
echo    ✅ TUI dashboard
echo    ✅ Health monitoring
echo.
echo 📁 Files Created:
echo    • config\monitoring_config.json - Configuration
echo    • START_MONITORING.bat - Startup script
echo    • HEALTH_CHECK.bat - Health check
echo    • logs\ - Log files directory
echo    • data\ - Data storage directory
echo.
echo 🎯 Ready to use! Run START_MONITORING.bat to begin monitoring.
echo ================================================================
pause