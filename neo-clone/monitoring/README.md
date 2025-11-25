# Neo-Clone + OpenCode TUI Monitoring System

A comprehensive distributed tracing and monitoring system for the Neo-Clone AI agent integrated with OpenCode TUI.

## Overview

This monitoring system provides real-time visibility into Neo-Clone operations, performance metrics, distributed tracing, and bottleneck detection. It's designed to work seamlessly with the OpenCode TUI agent selection system.

## Features

### 🔍 Distributed Tracing

- OpenTelemetry-based distributed tracing with fallback mock implementation
- Automatic span creation for brain operations and skill execution
- Trace correlation across multiple operations
- Support for Jaeger and OTLP exporters

### 📊 Metrics Collection

- Real-time performance metrics (CPU, memory, response times)
- Custom metrics recording
- Prometheus-compatible metrics export
- Historical data analysis and trend detection

### 🎯 Performance Profiling

- CPU and memory profiling with cProfile
- Bottleneck detection and classification
- Performance trend analysis
- Optimization recommendations

### 📈 TUI Dashboard

- Real-time monitoring dashboard integrated with OpenCode TUI
- Interactive performance visualizations
- Bottleneck alerts and recommendations
- System health monitoring

### 🧠 Brain Operations Monitoring

- Automatic instrumentation of Neo-Clone brain operations
- Intent analysis and reasoning trace monitoring
- Memory system performance tracking
- Skill execution monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenCode TUI                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Agent Selector  │  │ Model Selector  │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Monitoring Integration Layer                   │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Distributed     │  │ Metrics         │                  │
│  │ Tracing         │  │ Collector       │                  │
│  └─────────────────┘  └─────────────────┘                  │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Performance     │  │ TUI Dashboard   │                  │
│  │ Profiler        │  │                 │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Neo-Clone Brain                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Brain Operations│  │ Skills System   │                  │
│  │ Monitor         │  │ Monitor         │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Dependencies

The monitoring system is designed to work with optional dependencies. Install based on your needs:

```bash
# Core monitoring (always available)
# No additional dependencies required - uses mock implementations

# Full OpenTelemetry support
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger opentelemetry-exporter-otlp

# Advanced profiling
pip install memory-profiler pyinstrument psutil

# TUI Dashboard
pip install textual rich

# Metrics export
pip install prometheus-client
```

### Setup

1. The monitoring system automatically initializes when Neo-Clone starts
2. Configuration is handled through the `MonitoringConfig` class
3. All components have fallback implementations for development

## Usage

### Basic Monitoring

```python
from neo_clone.monitoring import get_global_monitoring, MonitoredOperation

# Get the global monitoring instance
monitoring = get_global_monitoring()

# Monitor an operation manually
operation_id = monitoring.start_operation("custom_operation", metadata={"user": "test"})
# ... your code here ...
result = monitoring.end_operation(operation_id, success=True)

# Use context manager for automatic monitoring
with MonitoredOperation(monitoring, "brain_operation"):
    # Your operation code here
    result = some_brain_function()
```

### Decorator-based Monitoring

```python
from neo_clone.monitoring import monitor_operation

@monitor_operation("skill_execution")
def my_skill_function(param1, param2):
    # This function will be automatically monitored
    return process_data(param1, param2)
```

### Custom Metrics

```python
# Record custom metrics
monitoring.record_metric("custom_counter", 1.0, tags={"type": "success"})
monitoring.record_metric("processing_time", 150.5, tags={"operation": "data_analysis"})

# Record events
monitoring.record_event("user_action", {"action": "login", "user_id": "123"})
```

### Performance Profiling

```python
from neo_clone.monitoring import get_global_profiler, ProfileOperation

profiler = get_global_profiler()

# Profile a specific operation
with ProfileOperation(profiler, "heavy_computation", "data_processing"):
    result = heavy_computation_function()

# Get performance analysis
analysis = profiler.analyze_performance_trends()
bottlenecks = profiler.get_bottleneck_summary()
```

### Integration with OpenCode TUI

The monitoring system automatically integrates with OpenCode TUI agent selection:

```python
# The neo-clone tool is automatically wrapped with monitoring
neo_clone("Use code_generation skill to create a neural network", "tool")

# Model selection is also monitored
model_selector("Complex data analysis", {"reasoning": True})
```

## Configuration

### MonitoringConfig

```python
from neo_clone.monitoring import MonitoringConfig, initialize_monitoring

config = MonitoringConfig(
    enabled=True,
    tracing_enabled=True,
    metrics_enabled=True,
    profiling_enabled=True,
    dashboard_enabled=True,

    # Tracing settings
    tracing_service_name="neo-clone-production",
    tracing_endpoint="http://jaeger:14268/api/traces",
    tracing_sample_rate=0.1,  # Sample 10% of traces

    # Metrics settings
    metrics_endpoint="http://prometheus:9090",
    metrics_export_interval=30.0,

    # Performance thresholds
    cpu_threshold=80.0,
    memory_threshold=85.0,
    response_time_threshold=2000.0,
)

# Initialize with custom config
monitoring = initialize_monitoring(config)
```

## Components

### 1. Distributed Tracing (`distributed_tracing.py`)

Provides OpenTelemetry-based distributed tracing with automatic fallback to mock implementation.

**Features:**

- Automatic span creation and management
- Trace context propagation
- Support for multiple exporters (Jaeger, OTLP)
- Mock implementation for development

**Usage:**

```python
from neo_clone.monitoring.distributed_tracing import get_global_tracer

tracer = get_global_tracer()
span_id = tracer.start_span("operation", "operation_id", {"key": "value"})
tracer.end_span(span_id, True, {"result": "success"})
```

### 2. Metrics Collection (`metrics_collector.py`)

Collects and exports performance metrics with Prometheus compatibility.

**Features:**

- Real-time metrics collection
- Custom metric recording
- Historical data analysis
- Alert generation

**Usage:**

```python
from neo_clone.monitoring.metrics_collector import get_global_collector

collector = get_global_collector()
collector.record_metric("operation_duration", 150.0, {"operation": "brain"})
```

### 3. Performance Profiler (`performance_profiler.py`)

Advanced performance profiling with bottleneck detection.

**Features:**

- CPU and memory profiling
- Bottleneck detection and classification
- Performance trend analysis
- Optimization recommendations

**Usage:**

```python
from neo_clone.monitoring.performance_profiler import get_global_profiler

profiler = get_global_profiler()
profile_id = profiler.start_operation_profiling("op_id", "operation_type")
# ... operation ...
profile = profiler.stop_operation_profiling(profile_id)
```

### 4. TUI Dashboard (`tui_dashboard.py`)

Real-time monitoring dashboard integrated with OpenCode TUI.

**Features:**

- Real-time performance visualization
- Interactive charts and graphs
- Bottleneck alerts
- System health monitoring

**Usage:**

```python
from neo_clone.monitoring.tui_dashboard import MonitoringDashboard

dashboard = MonitoringDashboard()
dashboard.start()
```

### 5. Brain Integration (`brain_integration.py`)

Automatic instrumentation of Neo-Clone brain operations.

**Features:**

- Intent analysis monitoring
- Reasoning trace tracking
- Memory system performance
- Automatic operation wrapping

### 6. Skills Integration (`skills_integration.py`)

Monitoring for Neo-Clone skills execution.

**Features:**

- Skill execution tracing
- Performance metrics per skill
- Error tracking and analysis
- Skill usage statistics

## Monitoring Data

### Traces

Distributed traces provide end-to-end visibility into operations:

```json
{
  "trace_id": "abc123...",
  "spans": [
    {
      "span_id": "def456...",
      "parent_span_id": null,
      "operation_name": "brain_operation",
      "start_time": "2025-01-15T10:30:00Z",
      "duration_ms": 1500,
      "status": "success",
      "tags": { "skill": "code_generation", "model": "claude-3-5-sonnet" }
    }
  ]
}
```

### Metrics

Performance metrics are collected and exported:

```json
{
  "cpu_usage": 45.2,
  "memory_usage": 67.8,
  "operation_duration": {
    "avg": 1200.0,
    "p95": 2500.0,
    "p99": 5000.0
  },
  "operation_count": {
    "total": 1250,
    "success": 1180,
    "error": 70
  }
}
```

### Performance Profiles

Detailed performance analysis:

```json
{
  "operation_id": "brain_op_123",
  "duration_ms": 1500,
  "slow_functions": [
    { "function": "process_large_data", "time_ms": 800 },
    { "function": "generate_response", "time_ms": 400 }
  ],
  "memory_leaks": [],
  "recommendations": ["Consider optimizing process_large_data function", "Implement caching for repeated operations"]
}
```

## Bottleneck Detection

The system automatically detects and classifies performance bottlenecks:

### Categories

- **CPU**: High CPU usage, inefficient algorithms
- **Memory**: Memory leaks, high memory usage
- **IO**: Slow file operations, network latency
- **Network**: API call delays, bandwidth issues
- **Algorithm**: Inefficient algorithms, poor complexity

### Severity Levels

- **Low**: Minor performance impact
- **Medium**: Noticeable performance degradation
- **High**: Significant performance issues
- **Critical**: System-impacting problems

### Recommendations

The system provides actionable recommendations:

```json
{
  "bottleneck": {
    "type": "cpu",
    "severity": "high",
    "description": "High CPU usage during data processing",
    "affected_operations": ["data_analysis", "report_generation"]
  },
  "recommendations": [
    "Implement parallel processing for data analysis",
    "Consider using more efficient algorithms",
    "Add caching for frequently accessed data"
  ]
}
```

## Health Monitoring

System health is continuously monitored:

```python
from neo_clone.monitoring import get_health_status

health = get_health_status()
# Returns:
# {
#   "status": "healthy",  # healthy, degraded, unhealthy
#   "score": 95,          # 0-100 health score
#   "details": {...}      # Detailed health information
# }
```

## Export and Reporting

### Export Monitoring Data

```python
from neo_clone.monitoring import create_monitoring_report

# Get comprehensive report
report = create_monitoring_report()

# Export to file
with open("monitoring_report.json", "w") as f:
    f.write(report)
```

### Performance Reports

```python
from neo_clone.monitoring.performance_profiler import get_global_profiler

profiler = get_global_profiler()
performance_report = profiler.export_performance_report()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Missing optional dependencies
   - Solution: Install required packages or use mock implementations

2. **High Memory Usage**: Monitoring overhead
   - Solution: Adjust sampling rates, disable profiling for production

3. **Missing Traces**: Tracing not working
   - Solution: Check tracer configuration, verify endpoint connectivity

4. **Dashboard Not Starting**: TUI dependencies missing
   - Solution: Install textual and rich packages

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("neo_clone.monitoring").setLevel(logging.DEBUG)
```

### Performance Tuning

Adjust monitoring overhead:

```python
config = MonitoringConfig(
    tracing_sample_rate=0.01,  # Only 1% of operations traced
    profiling_sample_rate=0.05,  # Only 5% of operations profiled
    metrics_export_interval=60.0,  # Export every minute
)
```

## Development

### Adding New Monitoring

1. Create new monitoring component in the monitoring directory
2. Import and integrate in `monitoring_integration.py`
3. Add configuration options to `MonitoringConfig`
4. Update documentation

### Testing

Run monitoring tests:

```bash
python -m pytest neo_clone/monitoring/tests/
```

### Mock Implementation

The system includes comprehensive mock implementations for development without external dependencies.

## Production Deployment

### Recommended Settings

```python
config = MonitoringConfig(
    tracing_sample_rate=0.01,  # Low sampling for production
    profiling_sample_rate=0.001,  # Minimal profiling
    metrics_export_interval=60.0,
    cpu_threshold=90.0,  # Higher thresholds for production
    memory_threshold=90.0,
    response_time_threshold=5000.0,
)
```

### Monitoring Stack

Recommended production monitoring stack:

- **Tracing**: Jaeger or Tempo
- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack
- **Alerting**: AlertManager

## API Reference

### MonitoringIntegration

Main class for monitoring system integration.

**Methods:**

- `initialize()`: Initialize monitoring components
- `start_operation()`: Start monitoring an operation
- `end_operation()`: End monitoring an operation
- `record_metric()`: Record custom metric
- `record_event()`: Record custom event
- `get_monitoring_summary()`: Get comprehensive summary
- `export_monitoring_data()`: Export all data
- `shutdown()`: Cleanup and shutdown

### Global Functions

- `get_global_monitoring()`: Get global monitoring instance
- `initialize_monitoring()`: Initialize with custom config
- `monitor_operation()`: Decorator for function monitoring
- `get_health_status()`: Get system health status
- `create_monitoring_report()`: Create comprehensive report

## License

This monitoring system is part of the Neo-Clone project and follows the same license terms.

## Contributing

Contributions to the monitoring system are welcome! Please follow the project's contribution guidelines and ensure all tests pass.

---

For more information, see the individual component documentation and the main Neo-Clone project documentation.
