"""
Setup script for Neo-Clone Monitoring System

This script provides dependency management and installation for the monitoring system
with optional dependencies based on user requirements.
"""

from setuptools import setup, find_packages
import sys
import os

# Version information
VERSION = "1.0.0"

# Core dependencies (always required)
CORE_REQUIREMENTS = [
    "asyncio-throttle>=1.0.2",
]

# Optional dependencies by feature
OPTIONAL_REQUIREMENTS = {
    "tracing": [
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-exporter-jaeger>=1.20.0",
        "opentelemetry-exporter-otlp>=1.20.0",
        "opentelemetry-propagator-b3>=1.20.0",
        "opentelemetry-propagator-jaeger>=1.20.0",
    ],
    "metrics": [
        "prometheus-client>=0.17.0",
        "statsd>=4.0.0",
    ],
    "profiling": [
        "psutil>=5.9.0",
        "memory-profiler>=0.60.0",
        "pyinstrument>=4.4.0",
        "py-spy>=0.3.14",
    ],
    "dashboard": [
        "textual>=0.41.0",
        "rich>=13.0.0",
    ],
    "advanced": [
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "aiofiles>=23.0.0",
        "aiohttp>=3.8.0",
    ],
    "logging": [
        "structlog>=23.0.0",
        "colorlog>=6.7.0",
    ],
    "monitoring-stack": [
        "grafana-api>=1.0.3",
        "elasticsearch>=8.0.0",
        "redis>=4.5.0",
        "kafka-python>=2.0.0",
    ]
}

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

# All optional dependencies combined
ALL_OPTIONAL = []
for deps in OPTIONAL_REQUIREMENTS.values():
    ALL_OPTIONAL.extend(deps)

# Common combinations
COMMON_REQUIREMENTS = {
    "basic": CORE_REQUIREMENTS + OPTIONAL_REQUIREMENTS["profiling"],
    "standard": CORE_REQUIREMENTS + OPTIONAL_REQUIREMENTS["tracing"] + OPTIONAL_REQUIREMENTS["metrics"] + OPTIONAL_REQUIREMENTS["profiling"],
    "full": CORE_REQUIREMENTS + ALL_OPTIONAL,
    "development": CORE_REQUIREMENTS + ALL_OPTIONAL + DEV_REQUIREMENTS,
}

def read_readme():
    """Read README file"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Neo-Clone Monitoring System"

def get_requirements():
    """Get requirements based on command line arguments"""
    # Default to basic requirements
    requirements = CORE_REQUIREMENTS
    
    # Check for extra requirements arguments
    extras_requested = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("[") and arg.endswith("]"):
                extras_requested.extend(arg[1:-1].split(","))
            elif arg in OPTIONAL_REQUIREMENTS:
                extras_requested.append(arg)
    
    # Add requested extras
    for extra in extras_requested:
        if extra in OPTIONAL_REQUIREMENTS:
            requirements.extend(OPTIONAL_REQUIREMENTS[extra])
        elif extra in COMMON_REQUIREMENTS:
            requirements = COMMON_REQUIREMENTS[extra]
            break
    
    return list(set(requirements))  # Remove duplicates

setup(
    name="neo-clone-monitoring",
    version=VERSION,
    description="Comprehensive monitoring system for Neo-Clone AI agent",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Neo-Clone Team",
    author_email="team@neo-clone.ai",
    url="https://github.com/neo-clone/monitoring",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Networking :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require=OPTIONAL_REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "neo-clone-monitor=neo_clone.monitoring.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)