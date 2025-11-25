"""
Neo-OSINT: Enhanced AI-Powered OSINT Tool for Neo-Clone
An advanced threat intelligence and investigation system that expands upon Robin's capabilities
with integrated Neo-Clone AI brain, advanced analytics, and modular extensibility.
"""

__version__ = "1.0.0"
__author__ = "Neo-Clone AI"

from .core.engine import NeoOSINTEngine
from .core.config import NeoOSINTConfig
from .plugins.manager import PluginManager

__all__ = ["NeoOSINTEngine", "NeoOSINTConfig", "PluginManager"]