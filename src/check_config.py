#!/usr/bin/env python3
"""
Check Config object attributes
"""

import sys
from pathlib import Path

# Add neo-clone to path
neo_clone_path = Path(__file__).parent / "neo-clone"
sys.path.insert(0, str(neo_clone_path))

from config import get_config

config = get_config()
print("Config object attributes:")
for attr in dir(config):
    if not attr.startswith('_'):
        try:
            value = getattr(config, attr)
            print(f"  {attr}: {value}")
        except:
            print(f"  {attr}: <unable to access>")