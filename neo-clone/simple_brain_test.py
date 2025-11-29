#!/usr/bin/env python3
"""
Simple test for brain functionality
"""

import sys
import os
from pathlib import Path

# Add neo-clone to path
neo_clone_path = Path(__file__).parent
sys.path.insert(0, str(neo_clone_path))

def simple_test():
    """Simple brain test"""
    try:
        print("Testing brain import...")
        from brain import Brain
        print("SUCCESS: Brain imported")
        
        print("Creating brain instance...")
        cfg = type('Config', (), {
            'provider': 'ollama',
            'model': 'llama2', 
            'endpoint': 'http://localhost:11434'
        })()
        
        brain = Brain(cfg)
        print("SUCCESS: Brain instance created")
        
        print("Testing basic functionality...")
        # Test a simple method call
        if hasattr(brain, 'process'):
            print("SUCCESS: Brain has process method")
        else:
            print("INFO: Brain doesn't have process method")
            
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Simple Brain Test")
    print("=" * 30)
    
    success = simple_test()
    
    print("=" * 30)
    if success:
        print("TEST PASSED")
    else:
        print("TEST FAILED")