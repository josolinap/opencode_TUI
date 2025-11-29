#!/usr/bin/env python3
"""
Simple Model Availability Check
"""

import asyncio
import sys
import os

# Add neo-clone to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neo-clone'))

from active_model_detector import ActiveModelDetector

async def check_models():
    """Check what models are actually available"""
    print("=" * 60)
    print("CHECKING CURRENT MODEL AVAILABILITY")
    print("=" * 60)
    
    detector = ActiveModelDetector()
    
    print("\n1. Testing model availability...")
    models = await detector.detect_available_models(force_refresh=True)
    
    print("\n2. Available Models:")
    print("-" * 40)
    
    available_count = 0
    total_count = 0
    
    # Check all candidate models
    for model_name in detector.model_priority:
        total_count += 1
        if model_name in models:
            model_info = models[model_name]
            if model_info.available:
                print(f"  AVAILABLE: {model_name} ({model_info.provider}) - {model_info.cost}")
                available_count += 1
            else:
                print(f"  NOT AVAILABLE: {model_name} ({model_info.provider}) - {model_info.cost}")
        else:
            print(f"  NOT DETECTED: {model_name}")
    
    print("-" * 40)
    print(f"Summary: {available_count}/{total_count} models available")
    
    print("\n3. Working Models Details:")
    print("-" * 40)
    for name, info in models.items():
        if info.available:
            print(f"\nModel: {name}")
            print(f"  Provider: {info.provider}")
            print(f"  Cost: {info.cost}")
            print(f"  Context: {info.context_length}")
            print(f"  Capabilities: {', '.join(info.capabilities)}")
    
    return available_count

if __name__ == "__main__":
    available_count = asyncio.run(check_models())
    
    print("\n" + "=" * 60)
    if available_count > 0:
        print(f"SUCCESS: Found {available_count} working models!")
    else:
        print("ISSUE: No working models found!")
        print("Need to update model database to only include working models.")
    print("=" * 60)