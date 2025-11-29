#!/usr/bin/env python3
"""
Check Current Model Availability
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
    
    print("\n1. Testing OpenCode models...")
    models = await detector.detect_available_models(force_refresh=True)
    
    print("\n2. Model Detection Results:")
    print("-" * 40)
    
    available_count = 0
    total_count = 0
    
    # Check all candidate models
    for model_name in detector.model_priority:
        total_count += 1
        if model_name in models:
            model_info = models[model_name]
            if model_info.available:
                print(f"  ✓ {model_name:25} ({model_info.provider:12}) - {model_info.cost:8} - AVAILABLE")
                available_count += 1
            else:
                print(f"  ✗ {model_name:25} ({model_info.provider:12}) - {model_info.cost:8} - NOT AVAILABLE")
        else:
            print(f"  ? {model_name:25} (unknown)        - unknown   - NOT DETECTED")
    
    print("-" * 40)
    print(f"Summary: {available_count}/{total_count} models available")
    
    print("\n3. Available Models Details:")
    print("-" * 40)
    for name, info in models.items():
        if info.available:
            print(f"\nModel: {name}")
            print(f"  Provider: {info.provider}")
            print(f"  Cost: {info.cost}")
            print(f"  Context: {info.context_length}")
            print(f"  Capabilities: {', '.join(info.capabilities)}")
            if info.endpoint:
                print(f"  Endpoint: {info.endpoint}")
    
    return available_count > 0

if __name__ == "__main__":
    success = asyncio.run(check_models())
    
    if success:
        print(f"\n✅ SUCCESS: Found working models!")
    else:
        print(f"\n❌ ISSUE: No working models found!")
        print("   This means we need to update the model database.")
    
    print("=" * 60)