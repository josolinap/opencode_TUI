#!/usr/bin/env python3
"""
Show expected models in OpenCode /models command
"""

import json

def show_expected_models():
    """Show what models should appear in /models"""
    print("Expected Models in OpenCode /models Command")
    print("=" * 60)
    
    try:
        with open("opencode.json", 'r') as f:
            config = json.load(f)
        
        providers = config.get('providers', {})
        
        print("Google Models (10 total):")
        print("-" * 30)
        
        google_models = providers.get('google', {}).get('models', {})
        for alias, model_name in google_models.items():
            print(f"  {alias:<20} -> {model_name}")
        
        print("\nOpenCode Built-in Models (3 total):")
        print("-" * 30)
        
        opencode_models = providers.get('opencode', {}).get('models', {})
        for alias, model_name in opencode_models.items():
            print(f"  {alias:<20} -> {model_name}")
        
        print(f"\nTotal Models Available: {len(google_models) + len(opencode_models)}")
        
        print("\n" + "=" * 60)
        print("Model Aliases (Quick Switch):")
        print("-" * 30)
        
        aliases = config.get('models', {})
        for alias, model in aliases.items():
            print(f"  {alias:<15} -> {model}")
        
        print("\n" + "=" * 60)
        print("How to Use:")
        print("1. Run: opencode")
        print("2. Type: /models")
        print("3. You should see all 13 models listed above")
        print("4. Switch with: /model <alias>")
        print("   Examples:")
        print("   /model gemini-flash")
        print("   /model gemini-pro")
        print("   /model gemma-large")
        print("   /model big-pickle")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    show_expected_models()