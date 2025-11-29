#!/usr/bin/env python3
"""
Validate OpenCode model detection
"""

import json
import subprocess
import sys
import os

def validate_config():
    """Validate OpenCode configuration"""
    print("Validating OpenCode Configuration")
    print("=" * 50)
    
    config_path = "opencode.json"
    
    if not os.path.exists(config_path):
        print("ERROR: opencode.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("OK: Configuration file loaded successfully")
        
        # Check providers
        providers = config.get('providers', {})
        print(f"OK: Found {len(providers)} providers:")
        
        for provider_name, provider_config in providers.items():
            models = provider_config.get('models', {})
            print(f"  - {provider_name}: {len(models)} models")
            
            for model_alias, model_name in models.items():
                print(f"    * {model_alias} -> {model_name}")
        
        # Check default models
        default_models = config.get('models', {})
        print(f"\nOK: Found {len(default_models)} default model aliases:")
        
        for alias, model in default_models.items():
            print(f"  - {alias}: {model}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {str(e)}")
        return False

def test_opencode_detection():
    """Test if OpenCode can detect models"""
    print("\n" + "=" * 50)
    print("Testing OpenCode Model Detection")
    
    try:
        # Try to run opencode with --help to see if it's installed
        result = subprocess.run(['opencode', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("OK: OpenCode is installed and accessible")
            return True
        else:
            print("ERROR: OpenCode may not be properly installed")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: OpenCode command timed out")
        return False
    except FileNotFoundError:
        print("ERROR: OpenCode not found in PATH")
        print("Please install OpenCode first:")
        print("npm install -g opencode")
        return False
    except Exception as e:
        print(f"ERROR: Error testing OpenCode: {str(e)}")
        return False

def check_models_command():
    """Check if /models command should work"""
    print("\n" + "=" * 50)
    print("Model Detection Analysis")
    
    config_path = "opencode.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Count total models
        total_models = 0
        for provider_name, provider_config in config.get('providers', {}).items():
            models = provider_config.get('models', {})
            total_models += len(models)
        
        print(f"OK: Total models configured: {total_models}")
        
        # Check if Google provider is properly configured
        google_provider = config.get('providers', {}).get('google', {})
        if google_provider:
            api_key = google_provider.get('api_key', '')
            if api_key and api_key != 'your-api-key-here':
                print("OK: Google provider configured with API key")
                google_models = google_provider.get('models', {})
                print(f"OK: Google models available: {len(google_models)}")
            else:
                print("ERROR: Google provider missing API key")
        
        # Check default model
        default_model = config.get('models', {}).get('default')
        if default_model:
            print(f"OK: Default model set: {default_model}")
        else:
            print("ERROR: No default model set")
        
        print(f"\n📋 Expected models in /models command:")
        print("Google Models:")
        google_models = config.get('providers', {}).get('google', {}).get('models', {})
        for alias in google_models.keys():
            print(f"  - {alias}")
        
        print("\nOpenCode Models:")
        opencode_models = config.get('providers', {}).get('opencode', {}).get('models', {})
        for alias in opencode_models.keys():
            print(f"  - {alias}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error analyzing configuration: {str(e)}")
        return False

def main():
    """Main validation function"""
    print("OpenCode Model Detection Validator")
    print("================================")
    
    # Validate configuration
    if not validate_config():
        sys.exit(1)
    
    # Test OpenCode installation
    if not test_opencode_detection():
        print("\nINFO: To install OpenCode:")
        print("npm install -g opencode")
        sys.exit(1)
    
    # Check models command
    check_models_command()
    
    print("\n" + "=" * 50)
    print("Validation Complete!")
    print("\nNext Steps:")
    print("1. Run: opencode")
    print("2. Type: /models")
    print("3. You should see all configured models")
    print("4. Try: /model gemini-flash")
    print("5. Start chatting!")

if __name__ == "__main__":
    main()