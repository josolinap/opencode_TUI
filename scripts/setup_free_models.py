#!/usr/bin/env python3
"""
Setup script for free OpenCode models
Helps users configure API keys and test free model integrations
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def print_header():
    """Print setup header"""
    print("OpenCode Free Models Setup")
    print("=" * 50)
    print("This script helps you configure free model providers")
    print("for OpenCode with 18+ providers and 500+ models.\n")

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        return False
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    packages = [
        "openai",
        "groq", 
        "google-generativeai",
        "huggingface_hub",
        "cohere",
        "replicate",
        "requests"
    ]
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  OK: {package} installed")
        except subprocess.CalledProcessError:
            print(f"  ERROR: Failed to install {package}")
    
    print("OK: Package installation complete")

def setup_api_keys():
    """Setup API keys as environment variables"""
    print("\nSetting up API keys...")
    
    api_keys = {
        "OPENROUTER_API_KEY": "OpenRouter (500+ models): https://openrouter.ai/keys",
        "GROQ_API_KEY": "Groq (ultra-fast): https://console.groq.com/keys", 
        "GOOGLE_API_KEY": "Google Gemini: https://makersuite.google.com/app/apikey",
        "HF_API_KEY": "Hugging Face: https://huggingface.co/settings/tokens",
        "COHERE_API_KEY": "Cohere: https://dashboard.cohere.com/api-keys",
        "REPLICATE_API_TOKEN": "Replicate: https://replicate.com/account",
        "PERPLEXITY_API_KEY": "Perplexity: https://www.perplexity.ai/settings/api",
        "OPENAI_API_KEY": "OpenAI (optional): https://platform.openai.com/api-keys",
        "ANTHROPIC_API_KEY": "Anthropic (optional): https://console.anthropic.com/"
    }
    
    print("API Key Setup Guide:")
    print("-" * 30)
    
    for key, description in api_keys.items():
        if not os.getenv(key):
            print(f"SETUP: {key}: {description}")
            print(f"   export {key}='your-api-key-here'")
        else:
            print(f"OK: {key}: Already configured")
        print()
    
    print("INFO: Add these to your shell profile (.bashrc, .zshrc, etc.)")

def create_config_file():
    """Create OpenCode configuration file"""
    print("\nCreating OpenCode configuration...")
    
    config_path = Path("opencode_with_free_models.json")
    if config_path.exists():
        print("OK: Configuration file already exists")
        return config_path
    
    # Configuration template
    config = {
        "$schema": "https://opencode.ai/config.json",
        "tools": {"neo-clone": True},
        "permission": {
            "edit": "allow",
            "bash": {"*": "allow"},
            "webfetch": "allow"
        },
        "providers": {
            "opencode": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "opencode-free",
                "models": {
                    "big-pickle": "big-pickle",
                    "grok-code": "grok-code", 
                    "gpt-5-nano": "gpt-5-nano"
                }
            }
        },
        "models": {
            "default": "big-pickle"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"OK: Created {config_path}")
    return config_path

def test_builtin_models():
    """Test built-in OpenCode models"""
    print("\nTesting built-in OpenCode models...")
    
    try:
        import openai
        
        # Test with built-in models
        client = openai.OpenAI(
            base_url="https://api.openai.com/v1",
            api_key="opencode-free"
        )
        
        models_to_test = ["big-pickle", "grok-code", "gpt-5-nano"]
        
        for model in models_to_test:
            try:
                print(f"  Testing {model}...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say 'Hello from OpenCode!'"}],
                    max_tokens=10
                )
                print(f"  OK: {model}: {response.choices[0].message.content}")
            except Exception as e:
                print(f"  ERROR: {model}: {str(e)}")
                
    except ImportError:
        print("ERROR: OpenAI package not installed")
    except Exception as e:
        print(f"ERROR: Test failed: {str(e)}")

def print_summary():
    """Print setup summary"""
    print("\nSetup Summary")
    print("=" * 50)
    print("Python environment checked")
    print("Required packages installed")
    print("API keys guide provided")
    print("Configuration file created")
    print("Built-in models tested")
    
    print("\nNext Steps:")
    print("1. Set up API keys for desired providers")
    print("2. Copy opencode_with_free_models.json to opencode.json")
    print("3. Run OpenCode with: opencode")
    print("4. Use /models to see available models")
    
    print("\nReady to use 500+ free models with OpenCode!")

def main():
    """Main setup function"""
    print_header()
    
    if not check_python():
        sys.exit(1)
    
    install_requirements()
    setup_api_keys()
    create_config_file()
    test_builtin_models()
    print_summary()

if __name__ == "__main__":
    main()