#!/usr/bin/env python3
"""
Check available Google Gemini models
"""

import google.generativeai as genai

def list_available_models():
    """List all available Gemini models"""
    print("Checking available Google Gemini models...")
    print("=" * 60)
    
    try:
        # Configure API
        api_key = "AIzaSyC5ubP3IGV0DOYipo3Pre2WQmb-6eP4w9c"
        genai.configure(api_key=api_key)
        
        # List models
        models = genai.list_models()
        
        print("Available Models:")
        print("-" * 40)
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"Name: {model.name}")
                print(f"Display Name: {model.display_name}")
                print(f"Description: {model.description}")
                print(f"Generation Methods: {model.supported_generation_methods}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    list_available_models()