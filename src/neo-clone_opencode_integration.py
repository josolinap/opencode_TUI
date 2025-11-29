#!/usr/bin/env python3
"""
Neo-Clone + OpenCode TUI Integration Script
Actually working integration using CLI interface
"""

import subprocess
import json
import sys
from pathlib import Path

def run_neo_clone_with_opencode(prompt, model="big-pickle"):
    """Run Neo-Clone using OpenCode CLI"""
    
    # Create temporary agent config
    agent_config = {
        "name": "neo-clone-integrated",
        "description": "Neo-Clone with OpenCode integration",
        "mode": "primary", 
        "model": {
            "modelID": model,
            "providerID": "anthropic"
        },
        "permission": {
            "edit": "allow",
            "bash": {"*": "allow"},
            "webfetch": "allow"
        },
        "tools": {
            "neo-clone": True,
            "bash": True,
            "webfetch": True,
            "file": True
        },
        "prompt": prompt
    }
    
    # Save agent config
    config_path = Path("temp_neo_clone_agent.json")
    with open(config_path, 'w') as f:
        json.dump(agent_config, f, indent=2)
    
    try:
        # Run OpenCode with the agent
result = subprocess.run([
            'C:\Users\JO\.opencode\bin\opencode.exe', 'agent', 'create', 
            '--config', str(config_path)
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return {
                "success": True,
                "response": result.stdout,
                "model_used": model,
                "integration": "opencode_cli"
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "model_used": model
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_used": model
        }
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = run_neo_clone_with_opencode(prompt)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python integration_script.py 'your prompt here'")
