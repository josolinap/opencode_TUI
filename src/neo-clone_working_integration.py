#!/usr/bin/env python3
"""
Working Neo-Clone + OpenCode TUI Integration
Uses the correct OpenCode CLI interface
"""

import subprocess
import json
import sys
import tempfile
from pathlib import Path

def run_neo_clone_opencode(prompt, model="opencode/big-pickle"):
    """Run Neo-Clone using OpenCode TUI with correct CLI syntax"""
    
    try:
        # Use the correct OpenCode run command
        cmd = [
            r'C:\Users\JO\.opencode\bin\opencode.exe',
            'run',
            prompt,
            '--model', model,
            '--format', 'json'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd='C:\\Users\\JO\\opencode_TUI'
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "model_used": model,
            "prompt": prompt,
            "integration_method": "opencode_cli_run"
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out after 60 seconds",
            "model_used": model,
            "prompt": prompt
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_used": model,
            "prompt": prompt
        }

def test_all_models():
    """Test all available OpenCode models"""
    models = [
        "opencode/big-pickle",
        "opencode/grok-code", 
        "opencode/gpt-5-nano"
    ]
    
    results = {}
    
    for model in models:
        print(f"\nTesting model: {model}")
        result = run_neo_clone_opencode(f"Hello test from Neo-Clone using {model}", model)
        results[model] = result
        
        if result["success"]:
            print(f"  SUCCESS: {model} is working!")
            if result["stdout"]:
                print(f"  Output: {result['stdout'][:200]}...")
        else:
            print(f"  FAILED: {model}")
            print(f"  Error: {result.get('error', result.get('stderr', 'Unknown error'))}")
    
    return results

def create_neo_clone_agent():
    """Create a Neo-Clone specific agent configuration"""
    
    agent_config = {
        "name": "neo-clone-agent",
        "description": "Neo-Clone AI Agent with enhanced capabilities",
        "tools": {
            "neo-clone": True,
            "bash": True,
            "webfetch": True,
            "file": True,
            "edit": True
        },
        "permission": {
            "edit": "allow",
            "bash": {"*": "allow"},
            "webfetch": "allow"
        },
        "model": "opencode/big-pickle",
        "skills": [
            "code_generation",
            "text_analysis", 
            "data_inspector",
            "ml_training",
            "file_manager",
            "web_search",
            "minimax_agent"
        ]
    }
    
    # Save agent config
    agent_path = Path("neo-clone-agent.json")
    with open(agent_path, 'w') as f:
        json.dump(agent_config, f, indent=2)
    
    print(f"Neo-Clone agent configuration created: {agent_path}")
    return agent_path

def main():
    print("=" * 60)
    print("NEO-CLONE + OPENCODE TUI WORKING INTEGRATION")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python neo-clone_working_integration.py 'your prompt here'")
        print("  python neo-clone_working_integration.py --test-all")
        print("  python neo-clone_working_integration.py --create-agent")
        return
    
    command = sys.argv[1]
    
    if command == "--test-all":
        print("\nTesting all OpenCode models with Neo-Clone...")
        results = test_all_models()
        
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        working_models = []
        failed_models = []
        
        for model, result in results.items():
            if result["success"]:
                working_models.append(model)
            else:
                failed_models.append(model)
        
        print(f"\nWorking Models ({len(working_models)}):")
        for model in working_models:
            print(f"  ✓ {model}")
        
        print(f"\nFailed Models ({len(failed_models)}):")
        for model in failed_models:
            print(f"  ✗ {model}")
        
        # Save results
        with open("neo-clone_test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: neo-clone_test_results.json")
        
    elif command == "--create-agent":
        print("\nCreating Neo-Clone agent configuration...")
        agent_path = create_neo_clone_agent()
        print(f"\nAgent created successfully!")
        print(f"Configuration: {agent_path}")
        print("\nTo use the agent:")
        print(f'  opencode run --agent neo-clone-agent "your prompt here"')
        
    else:
        # Run with provided prompt
        prompt = " ".join(sys.argv[1:])
        print(f"\nRunning Neo-Clone with prompt: {prompt}")
        
        result = run_neo_clone_opencode(prompt)
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        
        print(f"Success: {result['success']}")
        print(f"Model: {result['model_used']}")
        print(f"Method: {result['integration_method']}")
        
        if result["success"]:
            if result["stdout"]:
                print(f"\nOutput:\n{result['stdout']}")
            else:
                print("\nOutput: (no output - command may be running interactively)")
        else:
            print(f"\nError: {result.get('error', result.get('stderr', 'Unknown error'))}")
        
        # Save result
        with open("neo-clone_last_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResult saved to: neo-clone_last_result.json")

if __name__ == "__main__":
    main()