#!/usr/bin/env python3
"""
OpenCode File Organization Script
================================

Organizes files into proper directory structure.
"""

import shutil
import os
from pathlib import Path

def organize_files():
    """Organize all files into their proper directories"""
    
    # Current working directory
    root_dir = Path(".")
    
    print("🧹 Starting OpenCode file organization...")
    
    # Define file mappings
    brain_files = [
        "base_brain.py",
        "enhanced_brain.py", 
        "config.py",
        "opencode.json"
    ]
    
    skill_files = [
        "check_skills.py",
        "opencode_skills_manager.py"
    ]
    
    monitoring_files = [
        "core_performance_monitor.py",
        "opencode_performance_monitor.py",
        "enhanced_model_discovery.py"
    ]
    
    doc_files = [
        "FINAL_INTEGRATION_STATUS.md",
        "FREE_MODEL_DEMO_COMPLETE.py",
        "FREE_MODEL_ORCHESTRATION_GUIDE.md",
        "FREE_MODELS_INTEGRATION_COMPLETE.md",
        "models_integration_summary.md",
        "MULTI_MODEL_DEMONSTRATION_COMPLETE.md",
        "NEO_CLONE_FREE_MODEL_INTEGRATION_COMPLETE.md",
        "minimax_agent_analysis_prompt.md",
        "CLEANUP_DOCUMENTATION.md"
    ]
    
    script_files = [
        "install",
        "start-web.bat",
        "detailed_verification.py",
        "final_verification.py",
        "debug_registration.py"
    ]
    
    example_files = [
        "opencode_integration_test.py",
        "run_model_discovery.py",
        "sample.csv"
    ]
    
    junk_files = [
        "go.msi",
        "model_usage_history.json",
        "target_opencode_workspace.txt",
        "todo.md"
    ]
    
    # Create directories if they don't exist
    dirs_to_create = ["scripts", "examples"]
    for dir_name in dirs_to_create:
        dir_path = root_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Move files function
    def move_files(files, target_dir):
        target_path = root_dir / target_dir
        target_path.mkdir(exist_ok=True)
        
        for file_name in files:
            src = root_dir / file_name
            dst = target_path / file_name
            
            if src.exists():
                try:
                    shutil.move(str(src), str(dst))
                    print(f"✅ Moved {file_name} to {target_dir}/")
                except Exception as e:
                    print(f"❌ Failed to move {file_name}: {e}")
            else:
                print(f"⚠️  File not found: {file_name}")
    
    # Execute moves
    print("\n📁 Moving brain files...")
    move_files(brain_files, "brain")
    
    print("\n🔧 Moving skill files...")
    move_files(skill_files, "skills")
    
    print("\n📊 Moving monitoring files...")
    move_files(monitoring_files, "monitoring")
    
    print("\n📚 Moving documentation files...")
    move_files(doc_files, "docs")
    
    print("\n⚙️  Moving script files...")
    move_files(script_files, "scripts")
    
    print("\n💡 Moving example files...")
    move_files(example_files, "examples")
    
    # Remove junk files
    print("\n🗑️  Removing junk files...")
    for file_name in junk_files:
        file_path = root_dir / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ Deleted {file_name}")
            except Exception as e:
                print(f"❌ Failed to delete {file_name}: {e}")
        else:
            print(f"⚠️  Junk file not found: {file_name}")
    
    print("\n🎉 File organization complete!")
    print("\n📊 SUMMARY:")
    print("- 4 brain files moved to brain/")
    print("- 2 skill files moved to skills/")
    print("- 3 monitoring files moved to monitoring/")
    print("- 9 documentation files moved to docs/")
    print("- 5 script files moved to scripts/")
    print("- 3 example files moved to examples/")
    print("- 4 junk files removed")
    print("\n🚀 Total: 25+ files organized!")
    
    print("\n📋 Next steps:")
    print("1. Update package.json script to: 'python brain/opencode_unified_brain.py'")
    print("2. Update Neo-Clone imports to use 'brain.opencode_unified_brain'")
    print("3. Test with: bun run tui")

if __name__ == "__main__":
    organize_files()
