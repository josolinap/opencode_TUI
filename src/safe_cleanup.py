#!/usr/bin/env python3
"""
Safe Workspace Cleanup - Phase 1
Remove obvious redundancies and organize structure
"""
import os
import shutil
from pathlib import Path

def safe_cleanup_phase1():
    """Phase 1: Remove obvious redundancies"""
    print("Starting Safe Cleanup Phase 1")
    print("="*50)
    
    root = Path(".")
    
    # 1. Remove backup files (keep last 5 most recent)
    print("\n1. Cleaning backup files...")
    backup_dir = root / "backups"
    if backup_dir.exists():
        backup_files = list(backup_dir.glob("*.backup"))
        print(f"   Found {len(backup_files)} backup files")
        
        # Sort by modification time, keep last 5
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        to_delete = backup_files[5:]  # Keep last 5
        
        for backup_file in to_delete:
            try:
                backup_file.unlink()
                print(f"   Deleted: {backup_file.name}")
            except Exception as e:
                print(f"   Error deleting {backup_file.name}: {e}")
        
        print(f"   Kept 5 most recent, deleted {len(to_delete)} backup files")
    
    # 2. Remove empty directories
    print("\n2. Removing empty directories...")
    empty_dirs = []
    for root_dir, dirs, files in os.walk("."):
        # Skip node_modules and .git
        if ".git" in root_dir or "node_modules" in root_dir:
            continue
            
        for dir_name in dirs:
            dir_path = Path(root_dir) / dir_name
            try:
                if not any(dir_path.iterdir()):
                    empty_dirs.append(dir_path)
            except PermissionError:
                pass
    
    for empty_dir in empty_dirs:
        try:
            empty_dir.rmdir()
            print(f"   Removed empty dir: {empty_dir}")
        except Exception as e:
            print(f"   Error removing {empty_dir}: {e}")
    
    print(f"   Removed {len(empty_dirs)} empty directories")
    
    # 3. Consolidate obvious duplicates
    print("\n3. Consolidating obvious duplicates...")
    
    # Move script/ contents to scripts/ if scripts/ exists
    script_dir = root / "script"
    scripts_dir = root / "scripts"
    
    if script_dir.exists() and scripts_dir.exists():
        print("   Consolidating script/ into scripts/")
        for item in script_dir.iterdir():
            dest = scripts_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
                print(f"   Moved: {item.name} -> scripts/")
        
        # Remove empty script/ dir
        try:
            script_dir.rmdir()
            print("   Removed empty script/ directory")
        except:
            print("   Could not remove script/ directory (not empty)")
    
    # 4. Create basic organization
    print("\n4. Creating basic organization...")
    
    # Create src/ directory if it doesn't exist
    src_dir = root / "src"
    if not src_dir.exists():
        src_dir.mkdir()
        print("   Created src/ directory")
    
    # Move core Python files to src/
    python_files = [
        "check_config.py", "check_current_models.py", "check_models_simple.py",
        "connect_neo_clone_to_opencode.py", "debug_skills.py", 
        "fix_neo_clone_opencode.py", "neo-clone_demo_working.py",
        "neo-clone_integration_fixed.py", "neo-clone_opencode_integration.py",
        "neo-clone_working_integration.py", "simple_brain_check.py",
        "simple_neo_clone_connection.py", "simple_skills_test.py",
        "test_model_execution.py", "test_openai_direct.py",
        "test_skills_manager.py", "workspace_cleanup.py", "safe_cleanup.py"
    ]
    
    moved_count = 0
    for py_file in python_files:
        src_file = root / py_file
        if src_file.exists():
            dest_file = src_dir / py_file
            if not dest_file.exists():
                shutil.move(str(src_file), str(dest_file))
                print(f"   Moved: {py_file} -> src/")
                moved_count += 1
    
    print(f"   Moved {moved_count} Python files to src/")
    
    print("\nPhase 1 Cleanup Complete!")
    print("\nSummary:")
    print(f"- Cleaned backup files (kept 5 most recent)")
    print(f"- Removed {len(empty_dirs)} empty directories")
    print(f"- Consolidated script/ directories")
    print(f"- Moved {moved_count} Python files to src/")
    print(f"- Created organized src/ structure")

if __name__ == "__main__":
    safe_cleanup_phase1()