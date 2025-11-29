#!/usr/bin/env python3
"""
Workspace Cleanup and Organization Tool
Analyzes current structure and provides safe cleanup operations
"""
import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set

class WorkspaceAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root = Path(root_path).resolve()
        self.analysis = {
            "duplicates": {},
            "empty_dirs": [],
            "root_clutter": [],
            "import_issues": [],
            "size_analysis": {}
        }
        
    def analyze_workspace(self) -> Dict:
        """Complete workspace analysis"""
        print("Analyzing workspace structure...")
        
        # 1. Find duplicates
        self.find_duplicates()
        
        # 2. Find empty directories
        self.find_empty_directories()
        
        # 3. Analyze root clutter
        self.analyze_root_clutter()
        
        # 4. Find import issues
        self.find_import_issues()
        
        # 5. Size analysis
        self.analyze_sizes()
        
        return self.analysis
    
    def find_duplicates(self):
        """Find duplicate files and directories"""
        print("Finding duplicates...")
        
        # Known duplicates
        duplicate_groups = {
            "skills": ["skills/", "neo-clone/skills/", "neo-clone/autonomous_skills/"],
            "scripts": ["script/", "scripts/"],
            "local_llm": ["local_llm_setup/", "neo-clone/local_llm_setup/"],
            "llm_docs": ["llm_integrations/", "neo-clone/llm_integrations/"],
            "monitoring": ["monitoring/", "neo-clone/monitoring/"],
            "backups": ["backups/", "neo-clone/backups/"]
        }
        
        for name, paths in duplicate_groups.items():
            existing = [p for p in paths if (self.root / p).exists()]
            if len(existing) > 1:
                self.analysis["duplicates"][name] = existing
    
    def find_empty_directories(self):
        """Find empty directories"""
        print("Finding empty directories...")
        
        for root, dirs, files in os.walk(self.root):
            # Skip .git and node_modules
            if ".git" in root or "node_modules" in root:
                continue
                
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        self.analysis["empty_dirs"].append(str(dir_path.relative_to(self.root)))
                except PermissionError:
                    pass
    
    def analyze_root_clutter(self):
        """Analyze files in root directory"""
        print("Analyzing root directory...")
        
        root_files = []
        for item in self.root.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                root_files.append(item.name)
        
        # Categorize root files
        categories = {
            "config": ["package.json", "tsconfig.json", "bunfig.toml", "opencode.json", ".gitignore"],
            "python": ["*.py", "requirements.txt", "pyproject.toml", "pytest.ini"],
            "docs": ["README.md", "*.md"],
            "build": ["Makefile", "build", "dist"],
            "scripts": ["*.sh", "*.bat", "*.cmd"],
            "temp": ["*.tmp", "*.log", "*.bak"]
        }
        
        self.analysis["root_clutter"] = {
            "total_files": len(root_files),
            "files": root_files,
            "categories": {}
        }
    
    def find_import_issues(self):
        """Find Python files with import issues"""
        print("Finding import issues...")
        
        import_issues = []
        
        for py_file in self.root.rglob("*.py"):
            if "node_modules" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for problematic imports
                problematic_imports = [
                    "from skills.",
                    "import skills.",
                    "from neo-clone.",
                    "import neo-clone.",
                    "from monitoring.",
                    "import monitoring."
                ]
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for problematic in problematic_imports:
                        if problematic in line:
                            import_issues.append({
                                "file": str(py_file.relative_to(self.root)),
                                "line": line_num,
                                "content": line.strip(),
                                "issue": f"Problematic import: {problematic}"
                            })
            except Exception as e:
                pass
        
        self.analysis["import_issues"] = import_issues
    
    def analyze_sizes(self):
        """Analyze directory sizes"""
        print("Analyzing directory sizes...")
        
        sizes = {}
        for item in self.root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    sizes[item.name] = {
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "file_count": len(list(item.rglob('*')))
                    }
                except Exception as e:
                    sizes[item.name] = {"error": str(e)}
        
        # Sort by size
        sorted_sizes = dict(sorted(sizes.items(), key=lambda x: x[1].get("size_mb", 0), reverse=True))
        self.analysis["size_analysis"] = sorted_sizes
    
    def generate_cleanup_plan(self) -> Dict:
        """Generate cleanup recommendations"""
        plan = {
            "immediate_actions": [],
            "reorganization": {},
            "consolidation": {},
            "deletions": []
        }
        
        # Immediate actions
        if self.analysis["duplicates"]:
            plan["immediate_actions"].append("Consolidate duplicate directories")
        
        if len(self.analysis["root_clutter"]["files"]) > 20:
            plan["immediate_actions"].append("Clean up root directory clutter")
        
        if self.analysis["empty_dirs"]:
            plan["immediate_actions"].append(f"Remove {len(self.analysis['empty_dirs'])} empty directories")
        
        # Consolidation plan
        for dup_name, paths in self.analysis["duplicates"].items():
            # Choose primary location (prefer shortest path)
            primary = min(paths, key=len)
            others = [p for p in paths if p != primary]
            plan["consolidation"][dup_name] = {
                "keep": primary,
                "merge_from": others
            }
        
        # Deletions
        if "backups" in self.analysis["duplicates"]:
            plan["deletions"].append("backups/ directory (94+ backup files)")
        
        return plan
    
    def print_analysis(self):
        """Print formatted analysis"""
        print("\n" + "="*80)
        print("WORKSPACE ANALYSIS REPORT")
        print("="*80)
        
        # Duplicates
        if self.analysis["duplicates"]:
            print(f"\nDUPLICATES FOUND:")
            for name, paths in self.analysis["duplicates"].items():
                print(f"  - {name}: {', '.join(paths)}")
        
        # Root clutter
        print(f"\nROOT DIRECTORY: {self.analysis['root_clutter']['total_files']} files")
        if self.analysis['root_clutter']['total_files'] > 20:
            print("  WARNING: Too many files in root - needs cleanup")
        
        # Empty dirs
        if self.analysis["empty_dirs"]:
            print(f"\nEMPTY DIRECTORIES: {len(self.analysis['empty_dirs'])}")
            for empty_dir in self.analysis["empty_dirs"][:5]:  # Show first 5
                print(f"  - {empty_dir}")
            if len(self.analysis["empty_dirs"]) > 5:
                print(f"  ... and {len(self.analysis['empty_dirs']) - 5} more")
        
        # Import issues
        if self.analysis["import_issues"]:
            print(f"\nIMPORT ISSUES: {len(self.analysis['import_issues'])}")
            for issue in self.analysis["import_issues"][:3]:  # Show first 3
                print(f"  - {issue['file']}:{issue['line']} - {issue['issue']}")
            if len(self.analysis["import_issues"]) > 3:
                print(f"  ... and {len(self.analysis['import_issues']) - 3} more")
        
        # Size analysis (top 5)
        print(f"\nLARGEST DIRECTORIES:")
        sizes = self.analysis["size_analysis"]
        for i, (name, info) in enumerate(list(sizes.items())[:5]):
            if "error" not in info:
                print(f"  {i+1}. {name}/ - {info['size_mb']} MB ({info['file_count']} files)")
        
        print("\n" + "="*80)

def main():
    analyzer = WorkspaceAnalyzer()
    analysis = analyzer.analyze_workspace()
    analyzer.print_analysis()
    
    # Generate and print cleanup plan
    plan = analyzer.generate_cleanup_plan()
    
    print("\nCLEANUP PLAN:")
    print("="*40)
    
    if plan["immediate_actions"]:
        print("\nIMMEDIATE ACTIONS:")
        for action in plan["immediate_actions"]:
            print(f"  - {action}")
    
    if plan["consolidation"]:
        print("\nCONSOLIDATION:")
        for name, details in plan["consolidation"].items():
            print(f"  - {name}:")
            print(f"    Keep: {details['keep']}")
            print(f"    Merge from: {', '.join(details['merge_from'])}")
    
    if plan["deletions"]:
        print("\nSAFE DELETIONS:")
        for deletion in plan["deletions"]:
            print(f"  - {deletion}")

    # Save analysis to file
    with open("workspace_analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: workspace_analysis_report.json")

if __name__ == "__main__":
    main()