#!/usr/bin/env python3
"""
Neo-Clone Directory Cleanup Script

Identifies and removes files that are safe to delete to reduce clutter.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set


class NeoCloneCleaner:
    """Cleans up the neo-clone directory by removing unnecessary files"""

    def __init__(self, neo_clone_path: str = "."):
        self.base_path = Path(neo_clone_path)
        self.keep_files = self._get_keep_files()
        self.keep_patterns = self._get_keep_patterns()

    def _get_keep_files(self) -> Set[str]:
        """Files that should definitely be kept"""
        return {
            # Core system files
            "__init__.py",
            "main.py",
            "config.py",
            "data_models.py",
            "minimax_agent.py",

            # Brain system
            "brain/base_brain.py",
            "brain/enhanced_brain.py",
            "brain/brain.py",
            "brain/opencode_unified_brain.py",
            "brain/unified_memory.py",
            "brain/persistent_memory.py",
            "brain/vector_memory.py",
            "brain/data_models.py",

            # Skills (already moved to skills/ directory)
            "code_generation.py",
            "text_analysis.py",
            "data_inspector.py",
            "web_search.py",
            "file_manager.py",
            "additional_skills.py",
            "more_skills.py",

            # Essential utilities
            "enhanced_llm_client.py",
            "logging_system.py",
            "cache_system.py",
            "memory.py",
            "plugin_system.py",

            # Configuration and requirements
            "requirements.txt",
            "config_opencode.py",

            # Documentation
            "README.md",
        }

    def _get_keep_patterns(self) -> List[str]:
        """Patterns for files to keep"""
        return [
            # Core directories
            "brain/",
            "data/",
            "monitoring/",
            "scripts/",

            # Important config files
            "opencode.json",

            # Keep some recent integration files
            "PHASE5_*",
            "MULTISESSION_*",
            "WEBSITE_AUTOMATION_*",
        ]

    def _should_delete_file(self, file_path: Path) -> bool:
        """Determine if a file should be deleted"""
        file_name = file_path.name

        # Keep essential files
        if file_name in self.keep_files:
            return False

        # Keep files matching patterns
        for pattern in self.keep_patterns:
            if pattern in str(file_path):
                return False

        # Delete test files
        if (file_name.startswith("test_") or
            file_name.endswith("_test.py") or
            "test" in file_name.lower() or
            "demo" in file_name.lower()):
            return True

        # Delete temporary and debug files
        if (file_name.endswith(".pyc") or
            file_name.startswith("temp_") or
            file_name.startswith("debug_") or
            file_name.endswith("_debug.py") or
            "snapshot" in file_name or
            "diagnostic" in file_name):
            return True

        # Delete old integration files
        if ("integration" in file_name.lower() and
            not file_name.startswith("enhanced_") and
            not file_name.startswith("opencode_")):
            return True

        # Delete duplicate files (keep the one with underscore)
        if file_name.replace("_", "") == file_name.replace("_", "") and "_" not in file_name:
            # Check if there's a version with underscore
            underscore_version = file_path.parent / file_name.replace("tool", "_tool")
            if underscore_version.exists():
                return True

        # Delete old evolution and phase files (keep only recent ones)
        if ("evolution" in file_name.lower() or
            "phase" in file_name.lower()) and not (
            "PHASE5" in file_name or
            "MULTISESSION" in file_name or
            "WEBSITE_AUTOMATION" in file_name
        ):
            return True

        # Delete weather and google specific demos (not core functionality)
        if ("weather" in file_name.lower() or
            "google" in file_name.lower()) and (
            "demo" in file_name.lower() or
            "test" in file_name.lower()
        ):
            return True

        # Delete old MCP files (keep only essential ones)
        if ("mcp" in file_name.lower() and
            not file_name.startswith("enhanced_") and
            not file_name == "mcp_protocol.py"):
            return True

        return False

    def analyze_directory(self) -> dict:
        """Analyze the directory and identify files to delete"""
        to_delete = []
        to_keep = []

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                if self._should_delete_file(file_path):
                    to_delete.append(file_path)
                else:
                    to_keep.append(file_path)

        return {
            "to_delete": to_delete,
            "to_keep": to_keep,
            "total_files": len(to_delete) + len(to_keep),
            "delete_count": len(to_delete),
            "keep_count": len(to_keep)
        }

    def cleanup_directory(self, dry_run: bool = True) -> dict:
        """Clean up the directory by removing unnecessary files"""
        analysis = self.analyze_directory()

        if dry_run:
            print("DRY RUN - No files will be deleted")
            print(f"Would delete {analysis['delete_count']} files")
            print(f"Would keep {analysis['keep_count']} files")
            print("\nFiles to be deleted:")
            for file_path in analysis['to_delete'][:20]:  # Show first 20
                print(f"  - {file_path}")
            if analysis['delete_count'] > 20:
                print(f"  ... and {analysis['delete_count'] - 20} more")
        else:
            print(f"Deleting {analysis['delete_count']} files...")
            deleted_count = 0
            for file_path in analysis['to_delete']:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

            print(f"Successfully deleted {deleted_count} files")

        return analysis

    def show_file_types(self) -> dict:
        """Show breakdown of file types in the directory"""
        file_types = {}

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext not in file_types:
                    file_types[ext] = {"count": 0, "to_delete": 0}

                file_types[ext]["count"] += 1
                if self._should_delete_file(file_path):
                    file_types[ext]["to_delete"] += 1

        return file_types


def main():
    """Main cleanup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up Neo-Clone directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--analyze", action="store_true", help="Analyze directory without cleanup")
    parser.add_argument("--types", action="store_true", help="Show file type breakdown")
    parser.add_argument("--force", action="store_true", help="Actually delete files (dangerous!)")

    args = parser.parse_args()

    cleaner = NeoCloneCleaner()

    if args.types:
        print("File Type Analysis:")
        print("=" * 50)
        types = cleaner.show_file_types()
        for ext, info in sorted(types.items()):
            delete_pct = (info["to_delete"] / info["count"] * 100) if info["count"] > 0 else 0
            print("15")
        return

    if args.analyze:
        print("Directory Analysis:")
        print("=" * 50)
        analysis = cleaner.analyze_directory()
        print(f"Total files: {analysis['total_files']}")
        print(f"Files to keep: {analysis['keep_count']}")
        print(f"Files to delete: {analysis['delete_count']}")
        print(".1f")
        return

    # Default behavior: dry run
    dry_run = not args.force
    cleaner.cleanup_directory(dry_run=dry_run)


if __name__ == "__main__":
    main()
