#!/usr/bin/env python3
"""
Neo-Clone Website Automation Dependencies Installer (Simplified)
"""

import subprocess
import sys
import os
from typing import List, Dict, Any

class SimpleDependencyInstaller:
    """Simplified dependency installation for Windows."""
    
    def __init__(self):
        self.dependencies = [
            "playwright",
            "playwright-stealth", 
            "seleniumbase",
            "beautifulsoup4",
            "lxml",
            "requests",
            "cryptography",
            "pyotp",
            "2captcha-python",
            "pillow",
            "opencv-python",
            "pytesseract",
            "numpy",
            "scikit-learn",
            "fake-useragent",
            "python-dotenv",
            "aiohttp",
            "aiofiles"
        ]
        
        self.installation_log = []
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and log the result."""
        print(f"[{description}]...")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"[SUCCESS] {description}")
                self.installation_log.append({
                    "command": command,
                    "description": description,
                    "status": "SUCCESS"
                })
                return True
            else:
                print(f"[FAILED] {description}")
                print(f"Error: {result.stderr}")
                self.installation_log.append({
                    "command": command,
                    "description": description,
                    "status": "FAILED",
                    "error": result.stderr
                })
                return False
                
        except Exception as e:
            print(f"[ERROR] {description}: {str(e)}")
            self.installation_log.append({
                "command": command,
                "description": description,
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    def install_dependencies(self) -> Dict[str, bool]:
        """Install all dependencies."""
        print("Installing Python dependencies...")
        print("=" * 50)
        
        # First upgrade pip
        self.run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        )
        
        results = {}
        
        for package in self.dependencies:
            success = self.run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package}"
            )
            results[package] = success
        
        # Install playwright browsers
        self.run_command(
            f"{sys.executable} -m playwright install",
            "Installing Playwright browsers"
        )
        
        return results
    
    def verify_installations(self) -> Dict[str, bool]:
        """Verify key installations."""
        print("Verifying installations...")
        print("=" * 50)
        
        key_packages = ["playwright", "seleniumbase", "requests", "beautifulsoup4"]
        verification_results = {}
        
        for package in key_packages:
            try:
                __import__(package)
                print(f"[VERIFIED] {package}")
                verification_results[package] = True
            except ImportError:
                print(f"[NOT FOUND] {package}")
                verification_results[package] = False
        
        return verification_results
    
    def install_all(self):
        """Run complete installation."""
        print("Neo-Clone Website Automation Dependencies Installer")
        print("=" * 60)
        
        # Install dependencies
        results = self.install_dependencies()
        
        # Verify installations
        verification = self.verify_installations()
        
        # Summary
        successful = len([r for r in results.values() if r])
        total = len(results)
        
        print("\nInstallation Summary")
        print("=" * 50)
        print(f"Total packages: {total}")
        print(f"Successful: {successful}")
        print(f"Success rate: {(successful/total)*100:.1f}%")
        
        if successful >= total * 0.8:
            print("\n[SUCCESS] Installation completed!")
            print("You can now use the website automation system.")
        else:
            print("\n[WARNING] Installation had issues.")
            print("Some features may not work correctly.")
        
        return successful >= total * 0.8


def main():
    installer = SimpleDependencyInstaller()
    try:
        return installer.install_all()
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        return False
    except Exception as e:
        print(f"\nInstallation failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)