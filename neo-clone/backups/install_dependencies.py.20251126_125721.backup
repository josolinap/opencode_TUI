#!/usr/bin/env python3
"""
📦 Neo-Clone Website Automation Dependencies Installer
==================================================

Automated installation of all required dependencies for website automation.
"""

import subprocess
import sys
import os
from typing import List, Dict, Any

class DependencyInstaller:
    """Automated dependency installation with error handling."""
    
    def __init__(self):
        self.dependencies = {
            # Core browser automation
            "playwright": "pip install playwright",
            "playwright-stealth": "pip install playwright-stealth",
            "seleniumbase": "pip install seleniumbase",
            
            # HTML parsing and analysis
            "beautifulsoup4": "pip install beautifulsoup4",
            "lxml": "pip install lxml",
            "requests": "pip install requests",
            
            # Security and encryption
            "cryptography": "pip install cryptography",
            "pyotp": "pip install pyotp",
            "2captcha-python": "pip install 2captcha-python",
            
            # Image processing and OCR
            "pillow": "pip install pillow",
            "opencv-python": "pip install opencv-python",
            "pytesseract": "pip install pytesseract",
            
            # Machine learning
            "numpy": "pip install numpy",
            "scikit-learn": "pip install scikit-learn",
            
            # Anti-detection
            "fake-useragent": "pip install fake-useragent",
            
            # Environment management
            "python-dotenv": "pip install python-dotenv",
            
            # Additional utilities
            "aiohttp": "pip install aiohttp",
            "aiofiles": "pip install aiofiles",
        }
        
        self.system_dependencies = {
            "windows": [],
            "linux": ["tesseract-ocr"],
            "darwin": ["tesseract"]
        }
        
        self.installation_log = []
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and log the result."""
        print(f"🔄 {description}...")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                self.installation_log.append({
                    "command": command,
                    "description": description,
                    "status": "SUCCESS",
                    "output": result.stdout
                })
                return True
            else:
                print(f"❌ {description} - FAILED")
                print(f"   Error: {result.stderr}")
                self.installation_log.append({
                    "command": command,
                    "description": description,
                    "status": "FAILED",
                    "error": result.stderr
                })
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT")
            self.installation_log.append({
                "command": command,
                "description": description,
                "status": "TIMEOUT"
            })
            return False
        except Exception as e:
            print(f"💥 {description} - ERROR: {str(e)}")
            self.installation_log.append({
                "command": command,
                "description": description,
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version."""
        return self.run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        )
    
    def install_python_dependencies(self) -> Dict[str, bool]:
        """Install all Python dependencies."""
        print("🐍 Installing Python dependencies...")
        print("=" * 60)
        
        results = {}
        
        for package, install_command in self.dependencies.items():
            success = self.run_command(
                f"{sys.executable} -m {install_command}",
                f"Installing {package}"
            )
            results[package] = success
        
        return results
    
    def install_system_dependencies(self) -> Dict[str, bool]:
        """Install system dependencies."""
        print("🖥️ Installing system dependencies...")
        print("=" * 60)
        
        # Detect OS
        if sys.platform.startswith('win'):
            os_name = "windows"
            package_manager = "choco install"
        elif sys.platform.startswith('linux'):
            os_name = "linux"
            package_manager = "sudo apt-get install"
        elif sys.platform.startswith('darwin'):
            os_name = "darwin"
            package_manager = "brew install"
        else:
            print(f"⚠️ Unknown OS: {sys.platform}")
            return {}
        
        results = {}
        
        for dep in self.system_dependencies.get(os_name, []):
            success = self.run_command(
                f"{package_manager} {dep}",
                f"Installing {dep}"
            )
            results[dep] = success
        
        return results
    
    def install_playwright_browsers(self) -> bool:
        """Install Playwright browsers."""
        return self.run_command(
            f"{sys.executable} -m playwright install",
            "Installing Playwright browsers"
        )
    
    def verify_installations(self) -> Dict[str, bool]:
        """Verify that all dependencies are installed."""
        print("🔍 Verifying installations...")
        print("=" * 60)
        
        verification_results = {}
        
        for package in self.dependencies.keys():
            try:
                if package == "playwright-stealth":
                    import playwright_stealth
                elif package == "seleniumbase":
                    import seleniumbase
                elif package == "beautifulsoup4":
                    import bs4
                elif package == "lxml":
                    import lxml
                elif package == "cryptography":
                    from cryptography.fernet import Fernet
                elif package == "pyotp":
                    import pyotp
                elif package == "2captcha-python":
                    import twocaptcha
                elif package == "pillow":
                    from PIL import Image
                elif package == "opencv-python":
                    import cv2
                elif package == "pytesseract":
                    import pytesseract
                elif package == "numpy":
                    import numpy
                elif package == "scikit-learn":
                    import sklearn
                elif package == "fake-useragent":
                    import fake_useragent
                elif package == "python-dotenv":
                    import dotenv
                elif package == "requests":
                    import requests
                elif package == "aiohttp":
                    import aiohttp
                elif package == "aiofiles":
                    import aiofiles
                else:
                    __import__(package)
                
                print(f"✅ {package} - VERIFIED")
                verification_results[package] = True
                
            except ImportError as e:
                print(f"❌ {package} - NOT FOUND: {str(e)}")
                verification_results[package] = False
        
        return verification_results
    
    def generate_installation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive installation report."""
        successful_installs = len([log for log in self.installation_log if log["status"] == "SUCCESS"])
        failed_installs = len([log for log in self.installation_log if log["status"] in ["FAILED", "ERROR", "TIMEOUT"]])
        
        return {
            "total_packages": len(self.dependencies),
            "successful_installs": successful_installs,
            "failed_installs": failed_installs,
            "success_rate": (successful_installs / len(self.dependencies)) * 100 if self.dependencies else 0,
            "installation_log": self.installation_log,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on installation results."""
        recommendations = []
        
        failed_packages = [log["command"] for log in self.installation_log if log["status"] in ["FAILED", "ERROR"]]
        
        if failed_packages:
            recommendations.append("Some packages failed to install. Try installing them manually.")
            recommendations.append("Make sure you have Python 3.8+ installed.")
            recommendations.append("Check your internet connection and try again.")
        
        if sys.platform.startswith('win'):
            recommendations.append("On Windows, you may need to run Command Prompt as Administrator.")
        elif sys.platform.startswith('linux'):
            recommendations.append("On Linux, you may need to use 'sudo' for system packages.")
        elif sys.platform.startswith('darwin'):
            recommendations.append("On macOS, make sure you have Homebrew installed.")
        
        recommendations.append("After installation, restart your terminal/IDE to ensure packages are available.")
        
        return recommendations
    
    def install_all(self) -> Dict[str, Any]:
        """Run complete installation process."""
        print("🚀 Neo-Clone Website Automation Dependencies Installer")
        print("=" * 60)
        
        # Step 1: Upgrade pip
        print("\n📦 Step 1: Upgrading pip...")
        pip_success = self.upgrade_pip()
        
        # Step 2: Install Python dependencies
        print("\n🐍 Step 2: Installing Python dependencies...")
        python_results = self.install_python_dependencies()
        
        # Step 3: Install system dependencies
        print("\n🖥️ Step 3: Installing system dependencies...")
        system_results = self.install_system_dependencies()
        
        # Step 4: Install Playwright browsers
        print("\n🌐 Step 4: Installing Playwright browsers...")
        browsers_success = self.install_playwright_browsers()
        
        # Step 5: Verify installations
        print("\n🔍 Step 5: Verifying installations...")
        verification_results = self.verify_installations()
        
        # Step 6: Generate report
        report = self.generate_installation_report()
        
        # Display summary
        print("\n📊 Installation Summary")
        print("=" * 60)
        print(f"Total packages: {report['total_packages']}")
        print(f"Successful installs: {report['successful_installs']}")
        print(f"Failed installs: {report['failed_installs']}")
        print(f"Success rate: {report['success_rate']:.1f}%")
        
        if report['recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        # Save report
        report_file = "installation_report.json"
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report


def main():
    """Main installation function."""
    installer = DependencyInstaller()
    
    try:
        report = installer.install_all()
        
        if report['success_rate'] >= 80:
            print("\n🎉 Installation completed successfully!")
            print("You can now run the advanced website automation demo.")
        else:
            print("\n⚠️ Installation completed with some issues.")
            print("Please check the recommendations above.")
        
        return report['success_rate'] >= 80
        
    except KeyboardInterrupt:
        print("\n⏹️ Installation cancelled by user.")
        return False
    except Exception as e:
        print(f"\n💥 Installation failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)