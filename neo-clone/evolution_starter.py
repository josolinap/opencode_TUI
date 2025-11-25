"""
Simple Evolution Engine Starter
Robust activation with minimal dependencies
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimpleEvolutionEngine:
    """Simplified evolution engine that works with minimal dependencies"""
    
    def __init__(self):
        self.is_running = False
        self.scan_interval = 300  # 5 minutes
        self.improvements_made = 0
        self.last_scan_time = None
        
    def start(self):
        """Start the evolution engine"""
        if self.is_running:
            logger.warning("Evolution engine already running")
            return
            
        self.is_running = True
        logger.info("🧬 Starting Simple Evolution Engine")
        
        # Start background scanning thread
        scan_thread = threading.Thread(target=self._continuous_scanning, daemon=True)
        scan_thread.start()
        
        logger.info("✅ Evolution engine started successfully")
        
    def _continuous_scanning(self):
        """Continuous background scanning"""
        while self.is_running:
            try:
                self._perform_scan()
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Scan error: {e}")
                time.sleep(60)  # Wait 1 minute on error
                
    def _perform_scan(self):
        """Perform improvement scan"""
        start_time = time.time()
        logger.info("🔍 Scanning for improvement opportunities...")
        
        opportunities_found = 0
        
        # Scan for common issues
        current_dir = Path.cwd()
        
        # Check for Python files with potential improvements
        for py_file in current_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple improvement detection
                if "TODO" in content or "FIXME" in content:
                    opportunities_found += 1
                    logger.info(f"📝 Found TODO/FIXME in: {py_file.relative_to(current_dir)}")
                    
                if len(content) > 10000 and "def " in content:
                    opportunities_found += 1
                    logger.info(f"📏 Long file detected: {py_file.relative_to(current_dir)}")
                    
            except Exception as e:
                logger.debug(f"Could not scan {py_file}: {e}")
        
        # Check for performance opportunities
        if opportunities_found > 0:
            logger.info(f"🎯 Found {opportunities_found} improvement opportunities")
            self.improvements_made += 1
        else:
            logger.info("✅ No immediate improvements needed")
            
        scan_duration = time.time() - start_time
        self.last_scan_time = time.time()
        
        logger.info(f"📊 Scan completed in {scan_duration:.2f}s")
        
    def get_status(self):
        """Get current status"""
        return {
            'running': self.is_running,
            'improvements_made': self.improvements_made,
            'last_scan': self.last_scan_time,
            'scan_interval': self.scan_interval
        }

def main():
    """Main function"""
    logger.info("🚀 Neo-Clone Evolution Engine Starter")
    
    try:
        # Create and start evolution engine
        engine = SimpleEvolutionEngine()
        engine.start()
        
        # Keep main thread alive and report status
        logger.info("🔄 Evolution engine is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(30)  # Report status every 30 seconds
                status = engine.get_status()
                if status['last_scan']:
                    logger.info(f"📈 Status: {status['improvements_made']} improvements made")
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping evolution engine...")
            engine.is_running = False
            logger.info("✅ Evolution engine stopped")
            
    except Exception as e:
        logger.error(f"❌ Failed to start evolution engine: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())