#!/usr/bin/env python3
"""
Activate Neo-Clone Autonomous Evolution Engine
Starts continuous self-improvement and monitoring
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add neo-clone to path
neo_clone_path = Path(__file__).parent
sys.path.insert(0, str(neo_clone_path))

try:
    from autonomous_evolution_engine import AutonomousEvolutionEngine
    from monitoring.metrics_collector import MetricsCollector
except ImportError as e:
    print(f"Import error: {e}")
    print("Evolution engine components not available")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evolution.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main activation function"""
    logger.info("🧬 Activating Neo-Clone Autonomous Evolution Engine")
    
    try:
        # Initialize evolution engine
        evolution_engine = AutonomousEvolutionEngine()
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        
        # Get current directory for scanning
        current_dir = Path.cwd()
        neo_clone_dir = current_dir / "neo-clone"
        
        if not neo_clone_dir.exists():
            neo_clone_dir = current_dir
        
        logger.info(f"📁 Scanning directory: {neo_clone_dir}")
        
        # Perform initial scan
        logger.info("🔍 Performing initial opportunity scan...")
        opportunities = evolution_engine.scanner.scan_codebase(
            str(neo_clone_dir), 
            include_internet=True
        )
        
        logger.info(f"✅ Found {len(opportunities)} improvement opportunities")
        
        # Display top opportunities
        if opportunities:
            logger.info("🎯 Top 5 opportunities:")
            for i, opp in enumerate(opportunities[:5], 1):
                logger.info(f"  {i}. [{opp.priority.upper()}] {opp.title}")
                logger.info(f"     Impact: {opp.impact_score:.2f} | {opp.description[:100]}...")
        
        # Start autonomous mode
        logger.info("🚀 Starting autonomous evolution mode...")
        evolution_engine.start_autonomous_mode()
        
        # Keep main thread alive
        logger.info("✅ Evolution engine is running autonomously")
        logger.info("📊 Monitoring system performance and implementing improvements")
        logger.info("🔄 Press Ctrl+C to stop (engine will continue in background)")
        
        try:
            while True:
                time.sleep(60)  # Check every minute
                
                # Log status
                if evolution_engine.metrics:
                    logger.info(f"📈 Status: {evolution_engine.metrics.opportunities_implemented} improvements made")
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping evolution engine...")
            evolution_engine.is_running = False
            logger.info("✅ Evolution engine stopped gracefully")
            
    except Exception as e:
        logger.error(f"❌ Failed to activate evolution engine: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()