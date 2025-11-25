"""
Security and anonymization features for Neo-OSINT
"""

import asyncio
import logging
import time
import random
import socket
from typing import Dict, Optional, Any
from stem import Signal
from stem.control import Controller

from ..core.config import NeoOSINTConfig


class Anonymizer:
    """Tor-based anonymization and identity management"""
    
    def __init__(self, config: NeoOSINTConfig):
        self.config = config
        self.logger = logging.getLogger("neo_osint.security")
        
        # Tor connection settings
        self.control_port = config.security.tor_control_port
        self.control_password = config.security.tor_control_password
        self.rotation_interval = config.security.rotation_interval
        
        # Identity rotation tracking
        self.last_rotation = 0
        self.request_count = 0
        
        # Initialize Tor controller if available
        self.controller = None
        self._initialize_tor_controller()
    
    def _initialize_tor_controller(self) -> None:
        """Initialize Tor controller for identity rotation"""
        if not self.config.security.use_tor:
            return
        
        try:
            self.controller = Controller.from_port(port=self.control_port)
            if self.control_password:
                self.controller.authenticate(password=self.control_password)
            else:
                self.controller.authenticate()
            
            self.logger.info("Tor controller initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Tor controller: {e}")
            self.controller = None
    
    async def rotate_identity(self, force: bool = False) -> bool:
        """Rotate Tor identity if needed"""
        if not self.config.security.use_tor or not self.controller:
            return False
        
        current_time = time.time()
        
        # Check if rotation is needed
        if not force and (current_time - self.last_rotation) < self.rotation_interval:
            return False
        
        try:
            # Send NEWNYM signal to Tor
            await asyncio.to_thread(self.controller.signal, Signal.NEWNYM)
            self.last_rotation = current_time
            self.request_count = 0
            
            self.logger.info("Tor identity rotated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate Tor identity: {e}")
            return False
    
    async def check_tor_connection(self) -> bool:
        """Check if Tor is working properly"""
        if not self.config.security.use_tor:
            return True
        
        try:
            # Check Tor connection via a known Tor check service
            import aiohttp
            
            proxies = {
                "http": f"socks5h://127.0.0.1:{self.config.security.tor_socks_port}",
                "https": f"socks5h://127.0.0.1:{self.config.security.tor_socks_port}"
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://check.torproject.org/",
                    proxy=proxies.get("http")
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        return "Congratulations" in content
            
            return False
            
        except Exception as e:
            self.logger.error(f"Tor connection check failed: {e}")
            return False
    
    async def get_current_ip(self) -> Optional[str]:
        """Get current IP address through Tor"""
        if not self.config.security.use_tor:
            return None
        
        try:
            import aiohttp
            
            proxies = {
                "http": f"socks5h://127.0.0.1:{self.config.security.tor_socks_port}",
                "https": f"socks5h://127.0.0.1:{self.config.security.tor_socks_port}"
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://api.ipify.org?format=json",
                    proxy=proxies.get("http")
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("ip")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get current IP: {e}")
            return None
    
    def increment_request_count(self) -> None:
        """Track request count for rotation decisions"""
        self.request_count += 1
        
        # Rotate identity after certain number of requests
        if self.config.security.rotate_identity and self.request_count >= 10:
            asyncio.create_task(self.rotate_identity(force=True))
    
    async def get_anonymized_headers(self) -> Dict[str, str]:
        """Get anonymized HTTP headers"""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
        
        # Rotate user agent if enabled
        if self.config.security.user_agent_rotation:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:137.0) Gecko/20100101 Firefox/137.0"
            ]
            headers["User-Agent"] = random.choice(user_agents)
        else:
            headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        
        return headers
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.controller:
            try:
                self.controller.close()
            except Exception as e:
                self.logger.error(f"Error closing Tor controller: {e}")