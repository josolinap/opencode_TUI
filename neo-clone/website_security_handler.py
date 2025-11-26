#!/usr/bin/env python3
"""
🛡️ Neo-Clone Website Security Handler
======================================

Advanced security bypass system for CAPTCHA solving, 2FA handling,
and anti-detection techniques.
"""

import asyncio
import json
import logging
import time
import base64
import os
import io
import requests
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import tempfile
import random
import string

# Image processing
try:
    from PIL import Image
    import cv2
    import numpy as np
except ImportError:
    print("Installing image processing dependencies...")
    os.system("pip install pillow opencv-python numpy")
    from PIL import Image
    import cv2
    import numpy as np

# 2Captcha API
try:
    import twocaptcha
except ImportError:
    print("Installing 2Captcha API...")
    os.system("pip install 2captcha-python")
    import twocaptcha

# TOTP for 2FA
try:
    import pyotp
except ImportError:
    print("Installing pyotp...")
    os.system("pip install pyotp")
    import pyotp

# Email handling for email verification
try:
    import imaplib
    import email
    from email.header import decode_header
except ImportError:
    print("Email libraries not available - some features may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptchaType(Enum):
    """Supported CAPTCHA types."""
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    FUNCAPTCHA = "funcaptcha"
    TURNSTILE = "turnstile"
    IMAGE_CAPTCHA = "image_captcha"
    TEXT_CAPTCHA = "text_captcha"


class TwoFactorType(Enum):
    """Supported 2FA types."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"


@dataclass
class CaptchaConfig:
    """CAPTCHA solving configuration."""
    api_key: str
    service_url: str = "https://2captcha.com"
    timeout: int = 120
    max_retries: int = 3
    polling_interval: int = 5


@dataclass
class ProxyConfig:
    """Proxy configuration for anti-detection."""
    enabled: bool = True
    rotation_enabled: bool = True
    proxy_list: List[str] = None
    rotation_interval: int = 300  # seconds
    
    def __post_init__(self):
        if self.proxy_list is None:
            self.proxy_list = []


class CaptchaSolver:
    """Advanced CAPTCHA solving system."""
    
    def __init__(self, config: CaptchaConfig):
        self.config = config
        self.solver = twocaptcha.TwoCaptcha(config.api_key)
        self.solve_history = []
        
        logger.info("CaptchaSolver initialized")
    
    async def solve_recaptcha_v2(self, sitekey: str, page_url: str) -> Optional[str]:
        """Solve reCAPTCHA v2."""
        try:
            logger.info(f"Solving reCAPTCHA v2 for sitekey: {sitekey}")
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.solver.recaptcha(
                    sitekey=sitekey,
                    url=page_url
                )
            )
            
            solution = result['code']
            
            self.solve_history.append({
                "type": CaptchaType.RECAPTCHA_V2.value,
                "sitekey": sitekey,
                "page_url": page_url,
                "success": True,
                "timestamp": time.time()
            })
            
            logger.info("reCAPTCHA v2 solved successfully")
            return solution
            
        except Exception as e:
            logger.error(f"Failed to solve reCAPTCHA v2: {str(e)}")
            
            self.solve_history.append({
                "type": CaptchaType.RECAPTCHA_V2.value,
                "sitekey": sitekey,
                "page_url": page_url,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            })
            
            return None
    
    async def solve_hcaptcha(self, sitekey: str, page_url: str) -> Optional[str]:
        """Solve hCaptcha."""
        try:
            logger.info(f"Solving hCaptcha for sitekey: {sitekey}")
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.solver.hcaptcha(
                    sitekey=sitekey,
                    url=page_url
                )
            )
            
            solution = result['code']
            
            self.solve_history.append({
                "type": CaptchaType.HCAPTCHA.value,
                "sitekey": sitekey,
                "page_url": page_url,
                "success": True,
                "timestamp": time.time()
            })
            
            logger.info("hCaptcha solved successfully")
            return solution
            
        except Exception as e:
            logger.error(f"Failed to solve hCaptcha: {str(e)}")
            return None
    
    async def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """Solve image-based CAPTCHA."""
        try:
            logger.info("Solving image CAPTCHA")
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.solver.normal(
                    file=temp_file_path
                )
            )
            
            solution = result['code']
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            self.solve_history.append({
                "type": CaptchaType.IMAGE_CAPTCHA.value,
                "success": True,
                "timestamp": time.time()
            })
            
            logger.info("Image CAPTCHA solved successfully")
            return solution
            
        except Exception as e:
            logger.error(f"Failed to solve image CAPTCHA: {str(e)}")
            return None
    
    async def solve_funcaptcha(self, sitekey: str, page_url: str) -> Optional[str]:
        """Solve FunCaptcha."""
        try:
            logger.info(f"Solving FunCaptcha for sitekey: {sitekey}")
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.solver.funcaptcha(
                    sitekey=sitekey,
                    url=page_url
                )
            )
            
            solution = result['code']
            
            self.solve_history.append({
                "type": CaptchaType.FUNCAPTCHA.value,
                "sitekey": sitekey,
                "page_url": page_url,
                "success": True,
                "timestamp": time.time()
            })
            
            logger.info("FunCaptcha solved successfully")
            return solution
            
        except Exception as e:
            logger.error(f"Failed to solve FunCaptcha: {str(e)}")
            return None
    
    def get_solve_statistics(self) -> Dict[str, Any]:
        """Get CAPTCHA solving statistics."""
        if not self.solve_history:
            return {"total_solves": 0, "success_rate": 0}
        
        total_solves = len(self.solve_history)
        successful_solves = len([s for s in self.solve_history if s['success']])
        success_rate = (successful_solves / total_solves) * 100
        
        # Group by type
        by_type = {}
        for solve in self.solve_history:
            captcha_type = solve['type']
            if captcha_type not in by_type:
                by_type[captcha_type] = {"total": 0, "successful": 0}
            by_type[captcha_type]["total"] += 1
            if solve['success']:
                by_type[captcha_type]["successful"] += 1
        
        return {
            "total_solves": total_solves,
            "successful_solves": successful_solves,
            "success_rate": success_rate,
            "by_type": by_type,
            "recent_solves": self.solve_history[-10:]  # Last 10 solves
        }


class TwoFactorHandler:
    """Two-factor authentication handler."""
    
    def __init__(self):
        self.totp_secrets = {}
        self.email_configs = {}
        self.sms_handlers = {}
        
        logger.info("TwoFactorHandler initialized")
    
    def add_totp_secret(self, service: str, secret: str):
        """Add TOTP secret for a service."""
        self.totp_secrets[service] = secret
        logger.info(f"Added TOTP secret for {service}")
    
    def generate_totp_code(self, service: str) -> Optional[str]:
        """Generate TOTP code for a service."""
        try:
            if service not in self.totp_secrets:
                logger.error(f"No TOTP secret found for {service}")
                return None
            
            totp = pyotp.TOTP(self.totp_secrets[service])
            code = totp.now()
            
            logger.info(f"Generated TOTP code for {service}")
            return code
            
        except Exception as e:
            logger.error(f"Failed to generate TOTP code: {str(e)}")
            return None
    
    def add_email_config(self, service: str, imap_server: str, username: str, password: str):
        """Add email configuration for email-based 2FA."""
        self.email_configs[service] = {
            "imap_server": imap_server,
            "username": username,
            "password": password
        }
        logger.info(f"Added email config for {service}")
    
    async def get_email_code(self, service: str, sender_filter: str = None, subject_filter: str = None) -> Optional[str]:
        """Get 2FA code from email."""
        try:
            if service not in self.email_configs:
                logger.error(f"No email config found for {service}")
                return None
            
            config = self.email_configs[service]
            
            # Connect to IMAP server
            imap = imaplib.IMAP4_SSL(config["imap_server"])
            imap.login(config["username"], config["password"])
            
            # Select inbox
            imap.select("INBOX")
            
            # Search for recent emails
            search_criteria = '(UNSEEN)'
            if sender_filter:
                search_criteria += f' (FROM "{sender_filter}")'
            if subject_filter:
                search_criteria += f' (SUBJECT "{subject_filter}")'
            
            _, messages = imap.search(None, search_criteria)
            
            for msg_id in messages[0].split():
                _, msg_data = imap.fetch(msg_id, '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Extract email body
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode()
                        
                        # Look for 6-digit code
                        import re
                        code_match = re.search(r'\b(\d{6})\b', body)
                        if code_match:
                            code = code_match.group(1)
                            imap.store(msg_id, '+FLAGS', '\\Seen')
                            imap.logout()
                            logger.info(f"Found 2FA code in email: {code}")
                            return code
            
            imap.logout()
            logger.info("No 2FA code found in recent emails")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get email code: {str(e)}")
            return None


class AntiDetectionEngine:
    """Anti-detection and stealth techniques."""
    
    def __init__(self, proxy_config: ProxyConfig = None):
        self.proxy_config = proxy_config or ProxyConfig()
        self.current_proxy_index = 0
        self.last_rotation_time = time.time()
        
        # User agent database
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0",
        ]
        
        # Viewport sizes
        self.viewports = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1440, "height": 900},
            {"width": 1536, "height": 864},
        ]
        
        # Timezones
        self.timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "Europe/London",
            "Europe/Paris",
            "Asia/Tokyo",
        ]
        
        logger.info("AntiDetectionEngine initialized")
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        return random.choice(self.user_agents)
    
    def get_random_viewport(self) -> Dict[str, int]:
        """Get random viewport dimensions."""
        return random.choice(self.viewports)
    
    def get_random_timezone(self) -> str:
        """Get a random timezone."""
        return random.choice(self.timezones)
    
    def get_current_proxy(self) -> Optional[str]:
        """Get current proxy or None if not configured."""
        if not self.proxy_config.enabled or not self.proxy_config.proxy_list:
            return None
        
        # Check if rotation is needed
        if (self.proxy_config.rotation_enabled and 
            time.time() - self.last_rotation_time > self.proxy_config.rotation_interval):
            self.rotate_proxy()
        
        return self.proxy_config.proxy_list[self.current_proxy_index]
    
    def rotate_proxy(self):
        """Rotate to next proxy."""
        if not self.proxy_config.proxy_list:
            return
        
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_config.proxy_list)
        self.last_rotation_time = time.time()
        
        logger.info(f"Rotated to proxy: {self.proxy_config.proxy_list[self.current_proxy_index]}")
    
    def generate_random_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0) -> float:
        """Generate random delay for human-like behavior."""
        return random.uniform(min_seconds, max_seconds)
    
    def generate_mouse_movement(self) -> List[Tuple[int, int]]:
        """Generate realistic mouse movement path."""
        # Simple bezier curve simulation for mouse movement
        start_x, start_y = random.randint(100, 500), random.randint(100, 500)
        end_x, end_y = random.randint(600, 1200), random.randint(300, 800)
        
        # Generate intermediate points
        control_x = random.randint(min(start_x, end_x), max(start_x, end_x))
        control_y = random.randint(min(start_y, end_y), max(start_y, end_y))
        
        path = []
        steps = random.randint(10, 30)
        
        for i in range(steps + 1):
            t = i / steps
            # Quadratic bezier curve
            x = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
            path.append ( (int(x), int(y)) )
        
        return path
    
    def generate_typing_pattern(self, text: str) -> List[Tuple[str, float]]:
        """Generate realistic typing pattern with delays."""
        pattern = []
        
        for char in text:
            # Variable delay between keystrokes
            if char in ' \n\t':
                delay = random.uniform(0.1, 0.3)  # Longer delay for spaces
            elif char.isupper():
                delay = random.uniform(0.15, 0.25)  # Slightly longer for shift key
            else:
                delay = random.uniform(0.05, 0.15)  # Normal typing
            
            pattern.append((char, delay))
        
        return pattern


class WebsiteSecurityHandler:
    """Main security handler orchestrator."""
    
    def __init__(self, captcha_config: CaptchaConfig = None, proxy_config: ProxyConfig = None):
        self.captcha_solver = CaptchaSolver(captcha_config) if captcha_config else None
        self.two_factor_handler = TwoFactorHandler()
        self.anti_detection = AntiDetectionEngine(proxy_config)
        
        self.security_events = []
        
        logger.info("WebsiteSecurityHandler initialized")
    
    async def detect_and_solve_captcha(self, page_content: str, page_url: str) -> Optional[str]:
        """Detect and solve CAPTCHA on the page."""
        if not self.captcha_solver:
            logger.warning("No CAPTCHA solver configured")
            return None
        
        try:
            # Look for reCAPTCHA v2
            if 'google.com/recaptcha/api.js' in page_content or 'g-recaptcha' in page_content:
                # Extract sitekey
                import re
                sitekey_match = re.search(r'data-sitekey="([^"]+)"', page_content)
                if sitekey_match:
                    sitekey = sitekey_match.group(1)
                    solution = await self.captcha_solver.solve_recaptcha_v2(sitekey, page_url)
                    
                    if solution:
                        self.security_events.append({
                            "type": "captcha_solved",
                            "captcha_type": "recaptcha_v2",
                            "success": True,
                            "timestamp": time.time()
                        })
                        return solution
            
            # Look for hCaptcha
            elif 'hcaptcha.com' in page_content or 'h-captcha' in page_content:
                import re
                sitekey_match = re.search(r'data-sitekey="([^"]+)"', page_content)
                if sitekey_match:
                    sitekey = sitekey_match.group(1)
                    solution = await self.captcha_solver.solve_hcaptcha(sitekey, page_url)
                    
                    if solution:
                        self.security_events.append({
                            "type": "captcha_solved",
                            "captcha_type": "hcaptcha",
                            "success": True,
                            "timestamp": time.time()
                        })
                        return solution
            
            logger.info("No CAPTCHA detected on page")
            return None
            
        except Exception as e:
            logger.error(f"CAPTCHA detection/solving failed: {str(e)}")
            return None
    
    def get_stealth_browser_config(self) -> Dict[str, Any]:
        """Get stealth browser configuration."""
        return {
            "user_agent": self.anti_detection.get_random_user_agent(),
            "viewport": self.anti_detection.get_random_viewport(),
            "proxy": self.anti_detection.get_current_proxy(),
            "timezone": self.anti_detection.get_random_timezone(),
        }
    
    async def handle_two_factor(self, service: str, method: TwoFactorType) -> Optional[str]:
        """Handle two-factor authentication."""
        try:
            if method == TwoFactorType.TOTP:
                code = self.two_factor_handler.generate_totp_code(service)
            elif method == TwoFactorType.EMAIL:
                code = await self.two_factor_handler.get_email_code(service)
            else:
                logger.warning(f"2FA method {method.value} not yet implemented")
                return None
            
            if code:
                self.security_events.append({
                    "type": "2fa_handled",
                    "service": service,
                    "method": method.value,
                    "success": True,
                    "timestamp": time.time()
                })
            
            return code
            
        except Exception as e:
            logger.error(f"2FA handling failed: {str(e)}")
            return None
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security handling summary."""
        return {
            "captcha_stats": self.captcha_solver.get_solve_statistics() if self.captcha_solver else None,
            "security_events": self.security_events[-20:],  # Last 20 events
            "current_proxy": self.anti_detection.get_current_proxy(),
            "proxy_rotation_enabled": self.anti_detection.proxy_config.rotation_enabled,
        }


# Example usage
async def main():
    """Example usage of the WebsiteSecurityHandler."""
    
    # Configuration
    captcha_config = CaptchaConfig(
        api_key="YOUR_2CAPTCHA_API_KEY"  # Replace with actual API key
    )
    
    proxy_config = ProxyConfig(
        enabled=True,
        proxy_list=[
            "http://proxy1.example.com:8080",
            "http://proxy2.example.com:8080",
        ]
    )
    
    # Initialize security handler
    security_handler = WebsiteSecurityHandler(captcha_config, proxy_config)
    
    # Add TOTP secret for a service
    security_handler.two_factor_handler.add_totp_secret("example_service", "JBSWY3DPEHPK3PXP")
    
    # Get stealth configuration
    stealth_config = security_handler.get_stealth_browser_config()
    print(f"Stealth browser config: {stealth_config}")
    
    # Generate TOTP code
    totp_code = security_handler.two_factor_handler.generate_totp_code("example_service")
    if totp_code:
        print(f"Generated TOTP code: {totp_code}")
    
    # Get security summary
    summary = security_handler.get_security_summary()
    print(f"Security summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())