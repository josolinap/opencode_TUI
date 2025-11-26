#!/usr/bin/env python3
"""
🌐 Neo-Clone Website Automation Core
====================================

Advanced browser automation system combining Playwright and SeleniumBase
for maximum reliability and stealth capabilities.
"""

import asyncio
import json
import logging
import time
import base64
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import uuid

# Core automation libraries
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    from playwright_stealth import stealth_async
except ImportError:
    print("Installing Playwright and stealth dependencies...")
    os.system("pip install playwright playwright-stealth")
    os.system("playwright install")
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    from playwright_stealth import stealth_async

try:
    from seleniumbase import DriverContext
    from seleniumbase import sb
except ImportError:
    print("Installing SeleniumBase...")
    os.system("pip install seleniumbase")
    from seleniumbase import DriverContext
    from seleniumbase import sb

# Additional utilities
from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrowserType(Enum):
    """Supported browser types."""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class AutomationFramework(Enum):
    """Available automation frameworks."""
    PLAYWRIGHT = "playwright"
    SELENIUMBASE = "seleniumbase"


@dataclass
class BrowserConfig:
    """Browser configuration settings."""
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    viewport: Dict[str, int] = None
    user_agent: Optional[str] = None
    proxy: Optional[Dict[str, str]] = None
    ignore_https_errors: bool = True
    javascript_enabled: bool = True
    
    def __post_init__(self):
        if self.viewport is None:
            self.viewport = {"width": 1920, "height": 1080}


@dataclass
class StealthConfig:
    """Anti-detection configuration."""
    enable_stealth: bool = True
    random_user_agent: bool = True
    random_viewport: bool = False
    disable_webgl: bool = True
    disable_canvas_fingerprint: bool = True
    random_timezone: bool = True
    random_language: bool = True


class BrowserManager:
    """Unified browser management for both Playwright and SeleniumBase."""
    
    def __init__(self, config: BrowserConfig = None, stealth_config: StealthConfig = None):
        self.config = config or BrowserConfig()
        self.stealth_config = stealth_config or StealthConfig()
        
        # Playwright components
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # SeleniumBase components
        self.sb_driver = None
        self.sb_context = None
        
        # State tracking
        self.active_framework = None
        self.session_id = str(uuid.uuid4())
        self.is_initialized = False
        
        logger.info(f"BrowserManager initialized with session ID: {self.session_id}")
    
    async def initialize_playwright(self) -> bool:
        """Initialize Playwright browser instance."""
        try:
            self.playwright = await async_playwright().start()
            
            # Browser launch options
            launch_options = {
                "headless": self.config.headless,
                "ignore_https_errors": self.config.ignore_https_errors,
            }
            
            # Add proxy if configured
            if self.config.proxy:
                launch_options["proxy"] = self.config.proxy
            
            # Launch browser based on type
            if self.config.browser_type == BrowserType.CHROMIUM:
                self.browser = await self.playwright.chromium.launch(**launch_options)
            elif self.config.browser_type == BrowserType.FIREFOX:
                self.browser = await self.playwright.firefox.launch(**launch_options)
            elif self.config.browser_type == BrowserType.WEBKIT:
                self.browser = await self.playwright.webkit.launch(**launch_options)
            
            # Create context with stealth settings
            context_options = {
                "viewport": self.config.viewport,
                "java_script_enabled": self.config.javascript_enabled,
            }
            
            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent
            
            self.context = await self.browser.new_context(**context_options)
            self.page = await self.context.new_page()
            
            # Apply stealth techniques
            if self.stealth_config.enable_stealth:
                await stealth_async(self.page)
            
            self.active_framework = AutomationFramework.PLAYWRIGHT
            self.is_initialized = True
            
            logger.info("Playwright browser initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {str(e)}")
            return False
    
    def initialize_seleniumbase(self) -> bool:
        """Initialize SeleniumBase driver instance."""
        try:
            # SeleniumBase configuration
            sb_config = {
                "headless": self.config.headless,
                "browser": "chrome",  # SeleniumBase primarily uses Chrome
                "uc": True,  # Undetected mode
                "ad_block": True,
                "disable_ws": True,  # Disable webdriver signature
            }
            
            # Add proxy if configured
            if self.config.proxy:
                sb_config["proxy"] = f"{self.config.proxy.get('server')}"
            
            # Create driver context
            self.sb_context = DriverContext(**sb_config)
            self.sb_driver = self.sb_context.driver
            
            self.active_framework = AutomationFramework.SELENIUMBASE
            self.is_initialized = True
            
            logger.info("SeleniumBase driver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SeleniumBase: {str(e)}")
            return False
    
    async def navigate_to(self, url: str, framework: AutomationFramework = None) -> bool:
        """Navigate to a specified URL."""
        if not self.is_initialized:
            logger.error("Browser not initialized")
            return False
        
        target_framework = framework or self.active_framework
        
        try:
            if target_framework == AutomationFramework.PLAYWRIGHT:
                await self.page.goto(url, wait_until="domcontentloaded")
                logger.info(f"Playwright navigated to: {url}")
                return True
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                self.sb_driver.get(url)
                logger.info(f"SeleniumBase navigated to: {url}")
                return True
                
        except Exception as e:
            logger.error(f"Navigation failed: {str(e)}")
            return False
    
    async def take_screenshot(self, filename: str = None, framework: AutomationFramework = None) -> str:
        """Take a screenshot of the current page."""
        if not self.is_initialized:
            return None
        
        target_framework = framework or self.active_framework
        
        try:
            if not filename:
                filename = f"screenshot_{int(time.time())}_{self.session_id}.png"
            
            if target_framework == AutomationFramework.PLAYWRIGHT:
                await self.page.screenshot(path=filename)
                logger.info(f"Playwright screenshot saved: {filename}")
                return filename
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                self.sb_driver.save_screenshot(filename)
                logger.info(f"SeleniumBase screenshot saved: {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Screenshot failed: {str(e)}")
            return None
    
    async def get_page_content(self, framework: AutomationFramework = None) -> str:
        """Get the HTML content of the current page."""
        if not self.is_initialized:
            return None
        
        target_framework = framework or self.active_framework
        
        try:
            if target_framework == AutomationFramework.PLAYWRIGHT:
                content = await self.page.content()
                return content
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                content = self.sb_driver.page_source
                return content
                
        except Exception as e:
            logger.error(f"Failed to get page content: {str(e)}")
            return None
    
    async def wait_for_element(self, selector: str, timeout: int = 10000, framework: AutomationFramework = None) -> bool:
        """Wait for an element to appear on the page."""
        if not self.is_initialized:
            return False
        
        target_framework = framework or self.active_framework
        
        try:
            if target_framework == AutomationFramework.PLAYWRIGHT:
                await self.page.wait_for_selector(selector, timeout=timeout)
                return True
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                self.sb_driver.wait_for_element(selector, timeout=timeout)
                return True
                
        except Exception as e:
            logger.error(f"Element wait failed: {str(e)}")
            return False
    
    async def click_element(self, selector: str, framework: AutomationFramework = None) -> bool:
        """Click on an element."""
        if not self.is_initialized:
            return False
        
        target_framework = framework or self.active_framework
        
        try:
            if target_framework == AutomationFramework.PLAYWRIGHT:
                await self.page.click(selector)
                return True
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                self.sb_driver.click(selector)
                return True
                
        except Exception as e:
            logger.error(f"Click failed: {str(e)}")
            return False
    
    async def type_text(self, selector: str, text: str, clear_first: bool = True, framework: AutomationFramework = None) -> bool:
        """Type text into an input field."""
        if not self.is_initialized:
            return False
        
        target_framework = framework or self.active_framework
        
        try:
            if target_framework == AutomationFramework.PLAYWRIGHT:
                if clear_first:
                    await self.page.fill(selector, "")
                await self.page.fill(selector, text)
                return True
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                if clear_first:
                    self.sb_driver.clear(selector)
                self.sb_driver.type(selector, text)
                return True
                
        except Exception as e:
            logger.error(f"Type failed: {str(e)}")
            return False
    
    async def execute_javascript(self, script: str, framework: AutomationFramework = None) -> Any:
        """Execute JavaScript in the browser context."""
        if not self.is_initialized:
            return None
        
        target_framework = framework or self.active_framework
        
        try:
            if target_framework == AutomationFramework.PLAYWRIGHT:
                result = await self.page.evaluate(script)
                return result
            
            elif target_framework == AutomationFramework.SELENIUMBASE:
                result = self.sb_driver.execute_script(script)
                return result
                
        except Exception as e:
            logger.error(f"JavaScript execution failed: {str(e)}")
            return None
    
    async def switch_to_playwright(self) -> bool:
        """Switch to Playwright framework."""
        if self.active_framework == AutomationFramework.PLAYWRIGHT:
            return True
        
        # Close SeleniumBase if active
        if self.sb_driver:
            try:
                self.sb_driver.quit()
            except:
                pass
        
        # Initialize Playwright if not already done
        if not self.playwright:
            return await self.initialize_playwright()
        
        self.active_framework = AutomationFramework.PLAYWRIGHT
        return True
    
    def switch_to_seleniumbase(self) -> bool:
        """Switch to SeleniumBase framework."""
        if self.active_framework == AutomationFramework.SELENIUMBASE:
            return True
        
        # Close Playwright if active
        if self.browser:
            try:
                asyncio.create_task(self.browser.close())
            except:
                pass
        
        # Initialize SeleniumBase if not already done
        if not self.sb_driver:
            return self.initialize_seleniumbase()
        
        self.active_framework = AutomationFramework.SELENIUMBASE
        return True
    
    async def close(self):
        """Close all browser instances and cleanup."""
        try:
            # Close Playwright
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            # Close SeleniumBase
            if self.sb_driver:
                self.sb_driver.quit()
            
            logger.info("Browser manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        finally:
            self.is_initialized = False
            self.active_framework = None


class WebsiteAutomationCore:
    """Main website automation orchestrator."""
    
    def __init__(self, config: BrowserConfig = None, stealth_config: StealthConfig = None):
        self.browser_manager = BrowserManager(config, stealth_config)
        self.session_data = {}
        self.automation_history = []
        
        logger.info("WebsiteAutomationCore initialized")
    
    async def start_session(self, framework: AutomationFramework = AutomationFramework.PLAYWRIGHT) -> bool:
        """Start a new automation session."""
        try:
            if framework == AutomationFramework.PLAYWRIGHT:
                success = await self.browser_manager.initialize_playwright()
            else:
                success = self.browser_manager.initialize_seleniumbase()
            
            if success:
                self.session_data = {
                    "session_id": self.browser_manager.session_id,
                    "framework": framework.value,
                    "start_time": time.time(),
                    "browser_type": self.browser_manager.config.browser_type.value,
                    "headless": self.browser_manager.config.headless,
                }
                
                self.automation_history.append({
                    "action": "session_start",
                    "timestamp": time.time(),
                    "framework": framework.value,
                    "success": True
                })
                
                logger.info(f"Session started with {framework.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            return False
    
    async def visit_website(self, url: str) -> bool:
        """Visit a website."""
        try:
            success = await self.browser_manager.navigate_to(url)
            
            self.automation_history.append({
                "action": "visit_website",
                "timestamp": time.time(),
                "url": url,
                "success": success
            })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to visit website: {str(e)}")
            return False
    
    async def get_page_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current page."""
        try:
            content = await self.browser_manager.get_page_content()
            if not content:
                return None
            
            soup = BeautifulSoup(content, 'html.parser')
            
            page_info = {
                "url": await self.browser_manager.page.url if self.browser_manager.page else "Unknown",
                "title": soup.title.string if soup.title else "No title",
                "content_length": len(content),
                "forms": len(soup.find_all('form')),
                "inputs": len(soup.find_all('input')),
                "buttons": len(soup.find_all('button')),
                "links": len(soup.find_all('a')),
                "images": len(soup.find_all('img')),
                "scripts": len(soup.find_all('script')),
                "iframes": len(soup.find_all('iframe')),
            }
            
            return page_info
            
        except Exception as e:
            logger.error(f"Failed to get page info: {str(e)}")
            return None
    
    async def end_session(self):
        """End the current automation session."""
        try:
            await self.browser_manager.close()
            
            self.automation_history.append({
                "action": "session_end",
                "timestamp": time.time(),
                "success": True
            })
            
            logger.info("Session ended successfully")
            
        except Exception as e:
            logger.error(f"Failed to end session: {str(e)}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        if not self.session_data:
            return None
        
        duration = time.time() - self.session_data.get("start_time", time.time())
        
        summary = {
            "session_data": self.session_data,
            "duration_seconds": duration,
            "actions_count": len(self.automation_history),
            "automation_history": self.automation_history,
        }
        
        return summary


# Example usage and testing
async def main():
    """Example usage of the WebsiteAutomationCore."""
    
    # Configuration
    browser_config = BrowserConfig(
        browser_type=BrowserType.CHROMIUM,
        headless=False,  # Set to True for production
        viewport={"width": 1920, "height": 1080}
    )
    
    stealth_config = StealthConfig(
        enable_stealth=True,
        random_user_agent=True
    )
    
    # Initialize automation core
    automation = WebsiteAutomationCore(browser_config, stealth_config)
    
    try:
        # Start session
        print("🚀 Starting automation session...")
        success = await automation.start_session()
        
        if success:
            # Visit a website
            print("🌐 Visiting example.com...")
            await automation.visit_website("https://example.com")
            
            # Get page information
            print("📊 Getting page information...")
            page_info = await automation.get_page_info()
            if page_info:
                print(f"Page Title: {page_info['title']}")
                print(f"Forms found: {page_info['forms']}")
                print(f"Links found: {page_info['links']}")
            
            # Take screenshot
            print("📸 Taking screenshot...")
            screenshot_path = await automation.browser_manager.take_screenshot()
            if screenshot_path:
                print(f"Screenshot saved: {screenshot_path}")
            
            # Get session summary
            summary = automation.get_session_summary()
            if summary:
                print(f"Session duration: {summary['duration_seconds']:.2f} seconds")
                print(f"Actions performed: {summary['actions_count']}")
        
    except Exception as e:
        print(f"❌ Error during automation: {str(e)}")
    
    finally:
        # End session
        print("🔚 Ending automation session...")
        await automation.end_session()


if __name__ == "__main__":
    asyncio.run(main())