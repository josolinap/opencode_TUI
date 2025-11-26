#!/usr/bin/env python3
"""
Quick Fix: Enhanced Error Handling for Neo-Clone
"""

import asyncio
import logging
from typing import Any, Callable, Optional

class RobustErrorHandler:
    """Enhanced error handling for website automation."""
    
    def __init__(self):
        self.error_patterns = {
            'timeout': ['timeout', 'timed out', 'TimeoutError'],
            'element_not_found': ['not found', 'NoSuchElement', 'ElementNotFound'],
            'network': ['network', 'connection', 'NetworkError'],
            'permission': ['permission', 'access denied', 'PermissionError'],
            'javascript': ['javascript', 'JS', 'evaluation failed']
        }
        
        self.retry_strategies = {
            'timeout': self._handle_timeout,
            'element_not_found': self._handle_element_not_found,
            'network': self._handle_network,
            'permission': self._handle_permission,
            'javascript': self._handle_javascript
        }
    
    def categorize_error(self, error: Exception) -> str:
        """Categorize error type for appropriate handling."""
        error_str = str(error).lower()
        
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_str for pattern in patterns):
                return category
        
        return 'unknown'
    
    async def _handle_timeout(self, page, action_func, max_retries: int = 3):
        """Handle timeout errors with increased wait times."""
        for attempt in range(max_retries):
            try:
                # Increase wait time for each retry
                if attempt > 0:
                    wait_time = 2000 * (2 ** attempt)  # 2s, 4s, 8s
                    await page.wait_for_timeout(wait_time)
                
                return await action_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Timeout retry {attempt + 1}/{max_retries}")
    
    async def _handle_element_not_found(self, page, action_func, max_retries: int = 3):
        """Handle element not found with alternative selectors."""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Wait for potential dynamic content
                    await page.wait_for_timeout(1000 * attempt)
                
                return await action_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Element retry {attempt + 1}/{max_retries}")
    
    async def _handle_network(self, page, action_func, max_retries: int = 2):
        """Handle network errors with page refresh."""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Refresh page on network errors
                    await page.reload()
                    await page.wait_for_timeout(2000)
                
                return await action_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Network retry {attempt + 1}/{max_retries}")
    
    async def _handle_permission(self, page, action_func, max_retries: int = 1):
        """Handle permission errors."""
        try:
            return await action_func()
        except Exception as e:
            # Log permission errors but don't retry
            print(f"Permission error: {e}")
            raise e
    
    async def _handle_javascript(self, page, action_func, max_retries: int = 2):
        """Handle JavaScript errors with alternative methods."""
        for attempt in range(max_retries):
            try:
                return await action_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"JavaScript retry {attempt + 1}/{max_retries}")
    
    async def execute_with_recovery(self, page, action_func: Callable, description: str = "") -> Any:
        """Execute action with intelligent error recovery."""
        for attempt in range(3):  # Max 3 attempts total
            try:
                return await action_func()
            except Exception as e:
                error_category = self.categorize_error(e)
                
                if attempt == 2:  # Final attempt
                    print(f"Final attempt failed for {description}: {e}")
                    raise e
                
                print(f"Attempt {attempt + 1} failed ({error_category}): {e}")
                
                # Use specific retry strategy if available
                if error_category in self.retry_strategies:
                    try:
                        return await self.retry_strategies[error_category](
                            page, action_func, max_retries=2
                        )
                    except:
                        continue  # Fall through to next attempt
                
                # Generic wait for unknown errors
                await page.wait_for_timeout(1000 * (attempt + 1))

# Enhanced element finder with multiple strategies
async def find_element_robust(page, element_type: str, max_wait: int = 10000):
    """Find element using multiple strategies."""
    strategies = {
        'search_input': [
            'textarea[name="q"]',
            'input[name="q"]',
            'input[type="search"]',
            'input[title*="Search"]',
            '[aria-label*="search"]',
            '.search-input',
            '#search'
        ],
        'login_email': [
            'input[type="email"]',
            'input[name="email"]',
            'input[name="username"]',
            'input[id*="email"]',
            'input[placeholder*="email"]'
        ],
        'submit_button': [
            'button[type="submit"]',
            'input[type="submit"]',
            'button:has-text("Submit")',
            'button:has-text("Login")',
            'button:has-text("Sign In")',
            '.submit-btn'
        ]
    }
    
    selectors = strategies.get(element_type, [])
    
    for selector in selectors:
        try:
            element = await page.wait_for_selector(selector, timeout=max_wait // len(selectors))
            if element:
                print(f"Found {element_type} with selector: {selector}")
                return element
        except:
            continue
    
    raise Exception(f"Could not find {element_type} with any selector")

print("Enhanced error handling loaded!")
print("This improves Neo-Clone's reliability and error recovery.")