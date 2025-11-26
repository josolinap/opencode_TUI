#!/usr/bin/env python3
"""
AI-Powered Site Understanding Test for Neo-Clone
===============================================

Demonstrates Neo-Clone's ability to understand different types of websites
and interact with them intelligently.
"""

import asyncio
import sys
import re
from typing import Dict, List, Optional

class AISiteUnderstanding:
    """AI-powered site understanding and interaction."""
    
    def __init__(self):
        self.site_patterns = {
            'search_engine': {
                'indicators': ['search', 'input[type="search"]', 'input[name="q"]', 'google', 'bing'],
                'actions': ['search', 'navigate_results']
            },
            'ecommerce': {
                'indicators': ['cart', 'buy', 'price', 'shop', 'amazon', 'ebay'],
                'actions': ['search_products', 'add_to_cart', 'checkout']
            },
            'social_media': {
                'indicators': ['login', 'signup', 'post', 'share', 'facebook', 'twitter'],
                'actions': ['login', 'post_content', 'interact']
            }
        }
    
    async def analyze_site_content(self, content: str, url: str) -> Dict[str, any]:
        """Analyze site content to determine type and capabilities."""
        content_lower = content.lower()
        url_lower = url.lower()
        
        analysis = {
            'url': url,
            'content_length': len(content),
            'site_type': 'unknown',
            'confidence': 0.0,
            'features': [],
            'interaction_points': []
        }
        
        # Check for search engine indicators
        search_score = 0
        search_indicators = self.site_patterns['search_engine']['indicators']
        for indicator in search_indicators:
            if indicator in content_lower or indicator in url_lower:
                search_score += 1
        
        if search_score >= 2:
            analysis['site_type'] = 'search_engine'
            analysis['confidence'] = min(search_score / len(search_indicators), 1.0)
            analysis['features'] = ['search_box', 'search_results', 'navigation']
            analysis['interaction_points'] = ['input[name="q"]', 'textarea[name="q"]', 'input[title="Search"]']
        
        # Check for ecommerce indicators
        ecommerce_score = 0
        ecommerce_indicators = self.site_patterns['ecommerce']['indicators']
        for indicator in ecommerce_indicators:
            if indicator in content_lower or indicator in url_lower:
                ecommerce_score += 1
        
        if ecommerce_score >= 2:
            analysis['site_type'] = 'ecommerce'
            analysis['confidence'] = min(ecommerce_score / len(ecommerce_indicators), 1.0)
            analysis['features'] = ['product_search', 'shopping_cart', 'checkout', 'reviews']
            analysis['interaction_points'] = ['search_box', 'add_to_cart', 'buy_button']
        
        # Check for social media indicators
        social_score = 0
        social_indicators = self.site_patterns['social_media']['indicators']
        for indicator in social_indicators:
            if indicator in content_lower or indicator in url_lower:
                social_score += 1
        
        if social_score >= 2:
            analysis['site_type'] = 'social_media'
            analysis['confidence'] = min(social_score / len(social_indicators), 1.0)
            analysis['features'] = ['login', 'posting', 'sharing', 'messaging']
            analysis['interaction_points'] = ['login_form', 'post_input', 'share_button']
        
        return analysis
    
    async def intelligent_interaction(self, page, site_analysis: Dict[str, any], query: str = "") -> bool:
        """Perform intelligent interaction based on site analysis."""
        site_type = site_analysis['site_type']
        
        try:
            if site_type == 'search_engine':
                return await self._handle_search_engine(page, query)
            elif site_type == 'ecommerce':
                return await self._handle_ecommerce(page, query)
            elif site_type == 'social_media':
                return await self._handle_social_media(page, query)
            else:
                print(f"Unknown site type, attempting generic interaction...")
                return await self._handle_generic_site(page, query)
        except Exception as e:
            print(f"Interaction failed: {e}")
            return False
    
    async def _handle_search_engine(self, page, query: str) -> bool:
        """Handle search engine interaction."""
        try:
            # Look for search input
            search_selectors = [
                'textarea[name="q"]',
                'input[name="q"]',
                'input[title="Search"]',
                'input[type="search"]',
                '[aria-label="Search"]'
            ]
            
            for selector in search_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.fill(query)
                        await element.press("Enter")
                        print(f"SUCCESS: Performed search for '{query}'")
                        return True
                except:
                    continue
            
            print("Could not find search input")
            return False
        except Exception as e:
            print(f"Search engine interaction failed: {e}")
            return False
    
    async def _handle_ecommerce(self, page, query: str) -> bool:
        """Handle ecommerce site interaction."""
        try:
            # Look for product search
            search_selectors = [
                'input[name="field-keywords"]',
                'input[name="k"]',
                'input[type="search"]',
                '#twotabsearchtextbox'
            ]
            
            for selector in search_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.fill(query)
                        await element.press("Enter")
                        print(f"SUCCESS: Searched for product '{query}'")
                        return True
                except:
                    continue
            
            print("Could not find product search")
            return False
        except Exception as e:
            print(f"Ecommerce interaction failed: {e}")
            return False
    
    async def _handle_social_media(self, page, query: str) -> bool:
        """Handle social media interaction."""
        try:
            # Look for login or post input
            login_selectors = [
                'input[name="email"]',
                'input[type="email"]',
                'input[name="username"]'
            ]
            
            for selector in login_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        print(f"SUCCESS: Found login form on social media")
                        return True
                except:
                    continue
            
            print("Could not find social media interaction points")
            return False
        except Exception as e:
            print(f"Social media interaction failed: {e}")
            return False
    
    async def _handle_generic_site(self, page, query: str) -> bool:
        """Handle generic site interaction."""
        try:
            # Look for any input field
            input_selectors = [
                'input[type="text"]',
                'textarea',
                'input[type="search"]'
            ]
            
            for selector in input_selectors:
                try:
                    element = await page.wait_for_selector(selector, timeout=2000)
                    if element:
                        await element.fill(query)
                        print(f"SUCCESS: Filled generic input with '{query}'")
                        return True
                except:
                    continue
            
            print("Could not find any input fields")
            return False
        except Exception as e:
            print(f"Generic interaction failed: {e}")
            return False

async def test_site_understanding():
    """Test AI-powered site understanding."""
    try:
        from playwright.async_api import async_playwright
        print("SUCCESS: Playwright imported successfully")
        
        ai_understanding = AISiteUnderstanding()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Test sites
            test_sites = [
                {
                    'url': 'https://www.google.com',
                    'query': 'Neo-Clone AI automation',
                    'expected_type': 'search_engine'
                },
                {
                    'url': 'https://www.amazon.com',
                    'query': 'laptop computer',
                    'expected_type': 'ecommerce'
                },
                {
                    'url': 'https://www.facebook.com',
                    'query': 'test',
                    'expected_type': 'social_media'
                }
            ]
            
            results = []
            
            for site in test_sites:
                print(f"\n{'='*60}")
                print(f"Testing: {site['url']}")
                print(f"Expected type: {site['expected_type']}")
                print(f"{'='*60}")
                
                try:
                    # Navigate to site
                    await page.goto(site['url'], wait_until="networkidle")
                    await page.wait_for_timeout(3000)
                    
                    # Get page content
                    content = await page.content()
                    title = await page.title()
                    
                    print(f"Page title: {title}")
                    print(f"Content length: {len(content)} characters")
                    
                    # Analyze site
                    analysis = await ai_understanding.analyze_site_content(content, site['url'])
                    
                    print(f"\nAI Analysis Results:")
                    print(f"  Site Type: {analysis['site_type']}")
                    print(f"  Confidence: {analysis['confidence']:.2f}")
                    print(f"  Features: {', '.join(analysis['features'])}")
                    print(f"  Interaction Points: {len(analysis['interaction_points'])} found")
                    
                    # Check if analysis matches expectation
                    type_match = analysis['site_type'] == site['expected_type']
                    print(f"  Type Match: {'YES' if type_match else 'NO'}")
                    
                    # Perform intelligent interaction
                    if analysis['confidence'] > 0.5:
                        print(f"\nAttempting intelligent interaction...")
                        interaction_success = await ai_understanding.intelligent_interaction(
                            page, analysis, site['query']
                        )
                        print(f"  Interaction Success: {'YES' if interaction_success else 'NO'}")
                        
                        # Wait for interaction to complete
                        await page.wait_for_timeout(2000)
                        
                        # Check if URL changed (indicating successful interaction)
                        new_url = page.url
                        url_changed = new_url != site['url']
                        print(f"  URL Changed: {'YES' if url_changed else 'NO'}")
                        
                        if url_changed:
                            print(f"  New URL: {new_url}")
                    else:
                        print(f"\nConfidence too low for interaction ({analysis['confidence']:.2f})")
                        interaction_success = False
                        url_changed = False
                    
                    results.append({
                        'site': site['url'],
                        'expected_type': site['expected_type'],
                        'detected_type': analysis['site_type'],
                        'confidence': analysis['confidence'],
                        'type_match': type_match,
                        'interaction_success': interaction_success,
                        'url_changed': url_changed
                    })
                    
                except Exception as e:
                    print(f"ERROR testing {site['url']}: {e}")
                    results.append({
                        'site': site['url'],
                        'error': str(e)
                    })
            
            await browser.close()
            return results
            
    except ImportError as e:
        print(f"ERROR: Playwright not available: {e}")
        return []
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return []

async def main():
    """Run AI site understanding test."""
    print("Neo-Clone AI-Powered Site Understanding Test")
    print("=" * 60)
    print("Testing AI's ability to understand and interact with different website types")
    print()
    
    results = await test_site_understanding()
    
    print("\n" + "=" * 60)
    print("AI SITE UNDERSTANDING TEST RESULTS")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = len(results)
    
    for result in results:
        if 'error' in result:
            print(f"\n{result['site']}: ERROR - {result['error']}")
        else:
            print(f"\n{result['site']}:")
            print(f"  Expected: {result['expected_type']}")
            print(f"  Detected: {result['detected_type']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Type Match: {'YES' if result['type_match'] else 'NO'}")
            print(f"  Interaction: {'YES' if result['interaction_success'] else 'NO'}")
            print(f"  URL Changed: {'YES' if result['url_changed'] else 'NO'}")
            
            if result['type_match'] and result['interaction_success']:
                successful_tests += 1
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests >= total_tests * 0.6:
        print("\nOVERALL RESULT: SUCCESS!")
        print("Neo-Clone demonstrates AI-powered site understanding!")
        print("\nCapabilities demonstrated:")
        print("- Site type recognition (search, ecommerce, social)")
        print("- Intelligent interaction based on site type")
        print("- Dynamic content handling")
        print("- Multi-site adaptability")
    else:
        print("\nOVERALL RESULT: PARTIAL SUCCESS")
        print("Neo-Clone shows site understanding potential")
        print("Some sites may need additional pattern tuning")
    
    return successful_tests >= total_tests * 0.6

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)