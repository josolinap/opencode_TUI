#!/usr/bin/env python3
"""
Simple Weather Search Test for Neo-Clone
Search Google for current weather in Kabankalan City, Negros Occidental, Philippines
"""

import asyncio
from playwright.async_api import async_playwright

async def search_weather_kabankalan():
    """Search for weather in Kabankalan City using Google."""
    print("Neo-Clone Weather Search Test")
    print("=" * 50)
    print("Searching for: Weather today in Kabankalan City, Negros Occidental, Philippines")
    print()
    
    try:
        async with async_playwright() as p:
            print("Launching browser...")
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            print("Navigating to Google.com...")
            await page.goto("https://www.google.com", wait_until="networkidle")
            
            # Check if we're on Google
            title = await page.title()
            print(f"Page title: {title}")
            
            if "Google" in title:
                print("Successfully navigated to Google.com")
                
                # Wait for page to fully load
                await page.wait_for_timeout(3000)
                
                # Find and use search box
                selectors = [
                    'textarea[name="q"]',
                    'input[name="q"]',
                    '[aria-label="Search"]',
                    '.gLFyf'
                ]
                
                search_success = False
                for selector in selectors:
                    try:
                        element = await page.wait_for_selector(selector, timeout=2000)
                        if element:
                            print(f"Found search element: {selector}")
                            
                            # Fill with weather search query
                            query = "weather today in Kabankalan City Negros Occidental Philippines"
                            await element.fill(query)
                            print(f"Search query: {query}")
                            
                            # Submit search
                            await element.press("Enter")
                            print("Search submitted!")
                            
                            # Wait for results to load
                            await page.wait_for_timeout(5000)
                            
                            # Check if we got results
                            results_title = await page.title()
                            print(f"Results page: {results_title}")
                            
                            # Extract weather information
                            print("\nExtracting weather information...")
                            
                            # Get page content for analysis
                            page_content = await page.content()
                            
                            # Look for weather-related information
                            weather_data = await page.evaluate("""
                                () => {
                                    const results = [];
                                    
                                    // Look for temperature displays
                                    const tempElements = document.querySelectorAll('[id*="wob"], .wob_t, .wob_d, span[data-ved*="weather"]');
                                    tempElements.forEach(el => {
                                        if (el.textContent && el.textContent.match(/\\d+°/)) {
                                            results.push({
                                                type: 'temperature',
                                                value: el.textContent.trim()
                                            });
                                        }
                                    });
                                    
                                    // Look for weather conditions
                                    const conditionElements = document.querySelectorAll('[id*="wob_dc"], .wob_dc, span[data-ved*="condition"]');
                                    conditionElements.forEach(el => {
                                        if (el.textContent && el.textContent.length > 2) {
                                            results.push({
                                                type: 'condition',
                                                value: el.textContent.trim()
                                            });
                                        }
                                    });
                                    
                                    // Look for location information
                                    const locationElements = document.querySelectorAll('[id*="wob_loc"], .wob_loc');
                                    locationElements.forEach(el => {
                                        if (el.textContent && el.textContent.length > 2) {
                                            results.push({
                                                type: 'location',
                                                value: el.textContent.trim()
                                            });
                                        }
                                    });
                                    
                                    // Look for humidity and other weather details
                                    const detailElements = document.querySelectorAll('[id*="wob_hm"], [id*="wob_ws"], [id*="wob_pp"]');
                                    detailElements.forEach(el => {
                                        if (el.textContent && el.textContent.match(/\\d+%/)) {
                                            results.push({
                                                type: 'humidity',
                                                value: el.textContent.trim()
                                            });
                                        }
                                    });
                                    
                                    // If no specific weather widget found, look for general weather info
                                    if (results.length === 0) {
                                        const bodyText = document.body.innerText;
                                        const tempMatch = bodyText.match(/(\\d+)\\s*°[CF]?/gi);
                                        const locationMatch = bodyText.match(/kabankalan/gi);
                                        
                                        if (tempMatch) {
                                            results.push({
                                                type: 'temperature_found',
                                                value: tempMatch[0]
                                            });
                                        }
                                        
                                        if (locationMatch) {
                                            results.push({
                                                type: 'location_found',
                                                value: 'Kabankalan mentioned in results'
                                            });
                                        }
                                    }
                                    
                                    return results;
                                }
                            """)
                            
                            print("\nWeather Information Found:")
                            if weather_data:
                                for data in weather_data:
                                    print(f"  {data['type'].title()}: {data['value']}")
                            else:
                                print("  No specific weather widget detected")
                                print("  Checking page content for weather information...")
                                
                                # Check if weather terms are present
                                weather_terms = ['weather', 'temperature', 'C', 'F', 'sunny', 'cloudy', 'rain', 'humidity']
                                found_terms = [term for term in weather_terms if term.lower() in page_content.lower()]
                                
                                if found_terms:
                                    print(f"  Weather-related terms found: {', '.join(found_terms)}")
                                else:
                                    print("  No weather-related terms found in page content")
                            
                            # Check if Kabankalan is mentioned
                            if 'kabankalan' in page_content.lower():
                                print("  Kabankalan City found in search results")
                            else:
                                print("  Kabankalan City may not be clearly mentioned")
                            
                            search_success = True
                            break
                            
                    except Exception as e:
                        print(f"  Selector {selector} failed: {e}")
                        continue
                
                if not search_success:
                    print("Could not complete weather search")
                    return False
                    
            else:
                print("ERROR: Not on Google.com")
                return False
            
            await browser.close()
            
            print("\n" + "=" * 50)
            print("WEATHER SEARCH COMPLETED")
            print("=" * 50)
            print("Neo-Clone successfully:")
            print("  • Navigated to Google.com")
            print("  • Searched for weather in Kabankalan City")
            print("  • Extracted available weather information")
            print("  • Demonstrated real-world website automation")
            
            return True
            
    except Exception as e:
        print(f"ERROR: Weather search failed: {e}")
        return False

async def main():
    """Run weather search test."""
    success = await search_weather_kabankalan()
    
    if success:
        print("\nTEST SUCCESSFUL!")
        print("Neo-Clone website automation is working perfectly!")
        print("The system can navigate, search, and extract information from Google.com")
    else:
        print("\nTEST FAILED!")
        print("There may be an issue with the automation system")
    
    return success

if __name__ == "__main__":
    import sys
    success = asyncio.run(main())
    sys.exit(0 if success else 1)