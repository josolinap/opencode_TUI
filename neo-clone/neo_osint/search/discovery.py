"""
Enhanced search engine discovery and content scraping
"""

import asyncio
import aiohttp
import random
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..core.config import NeoOSINTConfig, SearchEngineConfig


@dataclass
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    snippet: str
    engine: str
    relevance_score: float = 0.0
    timestamp: float = 0.0


class SearchEngineDiscovery:
    """Enhanced search engine discovery and scraping system"""
    
    def __init__(self, config: NeoOSINTConfig):
        self.config = config
        self.logger = logging.getLogger("neo_osint.search")
        
        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:137.0) Gecko/20100101 Firefox/137.0",
            "Mozilla/5.0 (X11; Linux i686; rv:137.0) Gecko/20100101 Firefox/137.0"
        ]
        
        # Initialize search engines
        self.search_engines = self._initialize_search_engines()
        
        # Rate limiting
        self.last_request_time = {}
        self.request_lock = asyncio.Lock()
    
    def _initialize_search_engines(self) -> List[SearchEngineConfig]:
        """Initialize search engines from config"""
        engines = self.config.search_engines.copy()
        
        # If no engines configured, use defaults
        if not engines:
            engines = self.config.get_default_search_engines()
        
        # Filter enabled engines
        return [engine for engine in engines if engine.enabled]
    
    def _get_tor_proxies(self) -> Dict[str, str]:
        """Get Tor proxy configuration"""
        if not self.config.security.use_tor:
            return {}
        
        return {
            "http": f"socks5h://127.0.0.1:{self.config.security.tor_socks_port}",
            "https": f"socks5h://127.0.0.1:{self.config.security.tor_socks_port}"
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get random headers for request"""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        if self.config.security.user_agent_rotation:
            headers["User-Agent"] = random.choice(self.user_agents)
        else:
            headers["User-Agent"] = self.user_agents[0]
        
        return headers
    
    async def _rate_limit(self, engine_name: str) -> None:
        """Implement rate limiting for search engines"""
        async with self.request_lock:
            now = time.time()
            last_time = self.last_request_time.get(engine_name, 0)
            
            min_interval = 1.0 / self.config.security.max_request_rate
            elapsed = now - last_time
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                await asyncio.sleep(sleep_time)
            
            self.last_request_time[engine_name] = time.time()
    
    async def _search_single_engine(
        self,
        engine: SearchEngineConfig,
        query: str,
        session: aiohttp.ClientSession
    ) -> List[SearchResult]:
        """Search a single search engine"""
        await self._rate_limit(engine.name)
        
        url = engine.url.format(query=query.replace(" ", "+"))
        headers = self._get_headers()
        proxies = self._get_tor_proxies()
        
        try:
            timeout = aiohttp.ClientTimeout(total=engine.timeout)
            
            async with session.get(
                url,
                headers=headers,
                proxy=proxies.get("http") if proxies else None,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_search_results(content, engine.name)
                else:
                    self.logger.warning(f"Engine {engine.name} returned status {response.status}")
                    return []
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout searching {engine.name}")
            return []
        except Exception as e:
            self.logger.error(f"Error searching {engine.name}: {str(e)}")
            return []
    
    def _parse_search_results(self, html_content: str, engine_name: str) -> List[SearchResult]:
        """Parse search results from HTML content"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            results = []
            
            # Common patterns for search results
            for link in soup.find_all('a', href=True):
                href = link['href']
                title = link.get_text(strip=True)
                
                # Extract onion URLs
                onion_match = re.search(r'https?://[^\/]*\.onion[^\/]*', href)
                if onion_match:
                    onion_url = onion_match.group(0)
                    
                    # Get surrounding text as snippet
                    snippet = ""
                    parent = link.parent
                    if parent:
                        snippet = parent.get_text(strip=True)
                        snippet = re.sub(r'\s+', ' ', snippet)[:200]
                    
                    result = SearchResult(
                        title=title,
                        url=onion_url,
                        snippet=snippet,
                        engine=engine_name,
                        timestamp=time.time()
                    )
                    results.append(result)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing results from {engine_name}: {str(e)}")
            return []
    
    async def search(
        self,
        query: str,
        max_results: int = 50,
        include_clear_web: bool = False
    ) -> List[Dict[str, Any]]:
        """Search across multiple engines"""
        self.logger.info(f"Searching for: {query}")
        
        # Filter engines based on clear web preference
        engines_to_use = []
        for engine in self.search_engines:
            is_clear_web = not engine.url.startswith("http://") and ".onion" not in engine.url
            if include_clear_web or not is_clear_web:
                engines_to_use.append(engine)
        
        # Sort by priority
        engines_to_use.sort(key=lambda x: x.priority)
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_requests,
            limit_per_host=5
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        ) as session:
            # Search all engines concurrently
            tasks = [
                self._search_single_engine(engine, query, session)
                for engine in engines_to_use
            ]
            
            engine_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            all_results = []
            for results in engine_results:
                if isinstance(results, list):
                    all_results.extend(results)
            
            # Deduplicate by URL
            seen_urls = set()
            unique_results = []
            for result in all_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            
            # Sort by relevance and limit results
            unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
            limited_results = unique_results[:max_results]
            
            # Convert to dict format
            return [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "engine": result.engine,
                    "relevance_score": result.relevance_score,
                    "timestamp": result.timestamp
                }
                for result in limited_results
            ]
    
    async def scrape_content(
        self,
        results: List[Dict[str, Any]],
        max_workers: int = 10
    ) -> Dict[str, str]:
        """Scrape content from search results"""
        self.logger.info(f"Scraping content from {len(results)} URLs")
        
        content_map = {}
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_workers)
        
        async def scrape_single(result: Dict[str, Any]) -> Tuple[str, str]:
            async with semaphore:
                return await self._scrape_single_url(result)
        
        # Scrape all URLs concurrently
        tasks = [scrape_single(result) for result in results]
        scraped_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for data in scraped_data:
            if isinstance(data, tuple):
                url, content = data
                content_map[url] = content
            elif isinstance(data, Exception):
                self.logger.error(f"Scraping error: {str(data)}")
        
        self.logger.info(f"Successfully scraped {len(content_map)} pages")
        return content_map
    
    async def _scrape_single_url(self, result: Dict[str, Any]) -> Tuple[str, str]:
        """Scrape content from a single URL"""
        url = result["url"]
        title = result.get("title", "")
        
        headers = self._get_headers()
        proxies = self._get_tor_proxies()
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url,
                    headers=headers,
                    proxy=proxies.get("http") if proxies else None
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract text content
                        soup = BeautifulSoup(content, "html.parser")
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        text = soup.get_text()
                        # Clean up text
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        # Limit content length
                        max_length = 2000
                        if len(text) > max_length:
                            text = text[:max_length] + "..."
                        
                        return url, f"{title}\n{text}"
                    else:
                        return url, title
        
        except Exception as e:
            self.logger.warning(f"Error scraping {url}: {str(e)}")
            return url, title
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Close any open connections
        pass