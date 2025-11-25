"""
Tool Caching and Optimization System for MCP Integration

This module provides intelligent caching, optimization, and performance
enhancement for MCP tools including result caching, parameter optimization,
and adaptive execution strategies.

Author: Neo-Clone Enhanced
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import time
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
from collections import OrderedDict

# Configure logging
logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"           # Adaptive based on usage patterns
    WRITE_THROUGH = "write_through"  # Write-through caching
    WRITE_BEHIND = "write_behind"   # Write-behind caching


class CacheEntry:
    """Cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl = ttl
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value))
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cached value"""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'ttl': self.ttl,
            'size': self.size
        }


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    
    def update_hit(self) -> None:
        """Update hit statistics"""
        self.hits += 1
        self._update_hit_rate()
    
    def update_miss(self) -> None:
        """Update miss statistics"""
        self.misses += 1
        self._update_hit_rate()
    
    def update_eviction(self) -> None:
        """Update eviction statistics"""
        self.evictions += 1
    
    def _update_hit_rate(self) -> None:
        """Calculate hit rate"""
        total_requests = self.hits + self.misses
        self.hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0


class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = asyncio.Lock()  # Async-safe lock
        # Add hash-based indexing for faster lookups
        self.key_cache: Dict[str, str] = {}  # Maps parameter hashes to cache keys
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache - OPTIMIZED async version"""
        async with self.lock:
            if key not in self.cache:
                self.stats.update_miss()
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats.update_miss()
                self.stats.entry_count -= 1
                self.stats.total_size -= entry.size
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats.update_hit()
            return entry.access()
    
    def _generate_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate optimized cache key from tool name and parameters"""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True, ensure_ascii=False)
        # Create hash for efficient indexing
        param_hash = hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
        return f"{tool_name}:{param_hash}"
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache - OPTIMIZED async version"""
        async with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size -= old_entry.size
                self.stats.entry_count -= 1
                del self.cache[key]
            
            # Create new entry
            entry_ttl = ttl or self.default_ttl
            entry = CacheEntry(key, value, entry_ttl)
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Add new entry
            self.cache[key] = entry
            self.stats.total_size += entry.size
            self.stats.entry_count += 1
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.stats.total_size -= entry.size
            self.stats.entry_count -= 1
            self.stats.update_eviction()
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.stats.total_size = 0
            self.stats.entry_count = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size=self.stats.total_size,
                entry_count=self.stats.entry_count,
                hit_rate=self.stats.hit_rate
            )


class ToolCacheSystem:
    """Comprehensive tool caching system"""
    
    def __init__(self, cache_dir: str = "data/cache", max_memory_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Different cache types
        self.result_cache = LRUCache(max_size=1000, ttl=300)  # 5 minutes
        self.parameter_cache = LRUCache(max_size=500, ttl=600)  # 10 minutes
        self.metadata_cache = LRUCache(max_size=200, ttl=1800)  # 30 minutes
        
        # Persistent cache for expensive operations
        self.persistent_cache_file = self.cache_dir / "tool_cache.pkl"
        self.persistent_cache: Dict[str, Any] = {}
        
        # Optimization data
        self.tool_performance: Dict[str, Dict[str, Any]] = {}
        self.usage_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load persistent cache
        self._load_persistent_cache()
        
        # Start cleanup task
        self._cleanup_task = None
        self._running = False
    
    async def start(self) -> None:
        """Start the cache system"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Tool cache system started")
    
    async def stop(self) -> None:
        """Stop the cache system"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._save_persistent_cache()
        logger.info("Tool cache system stopped")
    
    def get_cached_result(self, tool_id: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached tool result"""
        cache_key = self._generate_cache_key(tool_id, parameters)
        return self.result_cache.get(cache_key)
    
    def cache_result(self, tool_id: str, parameters: Dict[str, Any], 
                   result: Any, ttl: Optional[int] = None) -> None:
        """Cache tool result"""
        cache_key = self._generate_cache_key(tool_id, parameters)
        
        # Determine TTL based on tool performance
        if ttl is None:
            ttl = self._calculate_adaptive_ttl(tool_id)
        
        self.result_cache.put(cache_key, result, ttl)
        
        # Update usage pattern
        self._update_usage_pattern(tool_id, parameters, True)
    
    def get_cached_parameters(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get cached tool parameters"""
        return self.parameter_cache.get(tool_id)
    
    def cache_parameters(self, tool_id: str, parameters: Dict[str, Any]) -> None:
        """Cache tool parameters"""
        self.parameter_cache.put(tool_id, parameters)
    
    def get_cached_metadata(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get cached tool metadata"""
        return self.metadata_cache.get(tool_id)
    
    def cache_metadata(self, tool_id: str, metadata: Dict[str, Any]) -> None:
        """Cache tool metadata"""
        self.metadata_cache.put(tool_id, metadata)
    
    def get_persistent_cached(self, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        return self.persistent_cache.get(key)
    
    def cache_persistent(self, key: str, value: Any) -> None:
        """Cache value in persistent cache"""
        self.persistent_cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def _generate_cache_key(self, tool_id: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for tool and parameters"""
        # Create deterministic key from tool_id and parameters
        key_data = {
            'tool_id': tool_id,
            'parameters': sorted(parameters.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _calculate_adaptive_ttl(self, tool_id: str) -> int:
        """Calculate adaptive TTL based on tool performance"""
        if tool_id not in self.tool_performance:
            return 300  # Default 5 minutes
        
        perf = self.tool_performance[tool_id]
        
        # Base TTL on execution time and success rate
        base_ttl = 300  # 5 minutes
        
        # Longer TTL for fast, reliable tools
        if perf.get('avg_execution_time', 0) < 1.0 and perf.get('success_rate', 0) > 95:
            base_ttl = 600  # 10 minutes
        
        # Shorter TTL for slow or unreliable tools
        elif perf.get('avg_execution_time', 0) > 5.0 or perf.get('success_rate', 0) < 80:
            base_ttl = 120  # 2 minutes
        
        return base_ttl
    
    def _update_usage_pattern(self, tool_id: str, parameters: Dict[str, Any], 
                           success: bool) -> None:
        """Update usage pattern for tool"""
        if tool_id not in self.usage_patterns:
            self.usage_patterns[tool_id] = []
        
        pattern = {
            'timestamp': time.time(),
            'parameters': parameters,
            'success': success,
            'param_hash': self._generate_cache_key(tool_id, parameters)
        }
        
        self.usage_patterns[tool_id].append(pattern)
        
        # Keep only recent patterns (last 100)
        if len(self.usage_patterns[tool_id]) > 100:
            self.usage_patterns[tool_id] = self.usage_patterns[tool_id][-100:]
    
    def update_tool_performance(self, tool_id: str, execution_time: float, 
                             success: bool) -> None:
        """Update tool performance data"""
        if tool_id not in self.tool_performance:
            self.tool_performance[tool_id] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'success_rate': 0.0,
                'last_updated': time.time()
            }
        
        perf = self.tool_performance[tool_id]
        perf['total_executions'] += 1
        perf['total_execution_time'] += execution_time
        perf['avg_execution_time'] = perf['total_execution_time'] / perf['total_executions']
        
        if success:
            perf['successful_executions'] += 1
        
        perf['success_rate'] = (perf['successful_executions'] / perf['total_executions']) * 100
        perf['last_updated'] = time.time()
    
    def get_optimization_recommendations(self, tool_id: str) -> List[str]:
        """Get optimization recommendations for a tool"""
        recommendations = []
        
        if tool_id not in self.tool_performance:
            return recommendations
        
        perf = self.tool_performance[tool_id]
        
        # Performance-based recommendations
        if perf['avg_execution_time'] > 3.0:
            recommendations.append(
                f"High average execution time ({perf['avg_execution_time']:.2f}s). "
                "Consider caching results or optimizing the tool."
            )
        
        if perf['success_rate'] < 90:
            recommendations.append(
                f"Low success rate ({perf['success_rate']:.1f}%). "
                "Review error handling and input validation."
            )
        
        # Usage pattern recommendations
        if tool_id in self.usage_patterns:
            patterns = self.usage_patterns[tool_id]
            if len(patterns) > 10:
                # Check for frequently used parameter combinations
                param_counts = {}
                for pattern in patterns:
                    param_hash = pattern['param_hash']
                    param_counts[param_hash] = param_counts.get(param_hash, 0) + 1
                
                most_common = max(param_counts.items(), key=lambda x: x[1])
                if most_common[1] > 5:
                    recommendations.append(
                        f"Parameter combination used {most_common[1]} times. "
                        "Consider pre-computing or optimizing this common case."
                    )
        
        return recommendations
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'result_cache': self.result_cache.get_stats().__dict__,
            'parameter_cache': self.parameter_cache.get_stats().__dict__,
            'metadata_cache': self.metadata_cache.get_stats().__dict__,
            'persistent_cache_size': len(self.persistent_cache),
            'tool_performance_count': len(self.tool_performance),
            'usage_patterns_count': len(self.usage_patterns),
            'total_memory_usage': self._calculate_memory_usage()
        }
    
    def _calculate_memory_usage(self) -> int:
        """Calculate total memory usage"""
        total = (
            self.result_cache.stats.total_size +
            self.parameter_cache.stats.total_size +
            self.metadata_cache.stats.total_size
        )
        
        # Add persistent cache size estimate
        try:
            total += len(pickle.dumps(self.persistent_cache))
        except Exception:
            pass
        
        return total
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await self._save_persistent_cache_async()
                await asyncio.sleep(60)  # Cleanup every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired cache entries"""
        # This is handled by individual cache implementations
        # but we can add additional cleanup logic here
        pass
    
    async def _save_persistent_cache_async(self) -> None:
        """Save persistent cache asynchronously"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_persistent_cache
            )
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
    
    def _save_persistent_cache(self) -> None:
        """Save persistent cache to disk"""
        try:
            with open(self.persistent_cache_file, 'wb') as f:
                pickle.dump(self.persistent_cache, f)
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk"""
        try:
            if self.persistent_cache_file.exists():
                with open(self.persistent_cache_file, 'rb') as f:
                    self.persistent_cache = pickle.load(f)
                
                # Clean up old entries
                current_time = time.time()
                expired_keys = []
                
                for key, data in self.persistent_cache.items():
                    if isinstance(data, dict) and 'timestamp' in data:
                        if current_time - data['timestamp'] > 86400:  # 24 hours
                            expired_keys.append(key)
                
                for key in expired_keys:
                    del self.persistent_cache[key]
                
                logger.debug(f"Loaded persistent cache with {len(self.persistent_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {e}")
            self.persistent_cache = {}
    
    def clear_all_caches(self) -> None:
        """Clear all caches"""
        self.result_cache.clear()
        self.parameter_cache.clear()
        self.metadata_cache.clear()
        self.persistent_cache.clear()
        logger.info("All caches cleared")
    
    def export_cache_data(self) -> Dict[str, Any]:
        """Export cache data for analysis"""
        return {
            'cache_stats': self.get_cache_stats(),
            'tool_performance': self.tool_performance,
            'usage_patterns': {
                tool_id: patterns[-10:]  # Last 10 patterns per tool
                for tool_id, patterns in self.usage_patterns.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }


# Global cache system instance
tool_cache = ToolCacheSystem()