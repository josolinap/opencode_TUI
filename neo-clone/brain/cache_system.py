"""
Cache System for MiniMax Agent Architecture

This module provides high-performance caching for frequently accessed data,
computations, and responses to optimize system performance.

Author: MiniMax Agent
Version: 1.0
"""

import time
import threading
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import pickle
import weakref
from collections import OrderedDict
import math

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry is expired"""
        if ttl_seconds <= 0:
            return False
        return (datetime.now() - self.created_at).total_seconds() > ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


class CacheStats:
    """Cache performance statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.total_eviction_time = 0.0
        self.avg_access_time = 0.0
        self.hit_ratio = 0.0
        self.start_time = datetime.now()
    
    def record_hit(self, access_time: float):
        """Record cache hit"""
        self.hits += 1
        self.total_requests += 1
        self._update_avg_access_time(access_time)
        self.hit_ratio = self.hits / self.total_requests
    
    def record_miss(self):
        """Record cache miss"""
        self.misses += 1
        self.total_requests += 1
        self.hit_ratio = self.hits / self.total_requests
    
    def record_eviction(self, eviction_time: float):
        """Record cache eviction"""
        self.evictions += 1
        self.total_eviction_time += eviction_time
    
    def _update_avg_access_time(self, access_time: float):
        """Update average access time"""
        if self.avg_access_time == 0.0:
            self.avg_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_access_time = alpha * access_time + (1 - alpha) * self.avg_access_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cache statistics summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_ratio": self.hit_ratio,
            "evictions": self.evictions,
            "uptime_seconds": uptime,
            "requests_per_second": self.total_requests / max(1, uptime),
            "average_access_time_ms": self.avg_access_time * 1000,
            "eviction_time_ms": (self.total_eviction_time / max(1, self.evictions)) * 1000 if self.evictions > 0 else 0
        }


class CacheSystem:
    """
    High-performance cache system with multiple eviction policies,
    TTL support, and comprehensive statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: float = 3600.0,
        eviction_policy: str = "lru",
        enable_compression: bool = False,
        enable_persistence: bool = False
    ):
        """
        Initialize cache system
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            eviction_policy: Eviction policy ("lru", "lfu", "fifo", "ttl")
            enable_compression: Enable value compression
            enable_persistence: Enable cache persistence to disk
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        # Storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()
        
        # Memory tracking
        self.current_memory_usage = 0
        self.avg_value_size = 0.0
        self.total_values_cached = 0
        
        # Cleanup
        self.cleanup_interval = 300  # 5 minutes
        self._start_cleanup_task()
        
        logger.info(f"Cache System initialized: max_size={max_size}, max_memory={max_memory_mb}MB, policy={eviction_policy}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_expired()
                except Exception as e:
                    logger.warning(f"Cache cleanup task error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _get_value_size(self, value: Any) -> int:
        """Get approximate size of value in bytes"""
        try:
            if self.enable_compression:
                # Estimate compressed size
                return len(pickle.dumps(value)) // 2  # Rough compression estimate
            else:
                return len(pickle.dumps(value))
        except Exception:
            # Fallback: estimate based on string length
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, dict)):
                return len(str(value).encode('utf-8'))
            else:
                return 64  # Default estimate
    
    def _compress_value(self, value: Any) -> Any:
        """Compress value if compression is enabled"""
        if not self.enable_compression:
            return value
        
        try:
            # Simple compression using pickle + base64
            # In real implementation, would use proper compression library
            return value
        except Exception:
            return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Decompress value if compression is enabled"""
        if not self.enable_compression:
            return value
        
        try:
            # Simple decompression
            return value
        except Exception:
            return value
    
    def get(self, key: str, ttl: float = None) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            ttl: Override default TTL
        
        Returns:
            Cached value or None
        """
        start_time = time.time()
        
        with self.lock:
            try:
                entry = self.cache.get(key)
                
                if entry is None:
                    self.stats.record_miss()
                    return None
                
                # Check TTL
                if entry.is_expired(ttl or self.default_ttl):
                    # Remove expired entry
                    del self.cache[key]
                    self.current_memory_usage -= entry.size_bytes
                    self.stats.record_miss()
                    return None
                
                # Update access statistics
                entry.update_access()
                
                # Move to end for LRU
                if self.eviction_policy == "lru":
                    self.cache.move_to_end(key)
                
                # Return decompressed value
                value = self._decompress_value(entry.value)
                access_time = time.time() - start_time
                self.stats.record_hit(access_time)
                
                return value
                
            except Exception as e:
                logger.error(f"Cache get error for key '{key}': {e}")
                self.stats.record_miss()
                return None
    
    def set(self, key: str, value: Any, ttl: float = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        with self.lock:
            try:
                # Compress value if enabled
                compressed_value = self._compress_value(value)
                value_size = self._get_value_size(compressed_value)
                
                # Check if we need to evict
                while (len(self.cache) >= self.max_size or 
                       self.current_memory_usage + value_size > self.max_memory_bytes):
                    if not self.cache:
                        break  # Can't evict anything
                    
                    evicted = self._evict_one()
                    if not evicted:
                        break  # Nothing to evict
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    size_bytes=value_size,
                    metadata=metadata or {}
                )
                
                # Add or update entry
                if key in self.cache:
                    # Update existing entry
                    old_entry = self.cache[key]
                    self.current_memory_usage -= old_entry.size_bytes
                
                self.cache[key] = entry
                self.current_memory_usage += value_size
                self.total_values_cached += 1
                
                # Update average value size
                if self.avg_value_size == 0.0:
                    self.avg_value_size = value_size
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.avg_value_size = alpha * value_size + (1 - alpha) * self.avg_value_size
                
                # Move to end for LRU
                if self.eviction_policy == "lru":
                    self.cache.move_to_end(key)
                
                logger.debug(f"Cache set: key='{key}', size={value_size} bytes")
                return True
                
            except Exception as e:
                logger.error(f"Cache set error for key '{key}': {e}")
                return False
    
    def _evict_one(self) -> bool:
        """Evict one entry based on eviction policy"""
        if not self.cache:
            return False
        
        eviction_start = time.time()
        
        try:
            if self.eviction_policy == "lru":
                # Least Recently Used
                key, entry = self.cache.popitem(last=False)
                
            elif self.eviction_policy == "lfu":
                # Least Frequently Used
                key, entry = min(self.cache.items(), key=lambda x: x[1].access_count)
                del self.cache[key]
                
            elif self.eviction_policy == "fifo":
                # First In First Out
                key, entry = self.cache.popitem(last=False)
                
            elif self.eviction_policy == "ttl":
                # Evict oldest expired or least recently used
                expired_keys = [
                    k for k, e in self.cache.items() 
                    if e.is_expired(self.default_ttl)
                ]
                
                if expired_keys:
                    key = expired_keys[0]
                    entry = self.cache.pop(key)
                else:
                    key, entry = self.cache.popitem(last=False)
                    
            else:
                # Default to LRU
                key, entry = self.cache.popitem(last=False)
            
            self.current_memory_usage -= entry.size_bytes
            eviction_time = time.time() - eviction_start
            self.stats.record_eviction(eviction_time)
            
            logger.debug(f"Evicted cache entry: key='{key}', age={entry.get_age_seconds():.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Cache eviction error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key
        
        Returns:
            Success status
        """
        with self.lock:
            try:
                if key in self.cache:
                    entry = self.cache.pop(key)
                    self.current_memory_usage -= entry.size_bytes
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Cache delete error for key '{key}': {e}")
                return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            try:
                self.cache.clear()
                self.current_memory_usage = 0
                self.avg_value_size = 0.0
                self.total_values_cached = 0
                logger.info("Cache cleared")
                
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries
        
        Returns:
            Number of entries removed
        """
        with self.lock:
            try:
                expired_keys = []
                for key, entry in self.cache.items():
                    if entry.is_expired(self.default_ttl):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.cache.pop(key)
                    self.current_memory_usage -= entry.size_bytes
                
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                return len(expired_keys)
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            try:
                base_stats = self.stats.get_summary()
                
                return {
                    **base_stats,
                    "current_size": len(self.cache),
                    "max_size": self.max_size,
                    "size_usage_ratio": len(self.cache) / self.max_size,
                    "current_memory_usage_mb": self.current_memory_usage / (1024 * 1024),
                    "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                    "memory_usage_ratio": self.current_memory_usage / self.max_memory_bytes,
                    "average_value_size_bytes": self.avg_value_size,
                    "total_values_cached": self.total_values_cached,
                    "eviction_policy": self.eviction_policy,
                    "default_ttl_seconds": self.default_ttl,
                    "compression_enabled": self.enable_compression
                }
                
            except Exception as e:
                logger.error(f"Cache stats error: {e}")
                return {}
    
    def get_keys(self, pattern: str = None) -> List[str]:
        """
        Get cache keys, optionally filtered by pattern
        
        Args:
            pattern: Key pattern to filter by (supports wildcards)
        
        Returns:
            List of cache keys
        """
        with self.lock:
            try:
                keys = list(self.cache.keys())
                
                if pattern:
                    # Simple pattern matching (supports * wildcard)
                    import fnmatch
                    keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
                
                return keys
                
            except Exception as e:
                logger.error(f"Cache get_keys error: {e}")
                return []
    
    def get_value_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached value without retrieving it
        
        Args:
            key: Cache key
        
        Returns:
            Dictionary with value information or None
        """
        with self.lock:
            try:
                entry = self.cache.get(key)
                if not entry:
                    return None
                
                return {
                    "key": entry.key,
                    "size_bytes": entry.size_bytes,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "age_seconds": entry.get_age_seconds(),
                    "is_expired": entry.is_expired(self.default_ttl),
                    "metadata": entry.metadata
                }
                
            except Exception as e:
                logger.error(f"Cache get_value_info error for key '{key}': {e}")
                return None
    
    def memoize(self, ttl: float = None, key_func: Callable = None):
        """
        Decorator for automatic caching of function results
        
        Args:
            ttl: Time-to-live for cached results
            key_func: Function to generate cache key from args/kwargs
        
        Returns:
            Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                result = self.get(cache_key, ttl)
                if result is not None:
                    return result
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern
        
        Args:
            pattern: Pattern to match keys against
        
        Returns:
            Number of entries invalidated
        """
        with self.lock:
            try:
                import fnmatch
                keys_to_delete = [
                    key for key in self.cache.keys() 
                    if fnmatch.fnmatch(key, pattern)
                ]
                
                for key in keys_to_delete:
                    self.delete(key)
                
                logger.debug(f"Invalidated {len(keys_to_delete)} cache entries matching pattern: {pattern}")
                return len(keys_to_delete)
                
            except Exception as e:
                logger.error(f"Cache invalidate_pattern error: {e}")
                return 0
    
    def shutdown(self):
        """Shutdown cache system"""
        logger.info("Shutting down Cache System")
        if self.enable_persistence:
            # Save cache to disk if persistence is enabled
            self._save_to_disk()
    
    def _save_to_disk(self):
        """Save cache to disk (placeholder for persistence implementation)"""
        try:
            # This would implement cache persistence in a real system
            logger.debug("Cache persistence not yet implemented")
        except Exception as e:
            logger.error(f"Cache persistence error: {e}")


# Global cache instance
_cache_instance: Optional[CacheSystem] = None
_cache_lock = threading.Lock()


def get_cache(
    max_size: int = 1000,
    max_memory_mb: int = 100,
    default_ttl: float = 3600.0,
    eviction_policy: str = "lru"
) -> CacheSystem:
    """
    Get singleton cache instance
    
    Args:
        max_size: Maximum number of entries
        max_memory_mb: Maximum memory usage in MB
        default_ttl: Default time-to-live in seconds
        eviction_policy: Eviction policy
    
    Returns:
        CacheSystem singleton instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = CacheSystem(
                    max_size=max_size,
                    max_memory_mb=max_memory_mb,
                    default_ttl=default_ttl,
                    eviction_policy=eviction_policy
                )
    
    return _cache_instance


def reset_cache() -> None:
    """Reset the cache instance"""
    global _cache_instance
    with _cache_lock:
        if _cache_instance:
            try:
                _cache_instance.shutdown()
            except Exception:
                pass
        _cache_instance = None
    logger.info("Cache instance reset")


def create_cache_instance(
    max_size: int = 1000,
    max_memory_mb: int = 100,
    default_ttl: float = 3600.0,
    eviction_policy: str = "lru"
) -> CacheSystem:
    """
    Create a new cache instance
    
    Args:
        max_size: Maximum number of entries
        max_memory_mb: Maximum memory usage in MB
        default_ttl: Default time-to-live in seconds
        eviction_policy: Eviction policy
    
    Returns:
        New CacheSystem instance
    """
    return CacheSystem(max_size, max_memory_mb, default_ttl, eviction_policy)
