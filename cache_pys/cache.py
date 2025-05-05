import time
import threading
import pickle
import os
from typing import Any, Dict, Optional, Union, Callable, Tuple, List
from abc import ABC, abstractmethod


class TTLStrategy(ABC):
    """
    Abstract base class for TTL strategies.
    All TTL strategies should implement these methods.
    """

    @abstractmethod
    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """
        Check if an item is expired according to this strategy.

        Args:
            key: The cache key
            creation_time: When the item was created/updated
            access_data: Additional data about the item (access count, last access time, etc.)

        Returns:
            bool: True if the item is expired, False otherwise
        """
        pass

    @abstractmethod
    def on_access(self, key: str, access_data: Dict) -> None:
        """
        Called when an item is accessed.

        Args:
            key: The cache key
            access_data: Additional data about the item to update
        """
        pass

    @abstractmethod
    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """
        Get a list of keys that are candidates for eviction.

        Args:
            cache_data: Dictionary mapping keys to their metadata

        Returns:
            List of keys that could be evicted, in order of priority
        """
        pass


class CacheConfig:
    """
    Configuration class for cache settings that can be loaded from environment variables.
    """

    @staticmethod
    def get_env_int(name: str, default: int) -> int:
        """
        Get an integer value from environment variable.

        Args:
            name: Environment variable name
            default: Default value if environment variable is not set or invalid

        Returns:
            int: Value from environment variable or default
        """
        try:
            value = os.environ.get(name)
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def get_env_float(name: str, default: float) -> float:
        """
        Get a float value from environment variable.

        Args:
            name: Environment variable name
            default: Default value if environment variable is not set or invalid

        Returns:
            float: Value from environment variable or default
        """
        try:
            value = os.environ.get(name)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def get_env_str(name: str, default: str) -> str:
        """
        Get a string value from environment variable.

        Args:
            name: Environment variable name
            default: Default value if environment variable is not set

        Returns:
            str: Value from environment variable or default
        """
        return os.environ.get(name, default)

    @classmethod
    def get_default_ttl(cls) -> int:
        """
        Get default TTL from environment or use fallback value.

        Returns:
            int: Default TTL in seconds
        """
        return cls.get_env_int("CACHE_DEFAULT_TTL", 3600)  # Default: 1 hour

    @classmethod
    def get_cleanup_interval(cls) -> int:
        """
        Get cleanup interval from environment or use fallback value.

        Returns:
            int: Cleanup interval in seconds
        """
        return cls.get_env_int("CACHE_CLEANUP_INTERVAL", 60)  # Default: 1 minute

    @classmethod
    def get_max_size(cls) -> int:
        """
        Get maximum cache size from environment or use fallback value.

        Returns:
            int: Maximum number of items in cache
        """
        return cls.get_env_int("CACHE_MAX_SIZE", 100)  # Default: 100 items

    @classmethod
    def get_strategy(cls) -> str:
        """
        Get default cache strategy from environment or use fallback value.

        Returns:
            str: Cache strategy name
        """
        return cls.get_env_str("CACHE_STRATEGY", "fixed")  # Default: fixed TTL

    @classmethod
    def get_weight_recency(cls) -> float:
        """
        Get recency weight for hybrid cache from environment or use fallback value.

        Returns:
            float: Weight for recency (0-1)
        """
        return cls.get_env_float("CACHE_WEIGHT_RECENCY", 0.5)  # Default: 50%

    @classmethod
    def get_weight_frequency(cls) -> float:
        """
        Get frequency weight for hybrid cache from environment or use fallback value.

        Returns:
            float: Weight for frequency (0-1)
        """
        return cls.get_env_float("CACHE_WEIGHT_FREQUENCY", 0.5)  # Default: 50%


class FixedTTL(TTLStrategy):
    """
    Standard fixed TTL strategy. Items expire after a fixed time period.
    """

    def __init__(self, ttl_seconds: Optional[int] = None):
        """
        Initialize with a fixed TTL.

        Args:
            ttl_seconds: Time-to-live in seconds (None to use environment variable)
        """
        self.ttl = ttl_seconds if ttl_seconds is not None else CacheConfig.get_default_ttl()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """Check if an item has exceeded its fixed TTL."""
        return time.time() > (creation_time + self.ttl)

    def on_access(self, key: str, access_data: Dict) -> None:
        """Update access count on item access."""
        access_data["access_count"] = access_data.get("access_count", 0) + 1
        access_data["last_access"] = time.time()

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """Return keys in order of oldest creation time."""
        return sorted(cache_data.keys(), key=lambda k: cache_data[k]["creation_time"])


class SlidingTTL(TTLStrategy):
    """
    Sliding window TTL strategy. The TTL is reset each time the item is accessed.
    """

    def __init__(self, ttl_seconds: Optional[int] = None):
        """
        Initialize with a sliding TTL.

        Args:
            ttl_seconds: Time-to-live in seconds from last access (None to use environment variable)
        """
        self.ttl = ttl_seconds if ttl_seconds is not None else CacheConfig.get_default_ttl()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """Check if an item has not been accessed within the TTL window."""
        last_access = access_data.get("last_access", creation_time)
        return time.time() > (last_access + self.ttl)

    def on_access(self, key: str, access_data: Dict) -> None:
        """Reset the last access time on item access."""
        access_data["access_count"] = access_data.get("access_count", 0) + 1
        access_data["last_access"] = time.time()

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """Return keys in order of least recently accessed."""
        return sorted(cache_data.keys(), key=lambda k: cache_data[k]["access_data"].get("last_access", 0))


class LRU(TTLStrategy):
    """
    Least Recently Used strategy. Items that haven't been accessed recently are evicted first.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize LRU strategy.

        Args:
            max_size: Maximum number of items to keep in cache (None to use environment variable)
        """
        self.max_size = max_size if max_size is not None else CacheConfig.get_max_size()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """LRU items never expire based on time."""
        return False

    def on_access(self, key: str, access_data: Dict) -> None:
        """Update last access time on item access."""
        access_data["last_access"] = time.time()

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """Return keys in order of least recently used."""
        if len(cache_data) <= self.max_size:
            return []

        return sorted(cache_data.keys(), key=lambda k: cache_data[k]["access_data"].get("last_access", 0))


class LFU(TTLStrategy):
    """
    Least Frequently Used strategy. Items that are used least frequently are evicted first.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize LFU strategy.

        Args:
            max_size: Maximum number of items to keep in cache (None to use environment variable)
        """
        self.max_size = max_size if max_size is not None else CacheConfig.get_max_size()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """LFU items never expire based on time."""
        return False

    def on_access(self, key: str, access_data: Dict) -> None:
        """Increment access count on item access."""
        access_data["access_count"] = access_data.get("access_count", 0) + 1

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """Return keys in order of least frequently used."""
        if len(cache_data) <= self.max_size:
            return []

        return sorted(cache_data.keys(), key=lambda k: cache_data[k]["access_data"].get("access_count", 0))


class FIFO(TTLStrategy):
    """
    First In, First Out strategy. Oldest items are evicted first.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize FIFO strategy.

        Args:
            max_size: Maximum number of items to keep in cache (None to use environment variable)
        """
        self.max_size = max_size if max_size is not None else CacheConfig.get_max_size()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """FIFO items never expire based on time."""
        return False

    def on_access(self, key: str, access_data: Dict) -> None:
        """FIFO doesn't update anything on access."""
        pass

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """Return keys in order of insertion (oldest first)."""
        if len(cache_data) <= self.max_size:
            return []

        return sorted(cache_data.keys(), key=lambda k: cache_data[k]["creation_time"])


class RandomReplacement(TTLStrategy):
    """
    Random Replacement strategy. Random items are evicted when needed.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize Random Replacement strategy.

        Args:
            max_size: Maximum number of items to keep in cache (None to use environment variable)
        """
        self.max_size = max_size if max_size is not None else CacheConfig.get_max_size()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """Random items never expire based on time."""
        return False

    def on_access(self, key: str, access_data: Dict) -> None:
        """Random replacement doesn't update anything on access."""
        pass

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """Return keys in random order."""
        import random

        if len(cache_data) <= self.max_size:
            return []

        keys = list(cache_data.keys())
        random.shuffle(keys)
        return keys


class HybridTTL(TTLStrategy):
    """
    Hybrid TTL strategy that combines time-based expiration with usage-based eviction.
    """

    def __init__(
        self,
        ttl_seconds: Optional[int] = None,
        max_size: Optional[int] = None,
        weight_recency: Optional[float] = None,
        weight_frequency: Optional[float] = None,
    ):
        """
        Initialize Hybrid TTL strategy.

        Args:
            ttl_seconds: Time-to-live in seconds (None to use environment variable)
            max_size: Maximum number of items to keep in cache (None to use environment variable)
            weight_recency: Weight given to recency (0-1, None to use environment variable)
            weight_frequency: Weight given to frequency (0-1, None to use environment variable)
        """
        self.ttl = ttl_seconds if ttl_seconds is not None else CacheConfig.get_default_ttl()
        self.max_size = max_size if max_size is not None else CacheConfig.get_max_size()
        self.weight_recency = weight_recency if weight_recency is not None else CacheConfig.get_weight_recency()
        self.weight_frequency = weight_frequency if weight_frequency is not None else CacheConfig.get_weight_frequency()

    def is_expired(self, key: str, creation_time: float, access_data: Dict) -> bool:
        """Check if an item has exceeded its TTL."""
        return time.time() > (creation_time + self.ttl)

    def on_access(self, key: str, access_data: Dict) -> None:
        """Update both access count and last access time."""
        access_data["access_count"] = access_data.get("access_count", 0) + 1
        access_data["last_access"] = time.time()

    def get_candidates_for_eviction(self, cache_data: Dict[str, Dict]) -> List[str]:
        """
        Return keys using a hybrid scoring mechanism.
        Score combines recency and frequency metrics.
        """
        if len(cache_data) <= self.max_size:
            return []

        # Normalize access counts and recency
        now = time.time()
        max_count = max((cache_data[k]["access_data"].get("access_count", 0) for k in cache_data), default=1)

        # Calculate scores for each item (higher score = more likely to be evicted)
        scores = []
        for key, data in cache_data.items():
            access_data = data["access_data"]
            access_count = access_data.get("access_count", 0)
            last_access = access_data.get("last_access", data["creation_time"])

            # Normalize values (0-1 range, higher is worse)
            norm_frequency = 1 - (access_count / max_count if max_count > 0 else 0)
            norm_recency = (now - last_access) / self.ttl if self.ttl > 0 else 0

            # Calculate combined score
            score = self.weight_recency * norm_recency + self.weight_frequency * norm_frequency

            scores.append((key, score))

        # Sort by score (highest first - most likely to be evicted)
        return [k for k, _ in sorted(scores, key=lambda x: x[1], reverse=True)]


class CacheItem:
    """
    Represents an item stored in the cache with metadata.
    """

    def __init__(self, value: Any):
        """
        Initialize a cache item with a value.

        Args:
            value: The value to store in the cache
        """
        self.value = value
        self.creation_time = time.time()
        self.access_data = {}

    def is_expired(self, strategy: TTLStrategy) -> bool:
        """
        Check if the cache item has expired according to the given strategy.

        Args:
            strategy: TTL strategy to use for checking expiration

        Returns:
            bool: True if expired, False otherwise
        """
        return strategy.is_expired(None, self.creation_time, self.access_data)


class AdvancedCache:
    """
    Advanced cache implementation with support for different TTL strategies.
    """

    def __init__(self, ttl_strategy: Optional[TTLStrategy] = None, cleanup_interval: Optional[int] = None):
        """
        Initialize an advanced cache with custom TTL strategy and automatic cleanup.

        Args:
            ttl_strategy: TTL strategy to use (defaults to strategy from environment variables)
            cleanup_interval: Interval in seconds for automatic cleanup of expired items
                             (None to use environment variable)
        """
        self._cache: Dict[str, CacheItem] = {}
        self._lock = threading.RLock()

        # Use environment variables if not explicitly provided
        if ttl_strategy is None:
            strategy_name = CacheConfig.get_strategy()
            if strategy_name == "fixed":
                ttl_strategy = FixedTTL()
            elif strategy_name == "sliding":
                ttl_strategy = SlidingTTL()
            elif strategy_name == "lru":
                ttl_strategy = LRU()
            elif strategy_name == "lfu":
                ttl_strategy = LFU()
            elif strategy_name == "fifo":
                ttl_strategy = FIFO()
            elif strategy_name == "random":
                ttl_strategy = RandomReplacement()
            elif strategy_name == "hybrid":
                ttl_strategy = HybridTTL()
            else:
                # Default to FixedTTL if the strategy is unknown
                ttl_strategy = FixedTTL()

        self.ttl_strategy = ttl_strategy
        self.cleanup_interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start a background thread to clean up expired cache items."""
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        """Background loop to periodically clean up expired cache items."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            self.cleanup()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key: The key to look up
            default: Value to return if key is not found or expired

        Returns:
            The cached value or default if not found
        """
        with self._lock:
            item = self._cache.get(key)
            if item is None or item.is_expired(self.ttl_strategy):
                return default

            # Update access data
            self.ttl_strategy.on_access(key, item.access_data)
            return item.value

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            key: The key to store the value under
            value: The value to store
        """
        with self._lock:
            self._cache[key] = CacheItem(value)
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The key to delete

        Returns:
            bool: True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def cleanup(self) -> int:
        """
        Remove all expired items from the cache.

        Returns:
            int: Number of items removed
        """
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired(self.ttl_strategy)]
            for k in expired_keys:
                del self._cache[k]
            return len(expired_keys)

    def _evict_if_needed(self) -> int:
        """
        Evict items according to the TTL strategy if needed.

        Returns:
            int: Number of items evicted
        """
        cache_data = {
            k: {"creation_time": v.creation_time, "access_data": v.access_data} for k, v in self._cache.items()
        }

        candidates = self.ttl_strategy.get_candidates_for_eviction(cache_data)
        evicted = 0

        for key in candidates:
            if key in self._cache:
                del self._cache[key]
                evicted += 1

        return evicted

    def get_many(self, keys: list) -> Dict[str, Any]:
        """
        Get multiple values from the cache.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dict mapping keys to their cached values (only for found keys)
        """
        result = {}
        with self._lock:
            for key in keys:
                item = self._cache.get(key)
                if item is not None and not item.is_expired(self.ttl_strategy):
                    self.ttl_strategy.on_access(key, item.access_data)
                    result[key] = item.value
        return result

    def set_many(self, items: Dict[str, Any]) -> None:
        """
        Set multiple values in the cache at once.

        Args:
            items: Dict mapping keys to values
        """
        with self._lock:
            for key, value in items.items():
                self._cache[key] = CacheItem(value)
            self._evict_if_needed()

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, int]:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for item in self._cache.values() if item.is_expired(self.ttl_strategy))

            # Calculate average access count
            access_counts = [item.access_data.get("access_count", 0) for item in self._cache.values()]
            avg_access = sum(access_counts) / total if total > 0 else 0

            return {
                "total_items": total,
                "expired_items": expired,
                "active_items": total - expired,
                "avg_access_count": avg_access,
            }

    def save_to_file(self, filename: str) -> bool:
        """
        Save the cache to a file using pickle.

        Args:
            filename: Path to the file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, "wb") as f:
                pickle.dump(self._cache, f)
            return True
        except Exception:
            return False

    def load_from_file(self, filename: str) -> bool:
        """
        Load the cache from a file.

        Args:
            filename: Path to the file

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(filename):
            return False

        try:
            with open(filename, "rb") as f:
                self._cache = pickle.load(f)
            return True
        except Exception:
            return False

    def __del__(self):
        """Clean up resources when the cache is destroyed."""
        if hasattr(self, "_stop_cleanup"):
            self._stop_cleanup.set()
            if hasattr(self, "_cleanup_thread"):
                self._cleanup_thread.join(timeout=1.0)


class CacheFactory:
    """
    Factory class for creating different types of caches.
    """

    @staticmethod
    def create_cache_from_env() -> AdvancedCache:
        """
        Create a cache based on environment variables.

        Returns:
            AdvancedCache configured from environment variables
        """
        return AdvancedCache()

    @staticmethod
    def create_fixed_ttl_cache(
        ttl_seconds: Optional[int] = None, cleanup_interval: Optional[int] = None
    ) -> AdvancedCache:
        """
        Create a cache with fixed TTL.

        Args:
            ttl_seconds: Time-to-live in seconds (None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with FixedTTL strategy
        """
        ttl = ttl_seconds if ttl_seconds is not None else CacheConfig.get_default_ttl()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=FixedTTL(ttl), cleanup_interval=interval)

    @staticmethod
    def create_sliding_ttl_cache(
        ttl_seconds: Optional[int] = None, cleanup_interval: Optional[int] = None
    ) -> AdvancedCache:
        """
        Create a cache with sliding TTL.

        Args:
            ttl_seconds: Time-to-live in seconds from last access (None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with SlidingTTL strategy
        """
        ttl = ttl_seconds if ttl_seconds is not None else CacheConfig.get_default_ttl()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=SlidingTTL(ttl), cleanup_interval=interval)

    @staticmethod
    def create_lru_cache(max_size: Optional[int] = None, cleanup_interval: Optional[int] = None) -> AdvancedCache:
        """
        Create a cache with LRU eviction.

        Args:
            max_size: Maximum number of items to keep (None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with LRU strategy
        """
        size = max_size if max_size is not None else CacheConfig.get_max_size()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=LRU(size), cleanup_interval=interval)

    @staticmethod
    def create_lfu_cache(max_size: Optional[int] = None, cleanup_interval: Optional[int] = None) -> AdvancedCache:
        """
        Create a cache with LFU eviction.

        Args:
            max_size: Maximum number of items to keep (None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with LFU strategy
        """
        size = max_size if max_size is not None else CacheConfig.get_max_size()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=LFU(size), cleanup_interval=interval)

    @staticmethod
    def create_fifo_cache(max_size: Optional[int] = None, cleanup_interval: Optional[int] = None) -> AdvancedCache:
        """
        Create a cache with FIFO eviction.

        Args:
            max_size: Maximum number of items to keep (None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with FIFO strategy
        """
        size = max_size if max_size is not None else CacheConfig.get_max_size()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=FIFO(size), cleanup_interval=interval)

    @staticmethod
    def create_random_cache(max_size: Optional[int] = None, cleanup_interval: Optional[int] = None) -> AdvancedCache:
        """
        Create a cache with random eviction.

        Args:
            max_size: Maximum number of items to keep (None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with RandomReplacement strategy
        """
        size = max_size if max_size is not None else CacheConfig.get_max_size()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=RandomReplacement(size), cleanup_interval=interval)

    @staticmethod
    def create_hybrid_cache(
        ttl_seconds: Optional[int] = None,
        max_size: Optional[int] = None,
        weight_recency: Optional[float] = None,
        weight_frequency: Optional[float] = None,
        cleanup_interval: Optional[int] = None,
    ) -> AdvancedCache:
        """
        Create a cache with hybrid eviction.

        Args:
            ttl_seconds: Time-to-live in seconds (None to use environment variable)
            max_size: Maximum number of items to keep (None to use environment variable)
            weight_recency: Weight for recency (0-1, None to use environment variable)
            weight_frequency: Weight for frequency (0-1, None to use environment variable)
            cleanup_interval: Interval in seconds for cleanup (None to use environment variable)

        Returns:
            AdvancedCache instance with HybridTTL strategy
        """
        ttl = ttl_seconds if ttl_seconds is not None else CacheConfig.get_default_ttl()
        size = max_size if max_size is not None else CacheConfig.get_max_size()
        w_recency = weight_recency if weight_recency is not None else CacheConfig.get_weight_recency()
        w_frequency = weight_frequency if weight_frequency is not None else CacheConfig.get_weight_frequency()
        interval = cleanup_interval if cleanup_interval is not None else CacheConfig.get_cleanup_interval()

        return AdvancedCache(ttl_strategy=HybridTTL(ttl, size, w_recency, w_frequency), cleanup_interval=interval)


def cached(ttl: Optional[int] = None, strategy: str = "fixed"):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds, or None to use environment variable
        strategy: TTL strategy to use ('fixed', 'sliding')

    Returns:
        Decorated function
    """
    actual_ttl = ttl if ttl is not None else CacheConfig.get_default_ttl()
    cache_data = {}

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create a cache key from the function name and arguments
            key = str((func.__name__, args, frozenset(kwargs.items())))

            if strategy == "fixed":
                # Check if we have a cached result and it's not expired
                if key in cache_data:
                    timestamp, result = cache_data[key]
                    if actual_ttl is None or time.time() - timestamp < actual_ttl:
                        return result

                # Call the function and cache the result
                result = func(*args, **kwargs)
                cache_data[key] = (time.time(), result)
                return result

            elif strategy == "sliding":
                # With sliding expiration, we update the timestamp on each access
                if key in cache_data:
                    timestamp, result = cache_data[key]
                    if actual_ttl is None or time.time() - timestamp < actual_ttl:
                        # Update the timestamp
                        cache_data[key] = (time.time(), result)
                        return result

                # Call the function and cache the result
                result = func(*args, **kwargs)
                cache_data[key] = (time.time(), result)
                return result

            else:
                # Fallback to no caching
                return func(*args, **kwargs)

        # Add a method to clear this function's cache
        wrapper.clear_cache = lambda: cache_data.clear()
        return wrapper

    return decorator
