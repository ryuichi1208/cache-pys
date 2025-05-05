import time
import os
import pytest
from cache_pys.cache import CacheFactory, CacheConfig, FixedTTL, SlidingTTL, LRU, LFU, HybridTTL


def test_fixed_ttl():
    """Test fixed TTL strategy using environment variables."""
    # Get TTL from environment
    ttl = CacheConfig.get_default_ttl()

    # Create cache
    cache = CacheFactory.create_fixed_ttl_cache()

    # Set a value
    cache.set("key1", "value1")

    # Value should be available immediately
    assert cache.get("key1") == "value1"

    # Wait for half of TTL
    time.sleep(ttl / 2)

    # Value should still be available
    assert cache.get("key1") == "value1"

    # Only run full expiration test if TTL is small enough for testing
    if ttl <= 10:
        # Wait for the remaining TTL plus a small buffer
        time.sleep((ttl / 2) + 1)

        # Value should be expired
        assert cache.get("key1") is None


def test_sliding_ttl():
    """Test sliding TTL strategy using environment variables."""
    # Get TTL from environment
    ttl = CacheConfig.get_default_ttl()

    # Create cache
    cache = CacheFactory.create_sliding_ttl_cache()

    # Set a value
    cache.set("key1", "value1")

    # Wait for half of TTL
    if ttl <= 10:
        time.sleep(ttl / 2)

        # Access the value to reset the timer
        assert cache.get("key1") == "value1"

        # Wait for half of TTL again
        time.sleep(ttl / 2)

        # Value should still be available because the timer was reset
        assert cache.get("key1") == "value1"

        # Wait for full TTL plus a small buffer
        time.sleep(ttl + 1)

        # Value should be expired
        assert cache.get("key1") is None


def test_lru():
    """Test LRU strategy using environment variables."""
    # Get max size from environment
    max_size = CacheConfig.get_max_size()

    # Create cache
    cache = CacheFactory.create_lru_cache()

    # Fill the cache
    for i in range(max_size):
        cache.set(f"key{i}", f"value{i}")

    # All values should be available
    for i in range(max_size):
        assert cache.get(f"key{i}") == f"value{i}"

    # Access the first item to make it most recently used
    assert cache.get("key0") == "value0"

    # Add one more item, which should evict the least recently used item (key1)
    cache.set("key_new", "value_new")

    # key0 should still be available because it was recently accessed
    assert cache.get("key0") == "value0"

    # key1 should be evicted (assuming max_size > 1)
    if max_size > 1:
        assert cache.get("key1") is None

    # new key should be available
    assert cache.get("key_new") == "value_new"


def test_lfu():
    """Test LFU strategy using environment variables."""
    # Get max size from environment
    max_size = CacheConfig.get_max_size()

    # Create cache
    cache = CacheFactory.create_lfu_cache()

    # Fill the cache
    for i in range(max_size):
        cache.set(f"key{i}", f"value{i}")

    # Access key0 multiple times to increase its frequency
    for _ in range(3):
        cache.get("key0")

    # Add one more item, which should evict the least frequently used item
    cache.set("key_new", "value_new")

    # key0 should still be available because it was frequently accessed
    assert cache.get("key0") == "value0"

    # new key should be available
    assert cache.get("key_new") == "value_new"


def test_hybrid():
    """Test hybrid strategy using environment variables."""
    # Get parameters from environment
    ttl = CacheConfig.get_default_ttl()
    max_size = CacheConfig.get_max_size()

    # Create cache
    cache = CacheFactory.create_hybrid_cache()

    # Set values
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Access key1 multiple times to increase its frequency and recency
    for _ in range(3):
        cache.get("key1")

    # Both values should be available
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"

    # Only run expiration test if TTL is small enough for testing
    if ttl <= 10:
        # Wait for TTL plus a small buffer
        time.sleep(ttl + 1)

        # Both values should be expired
        assert cache.get("key1") is None
        assert cache.get("key2") is None


def test_environment_variables():
    """Test reading configuration from environment variables."""
    # Save original environment
    original_ttl = os.environ.get("CACHE_DEFAULT_TTL")
    original_strategy = os.environ.get("CACHE_STRATEGY")

    try:
        # Set environment variables
        os.environ["CACHE_DEFAULT_TTL"] = "60"
        os.environ["CACHE_STRATEGY"] = "fixed"

        # Get values
        ttl = CacheConfig.get_default_ttl()
        strategy = CacheConfig.get_strategy()

        # Check values
        assert ttl == 60
        assert strategy == "fixed"

        # Create cache from environment
        cache = CacheFactory.create_cache_from_env()

        # Check that the cache uses the correct strategy
        assert isinstance(cache.ttl_strategy, FixedTTL)
        assert cache.ttl_strategy.ttl == 60

        # Change strategy
        os.environ["CACHE_STRATEGY"] = "sliding"

        # Create a new cache
        cache = CacheFactory.create_cache_from_env()

        # Check that the new cache uses the correct strategy
        assert isinstance(cache.ttl_strategy, SlidingTTL)

    finally:
        # Restore original environment
        if original_ttl is not None:
            os.environ["CACHE_DEFAULT_TTL"] = original_ttl
        else:
            os.environ.pop("CACHE_DEFAULT_TTL", None)

        if original_strategy is not None:
            os.environ["CACHE_STRATEGY"] = original_strategy
        else:
            os.environ.pop("CACHE_STRATEGY", None)


def test_cached_decorator():
    """Test cached decorator with environment variables."""
    from cache_pys.cache import cached

    # Save original environment
    original_ttl = os.environ.get("CACHE_DEFAULT_TTL")

    try:
        # Set small TTL for testing
        os.environ["CACHE_DEFAULT_TTL"] = "2"

        # Function with cache
        count = 0

        @cached()
        def test_function():
            nonlocal count
            count += 1
            return count

        # First call should execute the function
        assert test_function() == 1

        # Second call should use the cached result
        assert test_function() == 1

        # Wait for TTL to expire
        time.sleep(3)

        # After TTL, the function should execute again
        assert test_function() == 2

    finally:
        # Restore original environment
        if original_ttl is not None:
            os.environ["CACHE_DEFAULT_TTL"] = original_ttl
        else:
            os.environ.pop("CACHE_DEFAULT_TTL", None)
