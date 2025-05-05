import os
import time
import cache_pys.cli
from cache_pys.attr.client import Client
from cache_pys.cache import AdvancedCache, cached, CacheFactory, CacheConfig


def main():
    print("=== Environment Variable Configuration ===")
    print(f"Default TTL: {CacheConfig.get_default_ttl()} seconds")
    print(f"Cleanup Interval: {CacheConfig.get_cleanup_interval()} seconds")
    print(f"Max Size: {CacheConfig.get_max_size()} items")
    print(f"Strategy: {CacheConfig.get_strategy()}")

    print("\n=== Setting Environment Variables ===")
    os.environ["CACHE_DEFAULT_TTL"] = "10"  # Set TTL to 10 seconds
    os.environ["CACHE_MAX_SIZE"] = "5"  # Set max size to 5 items

    print(f"Updated Default TTL: {CacheConfig.get_default_ttl()} seconds")
    print(f"Updated Max Size: {CacheConfig.get_max_size()} items")

    print("\n=== Basic Cache Example ===")
    basic_cache = cache_pys.cli.Cache()
    basic_cache.set("key1", "value1")
    print(f"Basic cache get: {basic_cache.get('key1')}")

    print("\n=== Client Class Example ===")
    client = Client(name="Test", age=30)
    print(client)

    print("\n=== Cache from Environment Variables ===")
    # This cache will use settings from environment variables
    env_cache = CacheFactory.create_cache_from_env()
    env_cache.set("key1", "value1")
    print(f"Environment cache get: {env_cache.get('key1')}")

    # Wait for half of the TTL
    half_ttl = CacheConfig.get_default_ttl() / 2
    print(f"\nWaiting for {half_ttl} seconds (half of TTL)...")
    time.sleep(half_ttl)
    print(f"After {half_ttl}s: {env_cache.get('key1')}")

    # Wait for remaining TTL
    print(f"Waiting for {half_ttl + 1} more seconds (exceed TTL)...")
    time.sleep(half_ttl + 1)
    print(f"After full TTL + 1s: {env_cache.get('key1')}")

    # Using environment variables with function caching
    print("\n=== Function Caching with Environment Variables ===")

    # This will use the TTL from environment variable (10 seconds)
    @cached()
    def expensive_calculation(n):
        print(f"Executing expensive calculation for {n}...")
        time.sleep(1)  # Simulate expensive operation
        return n * n

    # First call - should execute
    print(f"First call result: {expensive_calculation(5)}")

    # Second call - should use cached result
    print(f"Second call result: {expensive_calculation(5)}")

    # Wait for TTL to expire
    ttl = CacheConfig.get_default_ttl()
    print(f"\nWaiting for {ttl + 1} seconds (exceed TTL)...")
    time.sleep(ttl + 1)

    # Third call - should execute again
    print(f"Third call result: {expensive_calculation(5)}")

    # Max size from environment variables
    print("\n=== Max Size from Environment Variables ===")
    # Max size is set to 5 in environment variables
    lru_cache = CacheFactory.create_lru_cache()

    # Add 6 items to a cache with max_size=5
    for i in range(6):
        lru_cache.set(f"key{i}", f"value{i}")
        print(f"Added key{i}")

    # Check which items are in the cache
    print("\nChecking cache contents:")
    for i in range(6):
        print(f"key{i}: {lru_cache.get(f'key{i}')}")

    print("\n=== Cache Statistics ===")
    stats = lru_cache.stats()
    print(f"Cache stats: {stats}")


if __name__ == "__main__":
    main()
