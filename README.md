# Cache-PYS

A Python library that provides advanced caching functionality with multiple TTL algorithms.

## Features

- Multiple TTL (Time To Live) algorithms:
  - Fixed TTL - Items expire after a fixed time period
  - Sliding TTL - TTL resets on each access
  - LRU (Least Recently Used) - Least recently accessed items are evicted first
  - LFU (Least Frequently Used) - Least frequently used items are evicted first
  - FIFO (First In First Out) - Oldest items are evicted first
  - Random - Items are randomly evicted
  - Hybrid - Combines multiple strategies in a weighted approach
- Thread-safe operations
- Background cleanup of expired items
- Function result caching using decorators
- Persistent cache storage to disk
- Bulk operations (get_many, set_many)
- Cache statistics
- Configuration through environment variables

## Installation

```bash
git clone https://github.com/ryuichi1208/cache-pys.git
cd cache-pys
pip install -e .
```

## Environment Variables

The cache can be configured using the following environment variables:

| Variable               | Description                                                                           | Default       |
| ---------------------- | ------------------------------------------------------------------------------------- | ------------- |
| CACHE_DEFAULT_TTL      | Default time-to-live in seconds                                                       | 3600 (1 hour) |
| CACHE_CLEANUP_INTERVAL | Interval for clearing expired items in seconds                                        | 60 (1 minute) |
| CACHE_MAX_SIZE         | Maximum number of items for size-based strategies                                     | 100           |
| CACHE_STRATEGY         | Default cache strategy ('fixed', 'sliding', 'lru', 'lfu', 'fifo', 'random', 'hybrid') | 'fixed'       |
| CACHE_WEIGHT_RECENCY   | Weight for recency in hybrid strategy (0-1)                                           | 0.5           |
| CACHE_WEIGHT_FREQUENCY | Weight for frequency in hybrid strategy (0-1)                                         | 0.5           |

Setting environment variables:

```bash
# Linux/Mac
export CACHE_DEFAULT_TTL=300  # 5 minutes
export CACHE_MAX_SIZE=1000    # 1000 items

# Windows
set CACHE_DEFAULT_TTL=300
set CACHE_MAX_SIZE=1000
```

## Quick Start

```python
from cache_pys.cache import CacheFactory, cached

# Create a cache from environment variables
cache = CacheFactory.create_cache_from_env()

# Or create a cache with specific settings
cache = CacheFactory.create_fixed_ttl_cache(ttl_seconds=60)

# Set values in the cache
cache.set("key1", "value1")

# Get values from the cache
value1 = cache.get("key1")  # Returns "value1" if not expired

# Bulk operations
cache.set_many({"key3": "value3", "key4": "value4"})
values = cache.get_many(["key1", "key3", "key5"])  # Returns {"key1": "value1", "key3": "value3"}

# Function caching using environment variable TTL
@cached()
def expensive_operation(param):
    # This function will only be executed once every CACHE_DEFAULT_TTL seconds
    # for the same parameter value
    return param * 2

# Function caching with explicit TTL
@cached(ttl=60)
def another_operation(param):
    return param * 3
```

## Multiple TTL Algorithms

```python
from cache_pys.cache import CacheFactory

# Fixed TTL cache - Items expire after a fixed time period
fixed_cache = CacheFactory.create_fixed_ttl_cache(ttl_seconds=3600)

# Sliding TTL cache - TTL resets on each access
sliding_cache = CacheFactory.create_sliding_ttl_cache(ttl_seconds=3600)

# LRU cache - Least recently accessed items are evicted first
lru_cache = CacheFactory.create_lru_cache(max_size=100)

# LFU cache - Least frequently used items are evicted first
lfu_cache = CacheFactory.create_lfu_cache(max_size=100)

# FIFO cache - Oldest items are evicted first
fifo_cache = CacheFactory.create_fifo_cache(max_size=100)

# Random replacement cache - Items are randomly evicted
random_cache = CacheFactory.create_random_cache(max_size=100)

# Hybrid cache - Considers both time-based and usage-based metrics
hybrid_cache = CacheFactory.create_hybrid_cache(
    ttl_seconds=3600,    # TTL in seconds
    max_size=100,        # Maximum size
    weight_recency=0.7,  # Weight for recency
    weight_frequency=0.3 # Weight for frequency
)
```

## Creating Custom TTL Strategies

```python
from cache_pys.cache import TTLStrategy, AdvancedCache

class MyCustomTTL(TTLStrategy):
    def __init__(self, max_accesses=10):
        self.max_accesses = max_accesses

    def is_expired(self, key, creation_time, access_data):
        # Item expires when access count exceeds limit
        return access_data.get('access_count', 0) >= self.max_accesses

    def on_access(self, key, access_data):
        access_data['access_count'] = access_data.get('access_count', 0) + 1

    def get_candidates_for_eviction(self, cache_data):
        # Evict items with highest access count first
        return sorted(cache_data.keys(),
                     key=lambda k: cache_data[k]['access_data'].get('access_count', 0),
                     reverse=True)

# Use a custom TTL strategy
custom_cache = AdvancedCache(ttl_strategy=MyCustomTTL(max_accesses=5))
```

## Client Class Usage

```python
from cache_pys.attr.client import Client

# Create a client
client = Client(name="Test", age=30)
print(client)  # Client(name=Test, age=30)

# Get help for the Client class
help_text = Client.help()
print(help_text)
```

## License

MIT
