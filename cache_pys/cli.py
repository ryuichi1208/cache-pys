class Cache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

def help():
    return """
    Cache Class:
    - __init__: Initializes the cache.
    - get: Retrieves a value from the cache by key.
    - set: Stores a value in the cache with a key.
    - delete: Removes a value from the cache by key.
    """
