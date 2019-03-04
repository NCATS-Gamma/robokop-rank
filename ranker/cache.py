"""Cache module."""
import logging
import os
import pickle
import redis
from lru import LRU

logger = logging.getLogger(__name__)


class CacheSerializer:
    """Generic serializer."""

    def __init__(self):
        pass


class PickleCacheSerializer(CacheSerializer):
    """Use Python's default serialization."""

    def __init__(self):
        pass
    def dumps(self, obj):
        return pickle.dumps(obj)
    def loads(self, string):
        return pickle.loads(string)


class JSONCacheSerializer(CacheSerializer):
    pass  # would be nice


class Cache:
    """Cache objects by configurable means."""

    def __init__(self,
                 cache_path="cache",
                 serializer=PickleCacheSerializer,
                 redis_host="localhost",
                 redis_port=6379,
                 redis_db=0,
                 redis_password="",
                 enabled=True):
        """Connect to cache."""
        
        self.enabled = enabled
        try:
            if redis_password:
                self.redis = redis.StrictRedis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password)
            else:
                self.redis = redis.StrictRedis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db)

            self.redis.get('x')
            logger.info("Cache connected to redis at %s:%s/%s",
                        redis_host,
                        redis_port,
                        redis_db)
        except Exception:
            self.redis = None
            logger.error("Failed to connect to redis at %s:%s/%s",
                         redis_host,
                         redis_port,
                         redis_db)
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.cache = LRU(1000)
        self.serializer = serializer()

    def get(self, key):
        """Get a cached item by key."""
        result = None
        if not self.enabled:
            return result
        if key in self.cache:
            result = self.cache[key]
        elif self.redis:
            rec = self.redis.get(key)
            if rec is not None:
                result = self.serializer.loads(rec)
            else:
                result = None
            self.cache[key] = result
        else:
            path = os.path.join(self.cache_path, key)
            if os.path.exists(path):
                with open(path, 'rb') as stream:
                    result = self.serializer.loads(stream.read())
                    self.cache[key] = result
        return result

    def mget(self, *keys):
        """Get multiple cached items by key."""
        result = None
        if not self.enabled:
            return result
        result = []
        if self.redis:
            values = self.redis.mget(keys)
            for rec in values:
                if rec is not None:
                    result.append(self.serializer.loads(rec))
                else:
                    result.append(None)
        else:
            for key in keys:
                path = os.path.join(self.cache_path, key)
                if os.path.exists(path):
                    with open(path, 'rb') as stream:
                        result.append(self.serializer.loads(stream.read()))
                else:
                    result.append(None)
        return result

    def set(self, key, value):
        """Add an item to the cache."""
        if not self.enabled:
            return
        if self.redis:
            if value is not None:
                self.redis.set(key, self.serializer.dumps(value))
                self.cache[key] = value
        else:
            path = os.path.join(self.cache_path, key)
            with open(path, 'wb') as stream:
                stream.write(self.serializer.dumps(value))
            self.cache[key] = value

    def flush(self):
        """Flush redis cache."""
        self.redis.flushdb()
