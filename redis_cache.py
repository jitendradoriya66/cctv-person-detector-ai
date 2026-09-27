import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("cctv_ai.redis")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RedisCacheManager:
    """
    Redis Cache and Pub/Sub Manager.
    Provides high-performance caching for detection events & stats with automatic,
    fail-safe fallback to local database if Redis server is offline or not configured.
    """
    def __init__(self):
        self.redis_client = None
        self.is_connected = False
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.Redis.from_url(redis_url, socket_timeout=1.5, decode_responses=True)
            client.ping()
            self.redis_client = client
            self.is_connected = True
            logger.info("⚡ Connected to Redis Server! Event caching & Pub/Sub enabled.")
        except Exception as err:
            self.is_connected = False
            self.redis_client = None
            logger.info("ℹ️ Redis server not reachable. Operating with local SQLite & in-memory fallback.")

    def get_json(self, key: str) -> Optional[Any]:
        if not self.is_connected or not self.redis_client:
            return None
        try:
            val = self.redis_client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl_sec: int = 30) -> bool:
        if not self.is_connected or not self.redis_client:
            return False
        try:
            self.redis_client.setex(key, ttl_sec, json.dumps(value))
            return True
        except Exception:
            return False

    def delete_prefix(self, prefix: str = "events_*") -> bool:
        if not self.is_connected or not self.redis_client:
            return False
        try:
            keys = self.redis_client.keys(f"{prefix}*")
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception:
            return False

    def publish_event(self, channel: str, message: Dict[str, Any]) -> bool:
        if not self.is_connected or not self.redis_client:
            return False
        try:
            self.redis_client.publish(channel, json.dumps(message))
            return True
        except Exception:
            return False


redis_cache = RedisCacheManager()
