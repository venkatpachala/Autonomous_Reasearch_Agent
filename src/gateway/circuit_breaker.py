import time
from typing import Dict
from loguru import logger

class CircuitBreaker:
    """
    Prevents cascading failures by tripping provider routes after repeated timeouts or errors.
    """
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.state: Dict[str, str] = {}  # "closed", "open", "half-open"

    def can_execute(self, provider: str) -> bool:
        key = provider.lower()
        state = self.state.get(key, "closed")
        
        if state == "open":
            last_fail = self.last_failure_time.get(key, 0.0)
            # If cooldown period has elapsed, transition to half-open
            if time.time() - last_fail > self.cooldown_seconds:
                logger.info(f"Circuit breaker for provider '{provider}' transitioning to HALF-OPEN.")
                self.state[key] = "half-open"
                return True
            return False
        return True

    def record_success(self, provider: str):
        key = provider.lower()
        self.failures[key] = 0
        if self.state.get(key) == "half-open":
            logger.success(f"Circuit breaker for provider '{provider}' recovered to CLOSED.")
        self.state[key] = "closed"

    def record_failure(self, provider: str):
        key = provider.lower()
        self.failures[key] = self.failures.get(key, 0) + 1
        self.last_failure_time[key] = time.time()
        
        if self.failures[key] >= self.failure_threshold:
            logger.error(
                f"Circuit breaker for provider '{provider}' has TRIPPED to OPEN. "
                f"Directing traffic to fallbacks for the next {self.cooldown_seconds}s."
            )
            self.state[key] = "open"

circuit_breaker = CircuitBreaker()
