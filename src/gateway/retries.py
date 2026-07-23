import asyncio
from typing import Callable, Any
from loguru import logger

async def retry_with_backoff(
    func: Callable,
    *args,
    retries: int = 3,
    initial_delay: float = 1.5,
    exponential_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Executes an async function with exponential backoff on exceptions.
    """
    delay = initial_delay
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                logger.error(f"Task execution failed after {attempt} attempts: {e}")
                raise e
            logger.warning(
                f"Attempt {attempt}/{retries} failed: [{type(e).__name__}] {e or 'No error message'}. "
                f"Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
            delay *= exponential_factor
