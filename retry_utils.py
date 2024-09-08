"""_summary_

Returns:
    _type_: _description_
"""
import asyncio
import functools
import logging
import time

logger = logging.getLogger(__name__)

def retry(retries, delay, exceptions=(Exception,), immediate_exceptions=()):
    """  
    Retry decorator with support for both synchronous and asynchronous functions.
    Args:  
        retries (int): Number of retry attempts.  
        delay (int): Delay between retries in seconds.  
        exceptions (tuple): Tuple of exception types that should trigger a retry.  
        immediate_exceptions (tuple): Tuple of exception types that should trigger an 
        immediate retry without delay
    """
    def decorator_retry(func):
        @functools.wraps(func)
        async def async_wrapper_retry(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return await func(*args, **kwargs)
                except immediate_exceptions as e:
                    attempts += 1
                    logger.error("Immediate retry attempt %d failed: %s", attempts, e)
                    if attempts >= retries:
                        logger.error("All retry attempts failed.")
                        raise
                except exceptions as e:
                    attempts += 1
                    logger.error("Attempt %d failed: %s", attempts, e)
                    if attempts < retries:
                        logger.info("Retrying in %d seconds...", delay)
                        await asyncio.sleep(delay)
                    else:
                        logger.error("All retry attempts failed.")
                        raise
        @functools.wraps(func)
        def sync_wrapper_retry(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except immediate_exceptions as e:
                    attempts += 1
                    logger.error("Attempt %d failed: %s", attempts, e)
                    if attempts >= retries:
                        logger.error("All retry attempts failed.")
                        raise
                except exceptions as e:
                    attempts += 1
                    logger.error("Attempt %d failed: %s", attempts, e)
                    if attempts < retries:
                        logger.info("Retrying in %d seconds...", delay)
                        time.sleep(delay)
                    else:
                        logger.error("All retry attempts failed.")
                        raise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper_retry
        else:
            return sync_wrapper_retry
    return decorator_retry
