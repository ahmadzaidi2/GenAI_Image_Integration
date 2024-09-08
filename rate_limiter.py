from fastapi import Request, HTTPException, status
from collections import defaultdict
from time import time
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

max_request = int(config['rate_limit']['max_request'])
window_sec = int(config['rate_limit']['window_sec'])

# In-memory store for tracking request counts. For production, consider using Redis or another shared store.
rate_limits = defaultdict(list)
MAX_REQUESTS = max_request
WINDOW_SECONDS = window_sec

def rate_limiter(request: Request):
    """
    Rate limiting dependency for FastAPI routes.

    Args:
        request (Request): FastAPI request object.

    Raises:
        HTTPException: Thrown when the rate limit is exceeded.
    """
    client_ip = request.client.host
    current_time = time()
    
    # Remove outdated request timestamps that are outside the current window
    rate_limits[client_ip] = [timestamp for timestamp in rate_limits[client_ip] if current_time - timestamp < WINDOW_SECONDS]
    
    # Check if client has exceeded the rate limit
    if len(rate_limits[client_ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests, please try again later."
        )
    
    # Record the new request timestamp
    rate_limits[client_ip].append(current_time)