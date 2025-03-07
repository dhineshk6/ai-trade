from typing import Dict, List, Optional, Union
import requests
import logging
import time
from datetime import datetime
import hmac
import hashlib
import base64
import json
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

class APIUtils:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 6, 15)
        self.current_user = 'dhineshk6'
        self.session = requests.Session()
        self.rate_limits = {}
        self.retry_count = 3
        self.retry_delay = 1  # seconds
    
    def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth: Optional[Dict] = None,
        rate_limit: bool = True
    ) -> Dict:
        """Make HTTP request with retry logic and rate limiting"""
        try:
            # Check rate limits
            if rate_limit and not self._check_rate_limit(url):
                raise Exception("Rate limit exceeded")
            
            # Add authentication if provided
            if auth:
                headers = headers or {}
                auth_headers = self._generate_auth_headers(
                    method,
                    url,
                    params,
                    data,
                    auth
                )
                headers.update(auth_headers)
            
            # Retry logic
            for attempt in range(self.retry_count):
                try:
                    response = self.session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        json=data,
                        timeout=10
                    )
                    
                    # Update rate limits
                    self._update_rate_limits(response.headers)
                    
                    # Check response
                    response.raise_for_status()
                    return response.json()
                    
                except RequestException as e:
                    if attempt == self.retry_count - 1:
                        raise
                    time.sleep(self.retry_delay * (attempt + 1))
            
        except Exception as e:
            logger.error(f"API request error: {e}")
            return {'error': str(e)}
    
    def _generate_auth_headers(
        self,
        method: str,
        url: str,
        params: Optional[Dict],
        data: Optional[Dict],
        auth: Dict
    ) -> Dict:
        """Generate authentication headers"""
        try:
            timestamp = str(int(time.time() * 1000))
            
            # Create signature message
            message = timestamp
            if params:
                message += '?' + '&'.join(
                    f"{k}={v}" for k, v in sorted(params.items())
                )
            if data:
                message += json.dumps(data)
            
            # Generate signature
            signature = hmac.new(
                auth['secret_key'].encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return {
                'API-Key': auth['api_key'],
                'API-Timestamp': timestamp,
                'API-Signature': signature
            }
            
        except Exception as e:
            logger.error(f"Error generating auth headers: {e}")
            return {}
    
    def _check_rate_limit(self, url: str) -> bool:
        """Check if request is within rate limits"""
        try:
            if url not in self.rate_limits:
                return True
            
            limit_info = self.rate_limits[url]
            current_time = time.time()
            
            if current_time >= limit_info['reset_time']:
                return True
            
            return limit_info['remaining'] > 0
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    def _update_rate_limits(self, headers: Dict):
        """Update rate limit information from response headers"""
        try:
            if 'X-RateLimit-Limit' in headers:
                limit = int(headers['X-RateLimit-Limit'])
                remaining = int(headers['X-RateLimit-Remaining'])
                reset_time = float(headers['X-RateLimit-Reset'])
                
                self.rate_limits[headers['X-RateLimit-Scope']] = {
                    'limit': limit,
                    'remaining': remaining,
                    'reset_time': reset_time
                }
                
        except Exception as e:
            logger.error(f"Error updating rate limits: {e}")

# Initialize API utils
api_utils = APIUtils()