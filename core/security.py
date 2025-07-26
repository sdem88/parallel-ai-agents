"""
Enterprise security system with authentication, authorization, and rate limiting
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import hashlib
import hmac
from functools import wraps

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import get_config


logger = structlog.get_logger(__name__)


class SecurityError(Exception):
    """Base security exception"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass


class AuthorizationError(SecurityError):
    """Authorization failed"""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded"""
    pass


class PasswordManager:
    """Secure password hashing and verification"""
    
    def __init__(self):
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12  # Strong hashing
        )
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_secure_password(self, length: int = 32) -> str:
        """Generate a cryptographically secure password"""
        return secrets.token_urlsafe(length)


class JWTManager:
    """JWT token management with enterprise features"""
    
    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.security.jwt_secret_key
        self.algorithm = self.config.security.jwt_algorithm
        self.expiration_hours = self.config.security.jwt_expiration_hours
        
        if len(self.secret_key) < 32:
            raise SecurityError("JWT secret key must be at least 32 characters")
    
    def create_access_token(
        self, 
        subject: str, 
        user_data: Dict[str, Any] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        
        to_encode = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        if user_data:
            to_encode.update(user_data)
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.info(
            "Access token created",
            subject=subject,
            expires_at=expire.isoformat()
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, subject: str) -> str:
        """Create a refresh token with longer expiration"""
        expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # Check token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token"""
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")
            
            # Create new access token
            return self.create_access_token(payload["sub"])
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.JWTError as e:
            raise AuthenticationError(f"Refresh token validation failed: {str(e)}")


class APIKeyManager:
    """API key management for service-to-service authentication"""
    
    def __init__(self):
        self.config = get_config()
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from configuration"""
        # In a real implementation, this would load from a database
        return {
            self.config.security.api_key: {
                "name": "default",
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.utcnow(),
                "last_used": None
            }
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return its metadata"""
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key].copy()
            key_info["last_used"] = datetime.utcnow()
            
            logger.info(
                "API key validated",
                key_name=key_info["name"],
                permissions=key_info["permissions"]
            )
            
            return key_info
        
        logger.warning("Invalid API key attempted", key_prefix=api_key[:8] + "...")
        return None
    
    def generate_api_key(self, name: str, permissions: List[str]) -> str:
        """Generate a new API key"""
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "last_used": None
        }
        
        logger.info("New API key generated", name=name, permissions=permissions)
        return api_key


class RateLimiter:
    """Advanced rate limiting with different strategies"""
    
    def __init__(self):
        self.config = get_config()
        self.limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[f"{self.config.security.rate_limit_per_minute}/minute"]
        )
    
    def get_limiter(self):
        """Get the rate limiter instance"""
        return self.limiter
    
    def create_custom_limit(self, limit: str):
        """Create a custom rate limit decorator"""
        def decorator(func):
            return self.limiter.limit(limit)(func)
        return decorator


class SecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self):
        self.config = get_config()
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter()
        
        # Security settings
        self.security_bearer = HTTPBearer()
        
        logger.info("Security manager initialized")
    
    async def authenticate_user(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Authenticate user via JWT token"""
        try:
            token_data = self.jwt_manager.verify_token(credentials.credentials)
            return {
                "user_id": token_data["sub"],
                "token_data": token_data,
                "auth_type": "jwt"
            }
        except AuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def authenticate_api_key(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate via API key"""
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return None
        
        key_info = self.api_key_manager.validate_api_key(api_key)
        
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return {
            "api_key_info": key_info,
            "auth_type": "api_key"
        }
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get authentication info from request context
                # This would be implemented based on your specific auth flow
                auth_info = kwargs.get("auth_info")
                
                if not auth_info:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Check permissions
                if auth_info["auth_type"] == "api_key":
                    permissions = auth_info["api_key_info"]["permissions"]
                    if required_permission not in permissions and "admin" not in permissions:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Permission '{required_permission}' required"
                        )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def secure_hash(self, data: str, salt: str = None) -> str:
        """Create a secure hash of data"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        combined = f"{data}{salt}".encode('utf-8')
        hash_object = hashlib.sha256(combined)
        
        return f"{salt}:{hash_object.hexdigest()}"
    
    def verify_hash(self, data: str, hash_with_salt: str) -> bool:
        """Verify data against a hash"""
        try:
            salt, expected_hash = hash_with_salt.split(':', 1)
            actual_hash = hashlib.sha256(f"{data}{salt}".encode('utf-8')).hexdigest()
            return hmac.compare_digest(expected_hash, actual_hash)
        except ValueError:
            return False
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for form protection"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.config.security.jwt_secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def verify_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token"""
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check if token is not too old
            if datetime.utcnow().timestamp() - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp_str}"
            expected_signature = hmac.new(
                self.config.security.jwt_secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, TypeError):
            return False


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# Dependency functions for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> Dict[str, Any]:
    """FastAPI dependency for JWT authentication"""
    return await get_security_manager().authenticate_user(credentials)


async def get_api_key_auth(request: Request) -> Dict[str, Any]:
    """FastAPI dependency for API key authentication"""
    auth_info = await get_security_manager().authenticate_api_key(request)
    if not auth_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    return auth_info


def require_permission(permission: str):
    """FastAPI dependency factory for permission checking"""
    return get_security_manager().require_permission(permission)