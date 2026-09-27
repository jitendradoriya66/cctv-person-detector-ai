import os
import hmac
import hashlib
import json
import base64
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer

logger = logging.getLogger("cctv_ai.auth")

SECRET_KEY = os.getenv("SECRET_KEY", "cctv_ai_sentinel_super_secret_jwt_key_2026_99x")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7  # 7 days

security_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a unique random salt."""
    salt = os.urandom(16)
    iterations = 100000
    hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.b64encode(salt).decode("utf-8")
    hash_b64 = base64.b64encode(hash_bytes).decode("utf-8")
    return f"pbkdf2:sha256:{iterations}${salt_b64}${hash_b64}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored PBKDF2 password hash."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 3 or not parts[0].startswith("pbkdf2:sha256:"):
            return False
        iterations = int(parts[0].split(":")[2])
        salt = base64.b64decode(parts[1])
        expected_hash = base64.b64decode(parts[2])
        actual_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual_hash, expected_hash)
    except Exception as e:
        logger.error(f"Error verifying password hash: {e}")
        return False


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def base64url_decode(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a signed HMAC-SHA256 JWT access token."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        
    payload.update({"exp": int(expire.timestamp()), "iat": int(datetime.utcnow().timestamp())})
    
    header_b64 = base64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    payload_b64 = base64url_encode(json.dumps(payload, separators=(',', ':')).encode('utf-8'))
    
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    signature_b64 = base64url_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a signed HMAC-SHA256 JWT access token."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = base64url_decode(signature_b64)
        
        if not hmac.compare_digest(actual_sig, expected_sig):
            logger.warning("JWT Token signature mismatch.")
            return None
        
        payload_bytes = base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        exp = payload.get("exp")
        if exp and int(time.time()) > exp:
            logger.warning("JWT Token expired.")
            return None
            
        return payload
    except Exception as e:
        logger.warning(f"Failed to decode JWT token: {e}")
        return None


def get_token_from_request(request: Request) -> Optional[str]:
    """Extracts JWT token from Cookie ('access_token') or Authorization header."""
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        if cookie_token.startswith("Bearer "):
            cookie_token = cookie_token[7:]
        return cookie_token

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
        
    return None


async def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Returns the logged-in user dictionary if token is valid, or None if unauthenticated."""
    token = get_token_from_request(request)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    return payload


async def get_current_user(request: Request) -> Dict[str, Any]:
    """Dependency that enforces authentication. Raises 401 HTTP error with clear message if unauthenticated."""
    user = await get_current_user_optional(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in with a valid user account to access Sentinel Command Center.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
