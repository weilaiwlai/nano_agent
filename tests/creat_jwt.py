import base64
import json
import hmac
import hashlib
import time

def create_jwt(subject: str, secret: str, expire_hours: int = 24) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "exp": int(time.time()) + expire_hours * 3600,
        "iat": int(time.time())
    }
    
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
    
    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
    
    return f"{message}.{signature_b64}"

# 使用你的配置
secret = "abC123def456GHI789jkl012MNO345PQR678"
token = create_jwt("user_001", secret)
print(f"JWT Token: {token}")