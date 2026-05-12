"""
sv_platform.auth.password — 사용자 비밀번호 해싱 (argon2)

argon2를 사용 — bcrypt보다 메모리-hard, 현대 표준.
"""
from passlib.context import CryptContext


_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(plain: str) -> str:
    """평문 → 해시. 회원가입/비밀번호 변경 시 호출."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """로그인 시 입력 비밀번호와 저장된 해시 비교."""
    try:
        return _pwd_context.verify(plain, hashed)
    except Exception:
        return False
