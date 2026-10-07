"""Безопасность: хеширование паролей (bcrypt) и JWT-токены (PyJWT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET
from webapp.database import get_db
from webapp.models import User

# bcrypt работает с первыми 72 байтами пароля — обрезаем сами,
# чтобы длинный пароль не вызывал ошибку кодирования
_BCRYPT_MAX_BYTES = 72

# Схема авторизации: заголовок "Authorization: Bearer <token>"
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """
    Хеширует пароль алгоритмом bcrypt.

    Args:
        password: Пароль в открытом виде.

    Returns:
        Хеш в виде строки (соль включена внутрь хеша).
    """
    raw = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Проверяет пароль против сохранённого хеша.

    Args:
        password: Введённый пароль.
        password_hash: Хеш из базы.

    Returns:
        True, если пароль совпадает.
    """
    raw = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(raw, password_hash.encode("utf-8"))
    except ValueError:
        # Повреждённый хеш в базе — считаем пароль неверным, не роняем запрос
        return False


def create_access_token(user_id: int, username: str) -> str:
    """
    Выпускает JWT-токен доступа.

    Args:
        user_id: ID пользователя (кладём в claim "sub").
        username: Имя пользователя (для удобства отладки).

    Returns:
        Подписанный JWT.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Декодирует и проверяет JWT.

    Args:
        token: Токен из заголовка Authorization.

    Returns:
        Полезная нагрузка токена.

    Raises:
        HTTPException: 401, если токен просрочен или подпись неверна.
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия истекла, войдите заново",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный токен",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI-зависимость: возвращает текущего пользователя по JWT.

    Args:
        credentials: Данные заголовка Authorization.
        db: Сессия БД.

    Returns:
        ORM-объект пользователя.

    Raises:
        HTTPException: 401, если токен отсутствует или пользователь не найден.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Некорректный токен")

    user = db.get(User, int(user_id))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Пользователь не найден"
        )
    return user
