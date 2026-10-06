"""Роутер авторизации: регистрация, вход, профиль."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.database import get_db
from webapp.models import User
from webapp.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from webapp.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Регистрирует пользователя и сразу выдаёт токен.

    Args:
        payload: Логин и пароль.
        db: Сессия БД.

    Returns:
        Токен доступа и имя пользователя.

    Raises:
        HTTPException: 409, если имя уже занято.
    """
    existing = db.scalar(select(User).where(User.username == payload.username))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Пользователь с таким именем уже существует",
        )

    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(user.id, user.username),
        username=user.username,
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Проверяет логин и пароль, выдаёт токен.

    Args:
        payload: Логин и пароль.
        db: Сессия БД.

    Returns:
        Токен доступа и имя пользователя.

    Raises:
        HTTPException: 401 при неверных данных.
    """
    user = db.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
        )

    return TokenResponse(
        access_token=create_access_token(user.id, user.username),
        username=user.username,
    )


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> User:
    """
    Возвращает профиль текущего пользователя.

    Args:
        current_user: Пользователь из JWT.

    Returns:
        Публичные данные пользователя.
    """
    return current_user
