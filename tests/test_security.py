"""Unit-тесты безопасности: bcrypt-хеши и JWT."""

from __future__ import annotations

from webapp.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_roundtrip() -> None:
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed)
    assert not verify_password("wrong-pass", hashed)


def test_password_longer_than_bcrypt_limit() -> None:
    """Пароль > 72 байт не должен ронять bcrypt (обрезаем сами)."""
    long_pass = "a" * 200
    hashed = hash_password(long_pass)
    assert verify_password(long_pass, hashed)


def test_access_token_roundtrip() -> None:
    token = create_access_token(42, "bob")
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["username"] == "bob"
