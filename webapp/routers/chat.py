"""Роутер чата: вопросы к RAG, история диалогов, управление чатами."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from webapp import rag_service
from webapp.database import get_db
from webapp.models import Chat, Message, Source, User
from webapp.schemas import (
    AskRequest,
    AskResponse,
    ChatCreate,
    ChatDetail,
    ChatOut,
    ChatRename,
    SourceOut,
)
from webapp.security import get_current_user

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _get_owned_chat(chat_id: int, user: User, db: Session) -> Chat:
    """
    Возвращает чат пользователя или выбрасывает 404.

    Проверка владельца обязательна: без неё любой авторизованный
    пользователь мог бы читать чужие диалоги по id.

    Args:
        chat_id: ID чата.
        user: Текущий пользователь.
        db: Сессия БД.

    Returns:
        ORM-объект чата.

    Raises:
        HTTPException: 404, если чата нет или он чужой.
    """
    chat = db.scalar(
        select(Chat)
        .where(Chat.id == chat_id, Chat.user_id == user.id)
        .options(selectinload(Chat.messages).selectinload(Message.sources))
    )
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Чат не найден")
    return chat


@router.post("", response_model=ChatOut, status_code=status.HTTP_201_CREATED)
def create_chat(
    payload: ChatCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Chat:
    """Создаёт новый пустой чат."""
    chat = Chat(user_id=user.id, title=payload.title or "Новый чат")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("", response_model=list[ChatOut])
def list_chats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Chat]:
    """Возвращает чаты пользователя, свежие — сверху."""
    return list(
        db.scalars(
            select(Chat)
            .where(Chat.user_id == user.id)
            .order_by(Chat.updated_at.desc())
        )
    )


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Chat:
    """Возвращает чат с полной историей сообщений и источниками."""
    return _get_owned_chat(chat_id, user, db)


@router.patch("/{chat_id}", response_model=ChatOut)
def rename_chat(
    chat_id: int,
    payload: ChatRename,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Chat:
    """Переименовывает чат."""
    chat = _get_owned_chat(chat_id, user, db)
    chat.title = payload.title
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    """Удаляет чат вместе с сообщениями (каскадно)."""
    chat = _get_owned_chat(chat_id, user, db)
    db.delete(chat)
    db.commit()


@router.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AskResponse:
    """
    Задаёт вопрос RAG-системе, сохраняет диалог и возвращает ответ с источниками.

    Если chat_id не указан — создаётся новый чат, а его заголовком
    становится первые слова вопроса.

    Args:
        payload: Вопрос, опционально chat_id / top_k / filter_lesson.
        db: Сессия БД.
        user: Текущий пользователь.

    Returns:
        Ответ модели и список источников (урок, страница, score).
    """
    if payload.chat_id is not None:
        chat = _get_owned_chat(payload.chat_id, user, db)
    else:
        title = payload.question.strip()[:60]
        chat = Chat(user_id=user.id, title=title or "Новый чат")
        db.add(chat)
        db.commit()
        db.refresh(chat)

    # Сохраняем вопрос пользователя
    user_message = Message(chat_id=chat.id, role="user", content=payload.question)
    db.add(user_message)
    db.commit()

    # Обращаемся к RAG; ошибки внешних сервисов превращаем в 502
    try:
        result = rag_service.ask(
            payload.question, top_k=payload.top_k, filter_lesson=payload.filter_lesson
        )
    except Exception as error:  # noqa: BLE001 — отдаём клиенту понятную ошибку
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Сбой при обращении к модели: {error}",
        )

    assistant_message = Message(chat_id=chat.id, role="assistant", content=result["answer"])
    db.add(assistant_message)
    db.flush()  # нужен id до создания источников

    for src in result["sources"]:
        db.add(
            Source(
                message_id=assistant_message.id,
                lesson=int(src["lesson"]),
                page=int(src["page"]),
                score=float(src["score"]),
            )
        )

    # Любое сообщение поднимает чат наверх в списке истории
    chat.updated_at = assistant_message.created_at or chat.updated_at
    db.commit()
    db.refresh(assistant_message)

    return AskResponse(
        chat_id=chat.id,
        message_id=assistant_message.id,
        question=payload.question,
        answer=result["answer"],
        sources=[
            SourceOut(lesson=s["lesson"], page=s["page"], score=s["score"])
            for s in result["sources"]
        ],
    )
