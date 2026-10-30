from fastapi import Header, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError
from app.core.security import Principal, decode_access_token
from app.models.character import Character
from app.models.conversation import Conversation


async def get_current_principal(
    authorization: str | None = Header(None, alias="Authorization"),
) -> Principal:
    """Extract and validate the JWT Bearer token from the Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Must be 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    try:
        return decode_access_token(token)
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_character_access(
    character_id: str,
    principal: Principal,
    db: AsyncSession,
) -> Character:
    """Verify that the character exists and is owned by the authenticated principal."""
    stmt = select(Character).where(
        and_(
            Character.id == character_id,
            Character.owner_user_id == principal.user_id,
        )
    )
    res = await db.execute(stmt)
    character = res.scalars().first()
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or access denied",
        )
    return character


async def verify_conversation_access(
    conversation_id: str,
    character_id: str,
    principal: Principal,
    db: AsyncSession,
) -> Conversation:
    """Verify that the conversation exists and belongs to the principal and character."""
    stmt = select(Conversation).where(
        and_(
            Conversation.id == conversation_id,
            Conversation.owner_user_id == principal.user_id,
            Conversation.character_id == character_id,
        )
    )
    res = await db.execute(stmt)
    conversation = res.scalars().first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied",
        )
    return conversation
