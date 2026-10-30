from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import Token, UserCreate, UserLogin, UserResponse

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user account."""
    stmt = select(User).where(User.username == user_in.username)
    res = await db.execute(stmt)
    if res.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/token", response_model=Token)
async def login_for_access_token(user_in: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate with username and password to obtain a JWT Bearer token."""
    stmt = select(User).where(User.username == user_in.username)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=user.id, username=user.username)
    return Token(access_token=token, token_type="bearer", user_id=user.id)
