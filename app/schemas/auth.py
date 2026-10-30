from pydantic import BaseModel, ConfigDict, EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


class TokenPayload(BaseModel):
    sub: str | None = None
    username: str | None = None
    exp: int | None = None


class UserCreate(BaseModel):
    username: str
    password: str
    email: EmailStr | None = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    email: str | None = None
    is_active: bool
