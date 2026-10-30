from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "MEMORYOS"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = Field(
        default="replace-with-a-strong-local-secret-key-32-chars-minimum",
        description="Cryptographic secret key for signing JWTs",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5438
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgrespassword"
    POSTGRES_DB: str = "memoryos"
    SQLALCHEMY_DATABASE_URI: str | None = None

    # Redis
    REDIS_URL: str = "redis://localhost:6385"
    REDIS_TTL_SECONDS: int = 3600
    REDIS_MAX_CONNECTIONS: int = 20

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    BATCH_EMBEDDING_SIZE: int = 32

    # LLM & Extraction
    LLM_PROVIDER: str = "groq"  # options: "groq", "offline_deterministic", "openai"
    LLM_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.50

    # Retrieval
    RETRIEVAL_RRF_K: int = 60
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.35
    MAX_RETRIEVAL_CANDIDATES: int = 20
    DEFAULT_RETRIEVAL_LIMIT: int = 5

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    def get_database_url(self) -> str:
        if self.SQLALCHEMY_DATABASE_URI:
            return self.SQLALCHEMY_DATABASE_URI
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
