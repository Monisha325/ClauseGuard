from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "clauseguard"
    postgres_password: str = "clauseguard"
    postgres_db: str = "clauseguard"
    postgres_host: str = "db"
    postgres_port: int = 5432

    db_connect_retries: int = 10
    db_connect_retry_delay_seconds: float = 2.0

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440

    # M36: per-user daily contract-upload cap (see middleware/rate_limit.py
    # for the enforcement logic and full reasoning). Defaulted, not
    # required, so existing deployments/tests keep working unchanged if
    # this var is never set -- 10/day is a reasonable real production
    # default; testing needs a much lower value (e.g. 1), set via the
    # REAL env var mechanism (.env or docker-compose), never a code edit.
    daily_upload_limit_per_user: int = 10

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError(
                "JWT_SECRET_KEY must be set to a real secret (at least 16 characters)"
            )
        return v

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
