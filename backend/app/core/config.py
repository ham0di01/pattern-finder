from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Pattern Trader API"

    # Database
    USE_SQLITE: bool = True
    
    # App Specific
    PATTERN_LENGTH: int = 24
    FUTURE_CANDLES: int = 12

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return "sqlite:///./data/sql_app.db"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
