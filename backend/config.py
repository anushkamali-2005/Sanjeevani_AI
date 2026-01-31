from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    google_api_key: str
    
    # Splunk
    splunk_host: str = "localhost"
    splunk_port: int = 8089
    splunk_username: str = "admin"
    splunk_password: str = "changeme"
    splunk_scheme: str = "https"
    
    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./support_agent.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Agent Config
    confidence_threshold_auto: float = 0.90
    confidence_threshold_recommend: float = 0.70
    max_parallel_actions: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
