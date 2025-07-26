from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Pragati Backend"
    dummy_variable: str = "default_value"
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()