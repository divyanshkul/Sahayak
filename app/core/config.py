from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Pragati Backend"
    dummy_variable: str = "default_value"
    debug: bool = False

    # Anthropic/Claude configuration
    anthropic_api_key: str = ""

    # Manim configuration
    google_api_key: str = ""
    google_genai_use_vertexai: str = "False"
    manim_server_path: str = ""
    python_env_path: str = ""
    manim_executable: str = ""

    # GCP Storage configuration
    gcp_bucket_name: str = ""
    gcp_credentials_path: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()