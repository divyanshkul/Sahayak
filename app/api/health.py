from fastapi import APIRouter
from app.models.response import MessageResponse
from app.core.config import settings

router = APIRouter()

@router.get("/", response_model=MessageResponse)
def health_check():
    return MessageResponse(message="Service is healthy")

@router.get("/config", response_model=MessageResponse)
def get_config():
    return MessageResponse(message=f"Dummy variable: {settings.dummy_variable}")