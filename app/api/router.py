from fastapi import APIRouter
from app.api import items, health, prabhandhak, shikshak_mitra, chat

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(prabhandhak.router, prefix="/prabhandhak", tags=["prabhandhak"])
api_router.include_router(shikshak_mitra.router, prefix="/shikshak-mitra", tags=["shikshak-mitra"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])

