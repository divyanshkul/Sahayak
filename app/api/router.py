from fastapi import APIRouter
from app.api import items, health, prabhandhak

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(prabhandhak.router, prefix="/prabhandhak", tags=["prabhandhak"])

