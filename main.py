from fastapi import FastAPI
from app.core.config import settings
from app.api.router import api_router
from app.models.response import MessageResponse
import uvicorn


app = FastAPI(title=settings.app_name, debug=settings.debug)

app.include_router(api_router, prefix="/api/v1")

@app.get("/", response_model=MessageResponse)
def read_root():
    return MessageResponse(message=f"Welcome to {settings.app_name}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=True)