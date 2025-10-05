from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"