from pydantic import BaseModel

class MessageResponse(BaseModel):
    message: str
    status: str = "success"