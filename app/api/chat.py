from fastapi import APIRouter
from app.models.chat import ChatRequest, ChatResponse
from app.agents.chat_agent.agent import get_chat_response

router = APIRouter()

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat with the teaching assistant"""
    response_text = get_chat_response(request.text)
    return ChatResponse(response=response_text)