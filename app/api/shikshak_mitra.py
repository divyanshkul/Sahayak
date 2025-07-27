from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import os
import sys
import asyncio
from pathlib import Path

# Add the manim agent to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent
agent_path = project_root / "agents" / "manim-agent"
sys.path.append(str(agent_path))

# Add the shikshak_mitra agent to the path
shikshak_agent_path = project_root / "agents" / "shikshak_mitra"
sys.path.append(str(shikshak_agent_path))

# Import manim agent functions
try:
    import importlib.util
    manim_spec = importlib.util.spec_from_file_location("manim_agent", agent_path / "agent.py")
    manim_module = importlib.util.module_from_spec(manim_spec)
    manim_spec.loader.exec_module(manim_module)
    generate_animation_for_api = manim_module.generate_animation_for_api
    extract_video_path = manim_module.extract_video_path
except Exception as e:
    print(f"Warning: Could not import manim agent: {e}")
    generate_animation_for_api = None
    extract_video_path = None

# Import shikshak mitra agent functions
try:
    shikshak_spec = importlib.util.spec_from_file_location("shikshak_agent", shikshak_agent_path / "agent.py")
    shikshak_module = importlib.util.module_from_spec(shikshak_spec)
    shikshak_spec.loader.exec_module(shikshak_module)
    invoke_shikshak_agent = shikshak_module.invoke_shikshak_agent
except Exception as e:
    print(f"Warning: Could not import shikshak mitra agent: {e}")
    invoke_shikshak_agent = None

router = APIRouter()

class AnimationRequest(BaseModel):
    prompt: str

class ShikshakMitraRequest(BaseModel):
    question: str

@router.post("/generation-questions")
async def generation_questions(request: ShikshakMitraRequest) -> Dict[str, Any]:
    """Generate questions using Shikshak Mitra agent with enhanced RAG and in-context learning"""
    if not invoke_shikshak_agent:
        raise HTTPException(status_code=500, detail="Shikshak Mitra agent not available")
    
    try:
        structured_response = await invoke_shikshak_agent(request.question)
        
        # Check if we got a structured JSON response or fallback response
        if "parse_error" in structured_response:
            return {
                "request": request.question,
                "questions": structured_response,
                "status": "partial_success",
                "message": "Response generated but JSON parsing failed"
            }
        else:
            return {
                "request": request.question,
                "questions": structured_response,
                "status": "success"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking Shikshak Mitra agent: {str(e)}")

@router.post("/generate-animation")
async def generate_animation(request: AnimationRequest) -> Dict[str, Any]:
    """Generate Manim animation from text prompt and return metadata"""
    if not generate_animation_for_api:
        raise HTTPException(status_code=500, detail="Manim agent not available")
    
    try:
        result = await generate_animation_for_api(request.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating animation: {str(e)}")

@router.get("/animation-video/{scene_name}")
async def get_animation_video(scene_name: str):
    """Get the generated animation video file"""
    # Look for video in common manim output locations
    possible_paths = [
        f"app/mcp/media/scene_{scene_name}/output/videos/720p30/{scene_name}.mp4",
        f"media/scene_{scene_name}/output/videos/720p30/{scene_name}.mp4",
        f"app/mcp/media/scene_{scene_name}/output/{scene_name}.mp4",
        f"{scene_name}.mp4"
    ]
    
    video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{scene_name}.mp4"
    )