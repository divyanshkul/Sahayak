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

try:
    from agent import generate_animation_for_api, extract_video_path
except ImportError as e:
    print(f"Warning: Could not import manim agent: {e}")
    generate_animation_for_api = None
    extract_video_path = None

router = APIRouter()

class AnimationRequest(BaseModel):
    prompt: str

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