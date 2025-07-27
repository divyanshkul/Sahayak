#!/usr/bin/env python3

import os
import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, StdioConnectionParams
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import warnings
import logging
import re
import uuid
from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import settings
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from app.core.config import settings
from app.utils.gcp_storage import GCPStorageUploader

# Clean up warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("google.adk.tools.mcp_tool.mcp_tool").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)

# Configuration from settings
print("🔧 Configuring Google AI settings...")

if settings.google_api_key:
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    print("✅ Google API Key configured")
else:
    print("❌ Warning: Google API Key not found in settings")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = settings.google_genai_use_vertexai
print(f"🔧 Google GenAI Vertex AI mode: {settings.google_genai_use_vertexai}")

if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]
    print("🗑️ Removed GEMINI_API_KEY from environment")

# Use relative paths from settings
MANIM_SERVER_PATH = settings.manim_server_path or os.path.join(project_root, "app", "mcp", "manim_mcp.py")
PYTHON_ENV_PATH = settings.python_env_path or "python3"

print(f"📁 Manim Server Path: {MANIM_SERVER_PATH}")
print(f"🐍 Python Environment Path: {PYTHON_ENV_PATH}")

# Validate critical configurations
if not settings.google_api_key:
    print("⚠️ WARNING: Missing Google API Key - Manim agent may not function properly")
if not os.path.exists(MANIM_SERVER_PATH):
    print(f"⚠️ WARNING: Manim server not found at {MANIM_SERVER_PATH}")
else:
    print("✅ Manim server found")

print("🏁 Google AI configuration completed")

# Global variables
manim_agent = None
session_service = None
runner = None

def extract_video_path(response_text):
    """Extract video file path from agent response"""
    print(f"🔍 Searching for video path in response: {response_text}")
    
    # Pattern 1: "Video: /path/to/file.mp4"
    video_pattern = r"Video:\s*([^\n\r]+\.mp4)"
    match = re.search(video_pattern, response_text)
    if match:
        path = match.group(1).strip()
        print(f"✅ Found video path (pattern 1): {path}")
        return path
    
    # Pattern 2: Any absolute path to mp4
    path_pattern = r"(/[^\s]+\.mp4)"
    match = re.search(path_pattern, response_text)
    if match:
        path = match.group(1).strip()
        print(f"✅ Found video path (pattern 2): {path}")
        return path
    
    # Pattern 3: Look for any .mp4 file path
    mp4_pattern = r"([^\s]*\.mp4)"
    match = re.search(mp4_pattern, response_text)
    if match:
        path = match.group(1).strip()
        print(f"✅ Found video path (pattern 3): {path}")
        return path
    
    print("❌ No video path found in response")
    return None

async def initialize_agent():
    """Initialize the Manim agent and session"""
    global manim_agent, session_service, runner
    
    manim_agent = Agent(
        name="manim_agent_api",
        model="gemini-2.5-flash",
        description="Creates simple Manim animations and visualizations without complex math.",
        instruction="""You are a Manim animation expert. Create clear, visual animations following these rules:

IMPORTANT GUIDELINES:
- NO MathTex() or LaTeX - use Text() instead
- For multiple objects, position them using coordinates (e.g., UP, DOWN, LEFT, RIGHT)
- Use clear timing with self.wait() between actions
- Always validate scene code before creating it
- Use all the shapes you know to generate real life objects, like use ellipse for clouds etc.

POSITIONING TIPS:
- Use UP, DOWN, LEFT, RIGHT, UL, UR, DL, DR for positioning
- For multiple shapes: shape1.shift(LEFT*2), shape2.shift(RIGHT*2)
- Use Transform() to morph one shape into another
- Use Create() to draw objects, FadeIn/FadeOut for appearance

EXAMPLE STRUCTURE:
```python
class SceneName(Scene):
    def construct(self):
        # Create objects
        circle = Circle().shift(LEFT*2)
        square = Square().shift(RIGHT*2)
        
        # Animate them
        self.play(Create(circle))
        self.wait(0.5)
        self.play(Create(square))
        self.wait(1)
        
        # Transform or move
        self.play(Transform(circle, square.copy().shift(LEFT*2)))
        self.wait(2)
```

Always follow: validate → create → render""",

        tools=[
            MCPToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=PYTHON_ENV_PATH,
                        args=[MANIM_SERVER_PATH],
                    ),
                    timeout=200,
                ),
            ),
        ],
    )
    
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=manim_agent,
        app_name="manim_api_app",
        session_service=session_service
    )

async def generate_animation_for_api(prompt: str) -> Dict[str, Any]:
    """Generate animation from prompt for API use"""
    # Initialize agent if not already done
    if not manim_agent:
        await initialize_agent()
    
    # Generate unique session ID and scene name
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    user_id = "api_user"
    scene_name = f"Scene_{uuid.uuid4().hex[:6]}"
    
    print(f"🎬 Starting animation generation for scene: {scene_name}")
    print(f"📝 User prompt: {prompt}")
    
    # Create session
    await session_service.create_session(
        app_name="manim_api_app",
        user_id=user_id,
        session_id=session_id
    )
    
    # Create full prompt
    full_prompt = f"Create a scene called '{scene_name}': {prompt}"
    
    print(f"🤖 Sending to agent: {full_prompt}")
    
    try:
        # Prepare message
        content = types.Content(role='user', parts=[types.Part(text=full_prompt)])
        
        final_response_text = "Agent did not produce a final response."
        tool_calls = []
        
        print("🔄 Agent execution started...")
        
        # Execute agent
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Log event details
            event_author = getattr(event, 'author', 'unknown')
            print(f"📡 Agent event from: {event_author}")
            
            # Check for content in the event
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'function_call'):
                            tool_calls.append(part.function_call)
                            tool_name = getattr(part.function_call, 'name', 'unknown_tool')
                            print(f"🛠️ Tool called: {tool_name}")
                        elif hasattr(part, 'function_response'):
                            tool_result = getattr(part.function_response, 'response', 'no response')
                            print(f"🔧 Tool result: {tool_result}")
            
            # Check for actions (state changes, escalations, etc.)
            if hasattr(event, 'actions') and event.actions:
                if hasattr(event.actions, 'escalate') and event.actions.escalate:
                    print(f"⚠️ Agent escalated")
                if hasattr(event.actions, 'state_delta'):
                    print(f"📊 State change detected")
            
            # Check if this is the final response
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                    print(f"✅ Agent final response received")
                elif hasattr(event, 'actions') and event.actions and hasattr(event.actions, 'escalate'):
                    final_response_text = f"Agent escalated: {getattr(event, 'error_message', 'No specific message.')}"
                    print(f"⚠️ Agent escalated")
                break
        
        print(f"🔍 Agent response: {final_response_text}")
        
        # Extract video path from response
        video_path = extract_video_path(final_response_text)
        print(f"🎯 Extracted video path: {video_path}")
        
        # If no video path in response, check the media directory for the scene
        if not video_path:
            # Use the working path that we know exists
            media_dir = os.path.join("app", "mcp", "media", f"scene_{scene_name}")
            
            if os.path.exists(media_dir):
                # Direct path to expected video location: output/videos/SceneName/1080p60/
                video_dir = os.path.join(media_dir, "output", "videos", scene_name, "1080p60")
                
                if os.path.exists(video_dir):
                    video_file = os.path.join(video_dir, f"{scene_name}.mp4")
                    
                    if os.path.exists(video_file):
                        video_path = video_file
                        print(f"📹 Found video file: {video_path}")
                    else:
                        # Take the first .mp4 file if scene name doesn't match exactly
                        files = os.listdir(video_dir)
                        for file in files:
                            if file.endswith('.mp4'):
                                video_path = os.path.join(video_dir, file)
                                print(f"📹 Found video file: {video_path}")
                                break
        
        video_exists = video_path and os.path.exists(video_path) if video_path else False
        
        # Upload to GCP if video exists
        public_video_url = None
        gcp_upload_status = "skipped"
        
        if video_exists and settings.gcp_bucket_name and settings.gcp_credentials_path:
            try:
                print(f"🚀 Uploading video to GCP Storage...")
                
                # Create timestamp-based folder structure
                current_date = datetime.now().strftime("%Y-%m-%d")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_path = f"manim-animations/{current_date}/{timestamp}"
                
                # Initialize uploader
                uploader = GCPStorageUploader(
                    bucket_name=settings.gcp_bucket_name,
                    credentials_path=settings.gcp_credentials_path
                )
                
                # Upload video with public access
                upload_result = uploader.upload_file(
                    file_path=video_path,
                    destination_blob_name="video.mp4",
                    make_public=True,
                    folder=folder_path
                )
                
                if upload_result:
                    if upload_result.startswith("gs://"):
                        gs_parts = upload_result.replace("gs://", "").split("/", 1)
                        bucket_name = gs_parts[0]
                        file_path = gs_parts[1] if len(gs_parts) > 1 else ""
                        public_video_url = f"https://storage.googleapis.com/{bucket_name}/{file_path}"
                    else:
                        public_video_url = upload_result
                    
                    gcp_upload_status = "success"
                    print(f"✅ Video uploaded successfully: {public_video_url}")
                else:
                    gcp_upload_status = "failed"
                    print(f"❌ Video upload failed")
                    
            except Exception as e:
                gcp_upload_status = "failed"
                print(f"❌ Error uploading video to GCP: {str(e)}")
        elif video_exists:
            print(f"⚠️ GCP upload skipped - missing bucket name or credentials")
        
        result = {
            "scene_name": scene_name,
            "prompt": prompt,
            "agent_response": final_response_text,
            "video_path": video_path,
            "video_exists": video_exists,
            "public_video_url": public_video_url,
            "gcp_upload_status": gcp_upload_status,
            "tool_calls_made": len(tool_calls),
            "processing_status": "success" if video_exists else "failed"
        }
        
        if result["video_exists"]:
            print(f"✅ Animation created successfully: {video_path}")
        else:
            print(f"❌ Video file not found after processing")
            print(f"🔍 Checked path: {video_path}")
            print(f"📋 Agent response: {final_response_text}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating animation: {str(e)}")
        return {
            "scene_name": scene_name,
            "prompt": prompt,
            "agent_response": f"Error: {str(e)}",
            "video_path": None,
            "video_exists": False,
            "public_video_url": None,
            "gcp_upload_status": "skipped",
            "tool_calls_made": 0,
            "processing_status": "error"
        }

# This module now only contains the core agent logic for use by the API