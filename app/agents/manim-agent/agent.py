#!/usr/bin/env python3

import os
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import warnings
import logging
import re
import uuid
from typing import Dict, Any, List
import sys
from pathlib import Path
from datetime import datetime
import json
import subprocess

# Add parent directory to path to import settings
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from app.core.config import settings
from app.utils.gcp_storage import GCPStorageUploader

# Clean up warnings and logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Configuration from settings
print("🔧 Configuring Claude AI settings...")

if settings.anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
    print("✅ Anthropic API Key configured")
else:
    print("❌ Warning: Anthropic API Key not found in settings")

# Use relative paths from settings
MANIM_EXECUTABLE = settings.manim_executable or "manim"
MANIM_MEDIA_DIR = os.path.join(project_root, "app", "mcp", "media")

print(f"🎬 Manim Executable: {MANIM_EXECUTABLE}")
print(f"📁 Manim Media Directory: {MANIM_MEDIA_DIR}")

# Validate critical configurations
if not settings.anthropic_api_key:
    print("⚠️ WARNING: Missing Anthropic API Key - Manim agent may not function properly")

print("🏁 Claude AI configuration completed")

# Global variables
manim_llm = None

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

def create_manim_scene_code(scene_name: str, scene_code: str) -> str:
    """Create a Python file with Manim scene code"""
    scene_dir = os.path.join(MANIM_MEDIA_DIR, f"scene_{scene_name}")
    os.makedirs(scene_dir, exist_ok=True)

    scene_file = os.path.join(scene_dir, f"{scene_name}.py")

    with open(scene_file, 'w') as f:
        f.write(scene_code)

    return scene_file

def render_manim_scene(scene_file: str, scene_name: str, quality: str = "1080p60") -> Dict[str, Any]:
    """Render a Manim scene and return the video path"""
    try:
        scene_dir = os.path.dirname(scene_file)

        # Run Manim command
        cmd = [
            MANIM_EXECUTABLE,
            scene_file,
            scene_name,
            "-qh",  # High quality
            "-o", f"{scene_name}.mp4",
            "--media_dir", os.path.join(scene_dir, "output")
        ]

        print(f"🎬 Rendering Manim scene: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode == 0:
            # Find the output video
            video_dir = os.path.join(scene_dir, "output", "videos", scene_name, quality)
            video_file = os.path.join(video_dir, f"{scene_name}.mp4")

            if os.path.exists(video_file):
                return {
                    "status": "success",
                    "video_path": video_file,
                    "message": "Scene rendered successfully"
                }
            else:
                # Search for any mp4 file
                for file in os.listdir(video_dir) if os.path.exists(video_dir) else []:
                    if file.endswith('.mp4'):
                        video_file = os.path.join(video_dir, file)
                        return {
                            "status": "success",
                            "video_path": video_file,
                            "message": "Scene rendered successfully"
                        }

                return {
                    "status": "error",
                    "message": f"Video file not found after rendering. Expected: {video_file}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "status": "error",
                "message": f"Manim rendering failed with code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Manim rendering timed out after 180 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error rendering scene: {str(e)}"
        }

async def initialize_agent():
    """Initialize the Claude LLM for Manim generation"""
    global manim_llm

    manim_llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
        max_tokens=8096,
    )

async def generate_animation_for_api(prompt: str) -> Dict[str, Any]:
    """Generate animation from prompt for API use"""
    # Initialize agent if not already done
    if not manim_llm:
        await initialize_agent()

    # Generate unique scene name
    scene_name = f"Scene_{uuid.uuid4().hex[:6]}"

    print(f"🎬 Starting animation generation for scene: {scene_name}")
    print(f"📝 User prompt: {prompt}")

    try:
        # Create prompt for Claude to generate Manim code
        system_prompt = """You are a Manim animation expert. Generate Python code for Manim animations following these rules:

IMPORTANT GUIDELINES:
- Import: from manim import *
- NO MathTex() or LaTeX - use Text() instead
- For multiple objects, position them using coordinates (e.g., UP, DOWN, LEFT, RIGHT)
- Use clear timing with self.wait() between actions
- Use all the shapes you know to generate real life objects, like use ellipse for clouds etc.

POSITIONING TIPS:
- Use UP, DOWN, LEFT, RIGHT, UL, UR, DL, DR for positioning
- For multiple shapes: shape1.shift(LEFT*2), shape2.shift(RIGHT*2)
- Use Transform() to morph one shape into another
- Use Create() to draw objects, FadeIn/FadeOut for appearance

EXAMPLE STRUCTURE:
```python
from manim import *

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

Always have supporting text in the scene. Return ONLY the Python code, no explanations."""

        user_prompt = f"Create a Manim scene class named '{scene_name}' that: {prompt}"

        messages = [
            HumanMessage(content=f"{system_prompt}\n\n{user_prompt}")
        ]

        print("🤖 Generating Manim code with Claude...")

        # Get response from Claude
        response = await manim_llm.ainvoke(messages)
        scene_code = response.content

        print(f"✅ Received scene code ({len(scene_code)} chars)")

        # Extract Python code if wrapped in markdown
        if "```python" in scene_code:
            scene_code = scene_code.split("```python")[1].split("```")[0].strip()
        elif "```" in scene_code:
            scene_code = scene_code.split("```")[1].split("```")[0].strip()

        # Ensure it has the import
        if "from manim import" not in scene_code:
            scene_code = "from manim import *\n\n" + scene_code

        print("📝 Creating scene file...")
        scene_file = create_manim_scene_code(scene_name, scene_code)
        print(f"✅ Scene file created: {scene_file}")

        print("🎬 Rendering scene...")
        render_result = render_manim_scene(scene_file, scene_name)

        if render_result["status"] == "success":
            video_path = render_result["video_path"]
            print(f"✅ Video rendered: {video_path}")

            # Upload to GCP if video exists
            public_video_url = None
            gcp_upload_status = "skipped"

            if settings.gcp_bucket_name and settings.gcp_credentials_path:
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
                            file_path_gcp = gs_parts[1] if len(gs_parts) > 1 else ""
                            public_video_url = f"https://storage.googleapis.com/{bucket_name}/{file_path_gcp}"
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
            else:
                print(f"⚠️ GCP upload skipped - missing bucket name or credentials")

            return {
                "scene_name": scene_name,
                "prompt": prompt,
                "agent_response": f"Scene created successfully with Claude",
                "scene_code": scene_code,
                "video_path": video_path,
                "video_exists": True,
                "public_video_url": public_video_url,
                "gcp_upload_status": gcp_upload_status,
                "processing_status": "success"
            }
        else:
            print(f"❌ Rendering failed: {render_result['message']}")
            return {
                "scene_name": scene_name,
                "prompt": prompt,
                "agent_response": render_result["message"],
                "scene_code": scene_code,
                "video_path": None,
                "video_exists": False,
                "public_video_url": None,
                "gcp_upload_status": "skipped",
                "processing_status": "error",
                "error_details": render_result
            }

    except Exception as e:
        print(f"❌ Error generating animation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "scene_name": scene_name,
            "prompt": prompt,
            "agent_response": f"Error: {str(e)}",
            "video_path": None,
            "video_exists": False,
            "public_video_url": None,
            "gcp_upload_status": "skipped",
            "processing_status": "error"
        }

# This module now only contains the core agent logic for use by the API