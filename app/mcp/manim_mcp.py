#!/usr/bin/env python3

import os
import tempfile
import subprocess
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("Manim Animation Server")

MANIM_EXECUTABLE = os.getenv("MANIM_EXECUTABLE", "manim")
BASE_DIR = Path(__file__).parent / "media"
BASE_DIR.mkdir(exist_ok=True)

@mcp.tool()
def create_scene(scene_name: str, scene_code: str) -> str:
    """
    Create a new Manim scene from Python code.
    
    Args:
        scene_name: Name for the scene class and file
        scene_code: Python code for the Manim scene
        
    Returns:
        Success message with file path
    """
    # Create temp directory
    temp_dir = BASE_DIR / f"scene_{scene_name}"
    temp_dir.mkdir(exist_ok=True)
    
    # Add manim imports if missing
    if "from manim import *" not in scene_code and "import manim" not in scene_code:
        scene_code = "from manim import *\n\n" + scene_code
    
    # Save scene to file
    scene_file = temp_dir / f"{scene_name}.py"
    scene_file.write_text(scene_code)
    
    return f"✅ Scene '{scene_name}' created successfully at {scene_file}"

@mcp.tool()
def render_scene(scene_name: str) -> str:
    """
    Render a Manim scene to video.
    
    Args:
        scene_name: Name of the scene to render
        
    Returns:
        Success message with video path or error details
    """
    scene_dir = BASE_DIR / f"scene_{scene_name}"
    scene_file = scene_dir / f"{scene_name}.py"
    
    if not scene_file.exists():
        return f"❌ Error: Scene '{scene_name}' not found. Create it first using create_scene."
    
    try:
        # Run manim render command
        cmd = [
            MANIM_EXECUTABLE,
            "render",
            str(scene_file),
            scene_name,
            "--media_dir", str(scene_dir / "output")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(scene_dir)
        )
        
        if result.returncode == 0:
            # Find the generated video
            output_dir = scene_dir / "output"
            video_files = list(output_dir.glob(f"**/*{scene_name}*.mp4"))
            
            if video_files:
                video_path = video_files[0]
                return f"✅ Scene '{scene_name}' rendered successfully!\n📹 Video: {video_path}"
            else:
                return f"⚠️ Render completed but no video found.\n📋 Output:\n{result.stdout}"
        else:
            return f"❌ Manim render failed!\n📋 Error:\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        return f"❌ Render timeout! Animation took longer than 5 minutes."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

@mcp.tool()
def validate_scene(scene_code: str) -> str:
    """
    Validate Manim scene code for syntax and structure.
    
    Args:
        scene_code: Python code to validate
        
    Returns:
        Validation result with any issues found
    """
    try:
        # Add imports for compilation check
        if "from manim import *" not in scene_code and "import manim" not in scene_code:
            full_code = "from manim import *\n\n" + scene_code
        else:
            full_code = scene_code
            
        # Try to compile the code
        compile(full_code, '<string>', 'exec')
        
        # Basic structural checks
        issues = []
        if "class" not in scene_code:
            issues.append("Missing class definition")
        if "Scene" not in scene_code:
            issues.append("Should inherit from Scene")
        if "def construct(self)" not in scene_code:
            issues.append("Missing construct() method")
        
        if issues:
            return f"⚠️ Syntax OK, but potential issues:\n" + "\n".join(f"• {issue}" for issue in issues)
        else:
            return "✅ Scene code looks good!"
            
    except SyntaxError as e:
        return f"❌ Syntax error on line {e.lineno}: {e.msg}"
    except Exception as e:
        return f"❌ Validation error: {str(e)}"

if __name__ == "__main__":
    mcp.run()