from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from app.services.attendance.attendance_service import AttendanceService
from app.agents.prabhandhak_agent.agent import PrabhandhakAgent
from typing import Dict, Any
import os

router = APIRouter()
attendance_service = AttendanceService()
prabhandhak_agent = PrabhandhakAgent()

@router.post("/attendance/upload-photo")
async def upload_photo(
    photo: UploadFile = File(...),
    class_id: str = Form(...)
) -> Dict[str, Any]:
    """Upload photo for attendance processing and return attendance data"""
    if not photo.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        result = await attendance_service.process_attendance_photo(photo, class_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing photo: {str(e)}")


@router.get("/attendance/output-image/{class_id}")
async def get_attendance_image(class_id: str):
    """Get the processed attendance image with bounding boxes"""
    image_path = os.path.join("output", f"output_{class_id}.jpg")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Output image not found. Please process attendance first.")
    
    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        filename=f"attendance_result_{class_id}.jpg"
    )


@router.post("/ocr/process-image")
async def process_image_ocr(
    photo: UploadFile = File(...)
) -> Dict[str, Any]:
    """Process image for OCR and answer generation using RAG"""
    if not photo.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await photo.read()
        
        # Process with OCR agent
        result = await prabhandhak_agent.process_image_ocr(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image for OCR: {str(e)}")