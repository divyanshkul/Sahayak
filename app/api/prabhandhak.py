from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.attendance.attendance_service import AttendanceService
from typing import Dict, Any

router = APIRouter()
attendance_service = AttendanceService()

@router.post("/attendance/upload-photo")
async def upload_photo(
    photo: UploadFile = File(...),
    class_id: str = Form(...)
) -> Dict[str, Any]:
    """Upload photo for attendance processing"""
    # Validate file type
    if not photo.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        result = await attendance_service.process_attendance_photo(photo, class_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing photo: {str(e)}")
    



    
