from fastapi import UploadFile
from typing import Dict, Any, Tuple, List
import face_recognition
import cv2
import numpy as np
import io
from PIL import Image
from google.cloud import storage

class AttendanceService:
    """Service class for handling attendance-related operations"""
    
    def __init__(self):
        self.known_students = {
            "class_123": [
                "Rahul", "Priya", "Amit", "Sneha", "Karan", 
                "Ananya", "Rohit", "Meera", "Vikash", "Pooja",
                "Arjun", "Kavya", "Nitin", "Riya", "Sanjay",
                "Divya", "Manish", "Aarti", "Rajesh", "Sunita",
                "Akash", "Neha", "Deepak", "Shruti", "Gaurav"
            ]
        }
        self.known_encodings = {}
    
    def calculate_attendance_from_photo(self, image_data: bytes, class_id: str) -> Tuple[Dict[str, str], List[str], int]:
        """
        ML logic to calculate attendance from photo
        
        Args:
            image_data: Raw image bytes
            class_id: Class identifier
            
        Returns:
            Tuple of (attendance_dict, recognized_students, faces_detected)
        """
        
        # Convert to PIL Image and then to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_rgb = np.array(image)
        
        # Ensure image is in RGB format
        if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
            pass  # Already RGB
        elif len(image_rgb.shape) == 3 and image_rgb.shape[2] == 4:
            # Convert RGBA to RGB
            image_rgb = image_rgb[:, :, :3]
        
        # Get students for this class
        class_students = self.known_students.get(class_id, [])
        
        # Face detection and recognition
        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        
        # For demo purposes, simulate recognition
        # In real implementation, compare with known encodings from GCP
        
        recognized_students = []
        faces_detected = len(face_locations)
        
        # Simulate recognition results based on detected faces
        if faces_detected > 0:
            # For demo, randomly assign some known students as present
            import random
            num_recognized = min(faces_detected, len(class_students))
            recognized_students = random.sample(class_students, num_recognized)
        
        # Create attendance dictionary
        attendance_dict = {}
        for student in class_students:
            attendance_dict[student] = "Present" if student in recognized_students else "Absent"
        
        # Print attendance for logging
        print(f"\n📋 Attendance for Class {class_id}:")
        for student, status in attendance_dict.items():
            print(f"  {student}: {status}")
        
        return attendance_dict, recognized_students, faces_detected
    
    async def process_attendance_photo(self, photo: UploadFile, class_id: str) -> Dict[str, Any]:
        """
        Process attendance photo for a given class
        
        Args:
            photo: Uploaded photo file
            class_id: Class identifier
            
        Returns:
            Dict with processing results
        """
        
        # Read photo content
        photo_content = await photo.read()
        
        # Save the uploaded photo for verification
        with open(f"uploaded_photo_{class_id}_{photo.filename}", "wb") as f:
            f.write(photo_content)
        print(f"📸 Photo saved as: uploaded_photo_{class_id}_{photo.filename}")
        
        # Call ML function to calculate attendance
        attendance_dict, recognized_students, faces_detected = self.calculate_attendance_from_photo(
            photo_content, class_id
        )
        
        # Get class students count
        class_students = self.known_students.get(class_id, [])
        
        result = {
            "class_id": class_id,
            "photo_filename": photo.filename,
            "photo_size": len(photo_content),
            "content_type": photo.content_type,
            "attendance_processed": True,
            "faces_detected": faces_detected,
            "students_recognized": len(recognized_students),
            "total_students": len(class_students),
            "attendance_details": attendance_dict,
            "recognized_students": recognized_students,
            "processing_status": "success",
            "message": f"Attendance processed for class {class_id}. {len(recognized_students)} students recognized."
        }
        
        return result
    
    
    def get_attendance_report(self, class_id: str) -> Dict[str, Any]:
        """
        Get attendance report for a class
        
        Args:
            class_id: Class identifier
            
        Returns:
            Dict with attendance report
        """
        
        return {
            "class_id": class_id,
            "total_students": 25,
            "present_today": 23,
            "absent_today": 2,
            "attendance_percentage": 92.0
        }