from fastapi import UploadFile
from typing import Dict, Any, Tuple, List
import face_recognition
import cv2
import numpy as np
import io
import os
from PIL import Image
from google.cloud import storage

class AttendanceService:
    """Service class for handling attendance-related operations"""
    
    def __init__(self):
        self.known_students = {}
        self.known_encodings = {}
        self.train_images_path = "train/"
        self._load_training_images()
        # Populate known_students from loaded encodings
        self._populate_known_students()
    
    def _load_training_images(self):
        """Load training images an
        d generate face encodings for known students"""
        print("🔄 Loading training images...")
        
        if not os.path.exists(self.train_images_path):
            print(f"❌ Training images directory '{self.train_images_path}' not found")
            return
        
        for filename in os.listdir(self.train_images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract student name from filename (remove extension)
                student_name = os.path.splitext(filename)[0].lower()
                
                try:
                    # Load image
                    image_path = os.path.join(self.train_images_path, filename)
                    image = face_recognition.load_image_file(image_path)
                    
                    # Generate face encoding
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        # Store the first face encoding found
                        self.known_encodings[student_name] = face_encodings[0]
                        print(f"✅ Loaded encoding for: {student_name}")
                    else:
                        print(f"⚠️  No face found in: {filename}")
                        
                except Exception as e:
                    print(f"❌ Error processing {filename}: {str(e)}")
        
        print(f"📚 Total students loaded: {len(self.known_encodings)}")

    def _populate_known_students(self):
        """Populate known_students dictionary based on loaded encodings"""
        if not self.known_encodings:
            print("⚠️  No encodings loaded, cannot populate known_students")
            return

        all_students = list(self.known_encodings.keys())

        self.known_students = {
            "class_123": all_students.copy(),
            "class_001": all_students.copy()
        }

        print(f"👥 Populated known_students with {len(all_students)} students for each class")

    def calculate_attendance_from_photo(self, image_data: bytes, class_id: str) -> Tuple[Dict[str, str], List[str], int, str]:
        """
        ML logic to calculate attendance from photo
        
        Args:
            image_data: Raw image bytes
            class_id: Class identifier
            
        Returns:
            Tuple of (attendance_dict, recognized_students, faces_detected, output_image_path)
        """
        
        # Convert to PIL Image and then to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_rgb = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Ensure image is in RGB format for face_recognition
        if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
            pass  # Already RGB
        elif len(image_rgb.shape) == 3 and image_rgb.shape[2] == 4:
            # Convert RGBA to RGB
            image_rgb = image_rgb[:, :, :3]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Get students for this class
        class_students = self.known_students.get(class_id, [])
        
        # Face detection and recognition
        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        
        recognized_students = []
        faces_detected = len(face_locations)
        
        # Real face recognition using trained encodings - matching your script exactly
        if faces_detected > 0 and self.known_encodings:
            print(f"🔍 Analyzing {faces_detected} detected faces...")
            
            # Convert to lists exactly like your script
            known_name_encodings = list(self.known_encodings.values())
            known_names = list(self.known_encodings.keys())
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_name_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_name_encodings, face_encoding)

                name = "Unknown"
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    if name not in recognized_students:
                        recognized_students.append(name)
                        print(f"✅ Recognized: {name} (confidence: {1-face_distances[best_match_index]:.2f})")

                # Draw bounding box and label - exactly like your script
                cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(image_bgr, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(image_bgr, name.capitalize(), (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        else:
            print("⚠️  No faces detected or no trained encodings available")
        
        # Save output image with bounding boxes
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_image_path = os.path.join(output_dir, f"output_{class_id}.jpg")
        cv2.imwrite(output_image_path, image_bgr)
        print(f"🖼️ Output image saved as {output_image_path}")
        
        # Create attendance dictionary
        attendance_dict = {}
        for student in class_students:
            attendance_dict[student] = "Present" if student in recognized_students else "Absent"
        
        # Print attendance for logging
        print(f"\n📋 Attendance for Class {class_id}:")
        for student, status in attendance_dict.items():
            print(f"  {student}: {status}")
        
        # Print attendance summary like your script
        print(f"\n📄 Attendance Summary:")
        for student in class_students:
            status = "Present" if student in recognized_students else "Absent"
            print(f"{student}: {status}")
        
        return attendance_dict, recognized_students, faces_detected, output_image_path
    
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
        attendance_dict, recognized_students, faces_detected, output_image_path = self.calculate_attendance_from_photo(
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
            "output_image_path": output_image_path,
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