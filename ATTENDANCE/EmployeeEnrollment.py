import cv2 
import face_recognition
import os
import numpy as np 
from databasehandler import DatabaseHandler 

def run_enrollment():
    db = DatabaseHandler() 
    db.create_table()
    
    folderPath = "EmployeeImages"
    
    if not os.path.exists(folderPath):
        print("Please create this folder and place employee images inside (named by ID, e.g., 1.jpg).")
        return

    print("--- Starting Employee Enrollment Process ---")
    
    for image in os.listdir(folderPath):
        if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        employee_id_str = os.path.splitext(image)[0] 
        
        try:
            employee_id = int(employee_id_str)
            image_path = os.path.join(folderPath, image)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Skipping {image}: Could not load image.")
                continue

            # 1. Register Employee in DB (if not already there)
            employee_data = db.get_employee(employee_id)
            if not employee_data:
                # You should ideally prompt for the name here in a real app
                db.add_employee(name=f"Employee {employee_id}", email=f"user{employee_id}@gmail.com", phone="N/A", position="N/A", join_date="N/A", salary=0, shift_start_time="N/A", shift_end_time="N/A", leave_id=None, status='Active')

            # 2. Find and Save Encoding
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            
            if encodings:
                encoding = encodings[0]
                # Save encoding directly to the database
                if db.save_face_encoding(employee_id, encoding):
                    print(f"[SUCCESS] Encoded and saved ID: {employee_id}")
            else:
                print(f"⚠️ [WARNING] No face found in image for ID: {employee_id}. Skipping encoding.")
                
        except ValueError:
            print(f"Skipping file '{image}'. Employee ID must be a valid integer.")
        except Exception as e:
            print(f"Critical error processing {image}: {e}")
            
    print("\n--- Enrollment Complete. Run AttendanceTracker.py next. ---")

if __name__ == '__main__':
    run_enrollment()    