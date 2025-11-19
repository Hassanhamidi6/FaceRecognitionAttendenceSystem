import cv2 
import face_recognition
import os
import numpy as np 
from DbHandler import AttendanceDBHandler 

def run_enrollment():
    """
    Loads employee images, generates face encodings, and saves them to the database.
    
    Pre-requisite: 
    1. EmployeeImages/ folder must exist.
    2. Image filenames must be the integer employee ID (e.g., '1.jpg', '105.png').
    """
    db = AttendanceDBHandler() 
    db.create_table()
    
    folderPath = "EmployeeImages"
    
    if not os.path.exists(folderPath):
        print(f"Error: Directory '{folderPath}' not found.")
        print("Please create this folder and place employee images inside (named by ID, e.g., 1.jpg).")
        return

    print("--- Starting Employee Enrollment Process ---")
    
    for filename in os.listdir(folderPath):
        # Check for image file extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        employee_id_str = os.path.splitext(filename)[0] 
        
        try:
            employee_id = int(employee_id_str)
            image_path = os.path.join(folderPath, filename)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Skipping {filename}: Could not load image.")
                continue

            # 1. Register Employee in DB (if not already there)
            employee_data = db.get_employee(employee_id)
            if not employee_data:
                # Assuming the ID is provided via the filename, we register with a placeholder name
                # You should ideally prompt for the name here in a real app
                db.add_employee(name=f"Employee {employee_id}", email=f"user{employee_id}@corp.com")

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
            print(f"Skipping file '{filename}'. Employee ID must be a valid integer.")
        except Exception as e:
            print(f"Critical error processing {filename}: {e}")

if __name__ == '__main__':
    run_enrollment()    