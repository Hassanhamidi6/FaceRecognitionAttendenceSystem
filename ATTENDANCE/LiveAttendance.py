import cv2
import face_recognition
import numpy as np
from databasehandler import DatabaseHandler
import time
from datetime import datetime


FACE_RECOGNITION_TOLERANCE = 0.50
COOLDOWN_PERIOD_SECONDS = 15 

def run_attendance_system():
    db = DatabaseHandler()
    db.create_table()
    
    # Load all registered face data from the database
    known_face_encodings, known_ids, known_names = db.load_all_encodings()
    
    if not known_face_encodings:
        print("Please run EnrollmentScript.py first to register employees.")
        return

    print(f"Successfully loaded {len(known_face_encodings)} known employees for recognition.")
    
    # Dictionary to track when an employee last interacted (to prevent rapid check-in/out spam)
    last_action_time = {} 
    
    # Start video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Resize frame for faster processing (1/4 size)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces and their encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            
            # Compare the current face with all known encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, FACE_RECOGNITION_TOLERANCE)
            
            name = "Unknown"
            employee_id = None
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                employee_id = known_ids[best_match_index]
                name = known_names[best_match_index]
                
                # --- Attendance Logic ---
                current_time = time.time()
                
                # Check for cooldown period
                if employee_id not in last_action_time or (current_time - last_action_time[employee_id]) > COOLDOWN_PERIOD_SECONDS:
                    
                    # 1. Check if employee has an active Check-In for today
                    latest_record = db.get_latest_attendance(employee_id)
                    
                    if latest_record:
                        # User is currently checked in -> Mark Check-Out
                        message = db.mark_check_out(employee_id)
                        print(f"-> {name} ({employee_id}): {message}")
                        
                    else:
                        # User is checked out or absent -> Mark Check-In
                        message = db.mark_check_in(employee_id)
                        print(f"-> {name} ({employee_id}): {message}")
                        
                    # Update cooldown timestamp
                    last_action_time[employee_id] = current_time
                    
                else:
                    # On Cooldown
                    time_left = int(COOLDOWN_PERIOD_SECONDS - (current_time - last_action_time[employee_id]))
                    print(f"  [COOLDOWN] {name} ({employee_id}) - Wait {time_left}s...")

            # Rescale the box coordinates back to the original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw the box and label on the original frame
            color = (0, 255, 0) if employee_id else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)


        cv2.imshow('Face Recognition Attendance - Press Q to Exit', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    print("Attendance System Shutdown.")


if __name__ == '__main__':
    run_attendance_system()