import os
import cv2
import time
import pickle
import numpy as np
from datetime import datetime
from DbHandler import DatabaseHandler
import face_recognition


# ---------------- CONFIG ----------------
SCALE_FACTOR = 0.25
TOLERANCE = 0.48
COOLDOWN_SECONDS = 30
ENCODINGS_PATH = "EncodeFile.p"

# Initialize database
db = DatabaseHandler()

# ---------------- LOAD ENCODINGS ----------------
def load_encodings(path=ENCODINGS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encodings file not found: {path}")
    with open(path, "rb") as f:
        EmpIds, encodeListKnown = pickle.load(f)
    print(f"[Encodings] Loaded {len(EmpIds)} IDs.")
    return EmpIds, encodeListKnown

EmpIds, encodeListKnown = load_encodings()

# Track last seen for cooldown
last_seen = {}

# ---------------- ATTENDANCE LOGIC ----------------
def mark_local_attendance(emp_id):
    emp = db.get_employee(emp_id)
    if not emp:
        print(f"[Attendance] ❌ Employee {emp_id} not found in DB.")
        return False

    history = db.get_attendance_history(emp_id)
    now = datetime.now()
    if not history or (history and history[0][2] is not None):
        db.mark_check_in(emp_id)
        return "Check-in"
    else:
        db.mark_check_out(emp_id)
        return "Check-out"

# ---------------- FACE RECOGNITION ----------------
def recognize_face_from_frame(frame):
    """
    Receives a frame (numpy array) and returns matched employee_id (if any).
    """
    global last_seen

    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small, model="hog")
    encodings = face_recognition.face_encodings(rgb_small, face_locations)

    if not encodings:
        return None, None  # No face detected

    encodeFace = encodings[0]  # single face for now
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=TOLERANCE)
    face_distances = face_recognition.face_distance(encodeListKnown, encodeFace)
    if len(face_distances) == 0:
        return None, None

    match_index = np.argmin(face_distances)
    if matches[match_index]:
        emp_id = EmpIds[match_index]

        now_ts = time.time()
        if emp_id not in last_seen or now_ts - last_seen[emp_id] > COOLDOWN_SECONDS:
            action = mark_local_attendance(emp_id)
            last_seen[emp_id] = now_ts
            return emp_id, action
    return None, None

# # Optional: test with webcam
# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     print("[System] Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         emp_id, action = recognize_face_from_frame(frame)
#         if emp_id:
#             print(f"Employee {emp_id} - {action}")
            
#         cv2.imshow("Frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
