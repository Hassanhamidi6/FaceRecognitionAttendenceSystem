# app.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse
import pickle
import numpy as np
import cv2
import face_recognition
from datetime import datetime, timedelta
from DbHandler import AttendanceDB

app = FastAPI(title="Face Attendance with Face-Recognition")
db = AttendanceDB()

# Load encodings file generated earlier:
# expected format: [EmpIds, encodeListKnown]
with open("EncodeFile.p", "rb") as f:
    EmpIds_raw, encodeListKnown = pickle.load(f)

# Normalize EmpIds to int if possible, otherwise use as-is
EmpIds = []
for e in EmpIds_raw:
    try:
        EmpIds.append(int(e))
    except Exception:
        EmpIds.append(e)

print("Loaded encodings for", len(EmpIds), "employees.")

# Cooldown: avoid marking the same person multiple times in quick succession
RECOGNITION_COOLDOWN_SECONDS = 30
_last_recognized = {}  # emp_id -> datetime of last recognition (for check-in/check-out prevention)

# ----------------- face recognition util -----------------
def recognize_face_from_bytes(image_bytes, tolerance=0.48):
    """Return employee_id if recognized, else None.
       tolerance lowers for stricter matching. Default is stricter than face_recognition default.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_img)
        if not faces:
            return None
        encodes = face_recognition.face_encodings(rgb_img, faces)
        for enc in encodes:
            if len(encodeListKnown) == 0:
                continue
            dists = face_recognition.face_distance(encodeListKnown, enc)
            match_index = np.argmin(dists)
            if dists[match_index] <= tolerance:
                return EmpIds[match_index]
    except Exception as e:
        print("Recognition error:", e)
        return None
    return None

# ----------------- endpoints -----------------
@app.post("/attendance/upload")
async def attendance_upload(file: UploadFile = File(...), check_in: bool = Query(True)):
    image_bytes = await file.read()
    emp_id = recognize_face_from_bytes(image_bytes)
    if not emp_id:
        return JSONResponse({"status": "error", "message": "Face not recognized"}, status_code=404)

    # cooldown check
    now = datetime.now()
    last = _last_recognized.get(emp_id)
    if last and (now - last).total_seconds() < RECOGNITION_COOLDOWN_SECONDS:
        return {"status": "ignored", "message": f"Recently recognized. Cooldown applied for {RECOGNITION_COOLDOWN_SECONDS}s."}

    _last_recognized[emp_id] = now

    if check_in:
        res = db.mark_check_in(emp_id)
    else:
        res = db.mark_check_out(emp_id)
    return {"status": "success", "employee_id": emp_id, "result": res}

# ----------------- live webcam streaming -----------------
def video_generator(check_in=True, tolerance=0.48):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)  # speed up
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_small)
            encs = face_recognition.face_encodings(rgb_small, faces)

            for enc, face_loc in zip(encs, faces):
                if len(encodeListKnown) == 0:
                    continue
                dists = face_recognition.face_distance(encodeListKnown, enc)
                match_index = np.argmin(dists)
                if dists[match_index] <= tolerance:
                    emp_id = EmpIds[match_index]
                    now = datetime.now()
                    last = _last_recognized.get(emp_id)
                    if not last or (now - last).total_seconds() > RECOGNITION_COOLDOWN_SECONDS:
                        # mark attendance
                        if check_in:
                            db.mark_check_in(emp_id)
                        else:
                            db.mark_check_out(emp_id)
                        _last_recognized[emp_id] = now
                        print(f"Marked {('check-in' if check_in else 'check-out')} for {emp_id} at {now.isoformat()}")

                    # draw bounding box on original frame (scale face_loc coords)
                    top, right, bottom, left = [v*2 for v in face_loc]  # scale back
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{emp_id}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # encode frame
            ret2, buf = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.get("/live_attendance")
def live_attendance(check_in: bool = True):
    return StreamingResponse(video_generator(check_in=check_in), media_type='multipart/x-mixed-replace; boundary=frame')

# ----------------- reporting endpoints -----------------
@app.get("/employee/{employee_id}/attendance")
def get_employee_attendance(employee_id: int):
    rows = db.get_attendance_by_employee(employee_id)
    cols = ["attendance_id","employee_id","check_in_time","check_out_time","total_hours","status"]
    return [dict(zip(cols, r)) for r in rows]

@app.get("/employee/{employee_id}/summary")
def get_employee_summary(employee_id: int):
    row = db.get_employee_summary(employee_id)
    if not row:
        return {}
    cols = ["employee_id","total_attendance","last_check_in","last_check_out","leaves_taken","total_hours_worked"]
    return dict(zip(cols, row))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
