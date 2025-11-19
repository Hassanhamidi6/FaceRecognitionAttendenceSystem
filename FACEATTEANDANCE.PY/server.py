# server.py
import base64
import threading
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import face_recognition
import os

from DbHandler import AttendanceDBHandler

# Configuration
DB_FILE = "EmployeeAttendance.db"
TOLERANCE = 0.48           # Euclidean threshold
COOLDOWN_SECONDS = 20     # seconds between DB updates for same person
MIN_WORK_SECONDS = 60 * 5  # 5 minutes minimum before check-out
IMAGE_SAVE_DIR = Path("employee_images")
IMAGE_SAVE_DIR.mkdir(exist_ok=True)

# App + DB
app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)
db = AttendanceDBHandler(DB_FILE)
db.create_table()

# in-memory
_known_encodings = []
_known_employee_ids = []
_known_names = []
_last_seen = {}
_lock = threading.Lock()

def load_known_encodings():
    global _known_encodings, _known_employee_ids, _known_names
    _known_encodings = []
    _known_employee_ids = []
    _known_names = []
    rows = db.get_all_encodings()
    for emp_id, enc_blob, name in rows:
        try:
            vec = np.frombuffer(enc_blob, dtype=np.float64)
            _known_encodings.append(vec)
            _known_employee_ids.append(emp_id)
            _known_names.append(name)
        except Exception as e:
            app.logger.warning("Failed to decode encoding for %s: %s", emp_id, e)
    app.logger.info("Loaded %d encodings", len(_known_encodings))
    return len(_known_encodings)

load_known_encodings()

def find_match(face_encoding):
    if len(_known_encodings) == 0:
        return None, None, None
    encs = np.stack(_known_encodings)
    dists = np.linalg.norm(encs - face_encoding, axis=1)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    if best_dist <= TOLERANCE:
        return _known_employee_ids[best_idx], _known_names[best_idx], best_dist
    return None, None, best_dist

def update_attendance_for_employee(employee_id):
    now = datetime.utcnow()
    today = now.date().isoformat()
    rows = db.get_attendance(employee_id=employee_id, date=today)
    if not rows:
        attendance_id = db.add_attendance_record(employee_id, today, check_in_time=now.isoformat(), check_out_time=None, status='Present')
        return "check_in", attendance_id, now.isoformat()

    attendance_id, emp_id, date, check_in_raw, check_out_raw, status = rows[0]
    check_in = None
    check_out = None
    try:
        if check_in_raw:
            check_in = datetime.fromisoformat(check_in_raw)
    except Exception:
        pass
    try:
        if check_out_raw:
            check_out = datetime.fromisoformat(check_out_raw)
    except Exception:
        pass

    if check_in and not check_out:
        delta = now - check_in
        if delta.total_seconds() >= MIN_WORK_SECONDS:
            db.update_attendance_times(attendance_id, check_out_time=now.isoformat(), status='Present')
            return "check_out", attendance_id, now.isoformat()
        else:
            db.update_attendance_times(attendance_id, check_in_time=now.isoformat(), status='Present')
            return "refresh_check_in", attendance_id, now.isoformat()
    else:
        db.update_attendance_times(attendance_id, check_out_time=now.isoformat(), status='Present')
        return "extend_session", attendance_id, now.isoformat()

def _image_bytes_from_payload(file_or_b64):
    if hasattr(file_or_b64, "read"):
        data = file_or_b64.read()
    elif isinstance(file_or_b64, str):
        b64 = file_or_b64
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
    else:
        raise ValueError("Unsupported image payload")
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img

# Admin endpoints (for admin.html)
@app.route("/register_employee", methods=["POST"])
def register_employee_endpoint():
    name = request.form.get("name") or request.form.get("Name")
    if not name:
        return jsonify({"error": "name required", "message": "name is required"}), 400
    email = request.form.get("email")
    phone = request.form.get("phone")
    position = request.form.get("position")
    img_file = request.files.get("photo") or request.files.get("image")
    img_b64 = request.form.get("image_b64")

    if not img_file and not img_b64:
        return jsonify({"error": "image required", "message": "image file or image_b64 required"}), 400

    try:
        img = _image_bytes_from_payload(img_file if img_file else img_b64)
    except Exception as e:
        return jsonify({"error": "failed to read image", "details": str(e)}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return jsonify({"error": "no face detected", "message": "No face detected in image"}), 400
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return jsonify({"error": "encoding_failed", "message": "Failed to compute encoding"}), 500

    emp_id = db.add_employee(name=name, email=email, phone=phone, position=position, join_date=None)
    image_path = IMAGE_SAVE_DIR / f"{emp_id}.jpg"
    cv2.imwrite(str(image_path), img)

    for enc in encodings:
        db.save_face_encoding(emp_id, np.array(enc, dtype=np.float64))

    load_known_encodings()
    return jsonify({"ok": True, "employee_id": emp_id, "encodings_saved": len(encodings), "message": "Registered"}), 201

@app.route("/employees", methods=["GET"])
def api_get_employees():
    rows = db.get_employees()
    out = []
    for r in rows:
        out.append({
            "employee_id": r[0],
            "name": r[1],
            "email": r[2],
            "phone": r[3],
            "position": r[4],
            "join_date": r[5],
            "created_at": r[6]
        })
    return jsonify(out)

@app.route("/attendance", methods=["GET"])
def api_get_attendance():
    employee_id = request.args.get("employee_id", type=int)
    date = request.args.get("date")  # YYYY-MM-DD
    rows = db.get_attendance(employee_id=employee_id, date=date)
    out = []
    for r in rows:
        out.append({
            "attendance_id": r[0],
            "employee_id": r[1],
            "date": r[2],
            "check_in_time": r[3],
            "check_out_time": r[4],
            "status": r[5]
        })
    return jsonify(out)

@app.route("/recognize", methods=["POST"])
def api_recognize():
    # accept multipart file or JSON base64
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        img_file = request.files.get("image")
        if not img_file:
            return jsonify({"error": "image file required"}), 400
        try:
            img = _image_bytes_from_payload(img_file)
        except Exception as e:
            return jsonify({"error": "failed to decode image file", "details": str(e)}), 400
    else:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "image (base64) is required in JSON body as {image: ...}"}), 400
        try:
            img = _image_bytes_from_payload(data["image"])
        except Exception as e:
            return jsonify({"error": "failed to decode base64 image", "details": str(e)}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model="hog")
    if not face_locations:
        return jsonify({"results": [], "message": "no faces detected"})

    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    results = []
    for enc in face_encodings:
        emp_id, name, dist = find_match(enc)
        if emp_id is None:
            results.append({
                "matched": False,
                "employee_id": None,
                "name": None,
                "distance": dist,
                "message": "unknown"
            })
            continue

        with _lock:
            last = _last_seen.get(emp_id)
            if last and (datetime.utcnow() - last).total_seconds() < COOLDOWN_SECONDS:
                results.append({
                    "matched": True,
                    "employee_id": emp_id,
                    "name": name,
                    "distance": dist,
                    "action": None,
                    "attendance_id": None,
                    "timestamp": datetime.utcnow().time().isoformat(),
                    "message": f"recently seen (cooldown {COOLDOWN_SECONDS}s)"
                })
                continue
            _last_seen[emp_id] = datetime.utcnow()

        action, attendance_id, ts = update_attendance_for_employee(emp_id)
        results.append({
            "matched": True,
            "employee_id": emp_id,
            "name": name,
            "distance": dist,
            "action": action,
            "attendance_id": attendance_id,
            "timestamp": ts,
            "message": "ok"
        })

    return jsonify({"results": results})

@app.route("/encodings/reload", methods=["POST"])
def reload_encodings():
    loaded = load_known_encodings()
    return jsonify({"ok": True, "loaded": loaded})

@app.route("/employee_images/<path:filename>", methods=["GET"])
def serve_images(filename):
    return send_from_directory(str(IMAGE_SAVE_DIR), filename)

# serve admin UI file from static folder
@app.route("/", methods=["GET"])
def serve_admin():
    return send_from_directory(app.static_folder, "admin.html")

if __name__ == "__main__":
    app.logger.info("Starting server, loading encodings...")
    load_known_encodings()
    app.run(host="0.0.0.0", port=5000, debug=True)
