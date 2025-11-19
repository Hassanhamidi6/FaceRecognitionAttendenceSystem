# db_handler.py
import sqlite3
from datetime import datetime
import numpy as np
from pathlib import Path
import typing

DB_DEFAULT = "EmployeeAttendance.db"

class AttendanceDBHandler:
    def __init__(self, db_name: str = DB_DEFAULT):
        self.db_name = db_name
        Path(self.db_name).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        conn = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()
        return conn, cursor

    def create_table(self):
        conn, cursor = self._connect()
        cursor.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS Employees (
            employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            position TEXT,
            join_date DATE,
            shift TEXT,
            created_at DATETIME DEFAULT (CURRENT_TIMESTAMP)
        );

        CREATE TABLE IF NOT EXISTS FaceEncodings (
            encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            encoding_vector BLOB NOT NULL,
            created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS Attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            date DATE NOT NULL,
            check_in_time DATETIME,
            check_out_time DATETIME,
            status TEXT DEFAULT 'Absent',
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE,
            UNIQUE(employee_id, date)
        );

        CREATE TABLE IF NOT EXISTS Weekends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_name TEXT UNIQUE CHECK(day_name IN ('Saturday','Sunday'))
        );

        CREATE TABLE IF NOT EXISTS Leaves (
            leave_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            leave_type TEXT NOT NULL,
            start_date DATE,
            end_date DATE,
            status TEXT DEFAULT 'Pending',
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
        );
        """)
        cursor.execute("INSERT OR IGNORE INTO Weekends (day_name) VALUES ('Saturday')")
        cursor.execute("INSERT OR IGNORE INTO Weekends (day_name) VALUES ('Sunday')")
        conn.commit()
        conn.close()

    # DataBase Operations
    def add_employee(self, name: str, email: str = None, phone: str = None, position: str = None, join_date: str = None) -> int:
        conn, cursor = self._connect()
        cursor.execute(
            "INSERT INTO Employees (Name, email, phone, position, join_date) VALUES (?, ?, ?, ?, ?)",
            (name, email, phone, position, join_date)
        )
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id

    def get_employees(self):
        conn, cursor = self._connect()
        cursor.execute("SELECT employee_id, Name, email, phone, position, join_date, created_at FROM Employees ORDER BY Name")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_employee(self, employee_id: int):
        conn, cursor = self._connect()
        cursor.execute("SELECT employee_id, Name, email, phone, position, join_date, created_at FROM Employees WHERE employee_id = ?", (employee_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    # FaceEncodings
    def save_face_encoding(self, employee_id: int, encoding_vector: np.ndarray) -> int:
        if not isinstance(encoding_vector, np.ndarray):
            raise ValueError("encoding_vector must be a numpy.ndarray")
        blob = encoding_vector.astype(np.float64).tobytes()
        conn, cursor = self._connect()
        cursor.execute("INSERT INTO FaceEncodings (employee_id, encoding_vector) VALUES (?, ?)", (employee_id, blob))
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id

    def get_all_encodings(self) -> typing.List[typing.Tuple[int, bytes, str]]:
        conn, cursor = self._connect()
        cursor.execute("""
            SELECT f.employee_id, f.encoding_vector, e.Name
            FROM FaceEncodings f
            JOIN Employees e ON e.employee_id = f.employee_id
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_face_encoding(self, employee_id: int) -> typing.Optional[np.ndarray]:
        conn, cursor = self._connect()
        cursor.execute("SELECT encoding_vector FROM FaceEncodings WHERE employee_id = ? LIMIT 1", (employee_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return np.frombuffer(row[0], dtype=np.float64)
        return None

    # Attendance
    def add_attendance_record(self, employee_id: int, date: str, check_in_time: str = None, check_out_time: str = None, status: str = 'Present') -> int:
        conn, cursor = self._connect()
        cursor.execute("INSERT OR IGNORE INTO Attendance (employee_id, date, check_in_time, check_out_time, status) VALUES (?, ?, ?, ?, ?)",
                       (employee_id, date, check_in_time, check_out_time, status))
        cursor.execute("SELECT attendance_id FROM Attendance WHERE employee_id = ? AND date = ?", (employee_id, date))
        row = cursor.fetchone()
        attendance_id = row[0]
        conn.commit()
        conn.close()
        return attendance_id

    def update_attendance_times(self, attendance_id: int, check_in_time: str = None, check_out_time: str = None, status: str = None):
        conn, cursor = self._connect()
        parts = []
        params = []
        if check_in_time is not None:
            parts.append("check_in_time = ?")
            params.append(check_in_time)
        if check_out_time is not None:
            parts.append("check_out_time = ?")
            params.append(check_out_time)
        if status is not None:
            parts.append("status = ?")
            params.append(status)
        params.append(attendance_id)
        if parts:
            sql = "UPDATE Attendance SET " + ", ".join(parts) + " WHERE attendance_id = ?"
            cursor.execute(sql, params)
            conn.commit()
        conn.close()

    def get_attendance(self, employee_id: int = None, date: str = None):
        conn, cursor = self._connect()
        q = "SELECT attendance_id, employee_id, date, check_in_time, check_out_time, status FROM Attendance"
        params = []
        conds = []
        if employee_id is not None:
            conds.append("employee_id = ?")
            params.append(employee_id)
        if date is not None:
            conds.append("date = ?")
            params.append(date)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY date DESC"
        cursor.execute(q, params)
        rows = cursor.fetchall()
        conn.close()
        return rows

    # Weekends / Leaves (small helpers)
    def get_weekends(self):
        conn, cursor = self._connect()
        cursor.execute("SELECT id, day_name FROM Weekends")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def add_leave_request(self, employee_id: int, leave_type: str, start_date: str, end_date: str, status: str = 'Pending'):
        conn, cursor = self._connect()
        cursor.execute("INSERT INTO Leaves (employee_id, leave_type, start_date, end_date, status) VALUES (?, ?, ?, ?, ?)",
                       (employee_id, leave_type, start_date, end_date, status))
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return last_id
