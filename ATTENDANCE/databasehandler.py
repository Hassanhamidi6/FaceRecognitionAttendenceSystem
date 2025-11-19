import sqlite3
import os 
from datetime import datetime, timedelta
import pickle # Used for serializing NumPy arrays
import numpy as np 

class DatabaseHandler:
    """
    Manages all SQLite database operations for the attendance system, 
    including Employees, Attendance, and FaceEncodings.
    """
    def __init__(self, db_name = "Attendance.db"):
        self.db_name = db_name

    def _connect(self):
        """Establishes a connection and enables foreign keys."""
        conn = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        return conn ,cursor

    ## 🛠️ DB Setup & Schemas
    # ----------------------------------------------------
    
    def create_table(self):
        """Creates all necessary tables (Employees, Attendance, FaceEncodings, etc.)."""
        conn, cursor = self._connect()
        print("Creating Tables...")

        # SQL statements combined into one block
        queries  = """
        -- ===========================
        -- EMPLOYEES TABLE (Personnel Info)
        -- ===========================
        CREATE TABLE IF NOT EXISTS Employees (
            employee_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            position TEXT,
            email TEXT UNIQUE,
            phone_number TEXT,
            salary REAL,
            office_join_date DATE,
            shift_start_time TIME,
            shift_end_time TIME,
            leave_id INTEGER,
            status TEXT DEFAULT 'Active'
        );

        -- ===========================
        -- FACE ENCODINGS TABLE (Biometric Data)
        -- Stores face encodings as BLOB (serialized NumPy array)
        -- ===========================
        CREATE TABLE IF NOT EXISTS FaceEncodings (
            employee_id INTEGER PRIMARY KEY,
            encoding_data BLOB NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
        );

        -- ===========================
        -- ATTENDANCE TABLE (Daily Records)
        -- total_hours is calculated on check_out
        -- ===========================
        CREATE TABLE IF NOT EXISTS Attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            check_in_time DATETIME,
            check_out_time DATETIME,
            total_hours REAL, 
            status TEXT DEFAULT 'Absent', 
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
        );
        -- ===========================
        -- LEAVE RECORDS TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS LeaveRecords (
            leave_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            leave_type TEXT NOT NULL,
            start_date DATE,
            end_date DATE,
            reason TEXT,
            status TEXT DEFAULT 'Pending',
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id) ON DELETE CASCADE
        );
        """
        cursor.executescript(queries)
        conn.commit()
        conn.close()
        print("Tables created and initialized.")


    # ADD DATA METHODS
    # ----------------------------------------------------
    
    def add_employee(self, name, email=None, phone=None, position=None, join_date=None, salary=None, shift_start_time=None, shift_end_time=None, leave_id=None, status='Active'):
        """Adds a new employee record and returns the new employee_id."""
        conn, cursor = self._connect()
        try:
            cursor.execute("""
                INSERT INTO Employees (name, email, office_join_date, phone_number, position, salary, shift_start_time, shift_end_time, leave_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, email, datetime.now().date(), phone, position,join_date, salary, shift_start_time, shift_end_time, leave_id, status))
            conn.commit()
            new_id = cursor.lastrowid
            print(f"Employee '{name}' added with ID: {new_id}.")
            return new_id
        except sqlite3.IntegrityError as e:
            print(f"Error adding employee: {e}. Check for duplicate email.")
            return None
        finally:
            conn.close()
    
    def save_face_encoding(self, employee_id, encoding_array):
        """Saves a face encoding (NumPy array) associated with an employee ID."""
        conn, cursor = self._connect()
        # Serialize the numpy array into a binary format (BLOB)
        serialized_encoding = pickle.dumps(encoding_array)
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO FaceEncodings (employee_id, encoding_data)
                VALUES (?, ?)
            """, (employee_id, serialized_encoding))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error saving encoding for {employee_id}: {e}")
            return False
        finally:
            conn.close()
    

    
    # FETCH METHODS
    # ----------------------------------------------------

    def get_employees(self):
        """Fetches all employees."""
        conn, cursor = self._connect()
        cursor.execute("SELECT employee_id, name FROM Employees")
        employees = cursor.fetchall()
        conn.close()
        return employees

    def get_employee(self, employee_id):
        """Fetches a single employee by ID."""
        conn, cursor = self._connect()
        cursor.execute("SELECT* FROM Employees WHERE employee_id = ?", (employee_id,))
        employee = cursor.fetchone()
        conn.close()
        return employee
    
    
    def load_all_encodings(self):
        """
        Loads all employee IDs, names, and their face encodings.
        Returns three parallel lists for use with face_recognition.
        """
        conn, cursor = self._connect()
        cursor.execute("""
            SELECT E.employee_id, E.name, F.encoding_data 
            FROM Employees E
            INNER JOIN FaceEncodings F ON E.employee_id = F.employee_id
        """)
        results = cursor.fetchall()
        conn.close()
        
        known_face_encodings = []
        known_ids = []
        known_names = []
        
        for employee_id, name, blob_data in results:
            try:
                # Deserialize the BLOB back into a NumPy array
                encoding = pickle.loads(blob_data)
                known_face_encodings.append(encoding)
                known_ids.append(employee_id)
                known_names.append(name)
            except Exception as e:
                print(f"Failed to load encoding for ID {employee_id}: {e}")
                
        return known_face_encodings, known_ids, known_names

    ## ⏰ Attendance Tracking
    # ----------------------------------------------------

    def get_latest_attendance(self, employee_id):
        """Fetches the latest un-checked-out attendance record for an employee."""
        conn, cursor = self._connect()
        # Look for a record from TODAY that is missing a check_out_time
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT attendance_id, check_in_time 
            FROM Attendance 
            WHERE employee_id = ? 
              AND check_out_time IS NULL 
              AND DATE(check_in_time) = ?
            ORDER BY check_in_time DESC LIMIT 1
        """, (employee_id, today))
        latest_record = cursor.fetchone()
        conn.close()
        return latest_record # (attendance_id, check_in_time_str)

    def calculate_total_hours(self, check_in_str, check_out_str):
        """Calculates the difference in hours between two DATETIME strings."""
        time_format = "%Y-%m-%d %H:%M:%S"
        try:
            check_in_dt = datetime.strptime(check_in_str.split(".")[0], time_format)
            check_out_dt = datetime.strptime(check_out_str.split(".")[0], time_format)
        except ValueError:
            # Fallback for times without milliseconds
            check_in_dt = datetime.strptime(check_in_str, time_format)
            check_out_dt = datetime.strptime(check_out_str, time_format)
            
        time_difference = check_out_dt - check_in_dt
        total_hours = time_difference.total_seconds() / 3600
        return round(total_hours, 2)


    def mark_check_in(self, employee_id):
        """Records a check-in time for an employee if not already checked in today."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if the employee already has an unclosed check-in today
        if self.get_latest_attendance(employee_id):
            return "Already Checked In"

        conn, cursor = self._connect()
        try:
            cursor.execute("INSERT INTO Attendance (employee_id, check_in_time, status) VALUES (?, ?, ?)", 
                           (employee_id, now, 'Present'))
            conn.commit()
            return f"Check-In successful at {now.split(' ')[1]}"
        except sqlite3.Error as e:
            return f"Error during Check-In: {e}"
        finally:
            conn.close()

    def mark_check_out(self, employee_id):
        """Records a check-out time, calculates total hours, and closes the record."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        latest_record = self.get_latest_attendance(employee_id)
        
        if not latest_record:
            return "No pending Check-In found for today."
            
        attendance_id, check_in_time_str = latest_record
        total_hours = self.calculate_total_hours(check_in_time_str, now)
        
        conn, cursor = self._connect()
        try:
            cursor.execute("""
                UPDATE Attendance 
                SET check_out_time = ?, 
                    total_hours = ?,
                    status = 'Completed'
                WHERE attendance_id = ?
            """, (now, total_hours, attendance_id))
            conn.commit()
            return f"Check-Out successful. Hours: {total_hours:.2f}h"
        except sqlite3.Error as e:
            return f"Error during Check-Out: {e}"
        finally:
            conn.close()
            
    def get_employee_attendance_history(self, employee_id, start_date=None, end_date=None):
        """
        Retrieves attendance records for a given employee, optionally filtered by date range.
        Dates should be in 'YYYY-MM-DD' format.
        """
        conn, cursor = self._connect()
        
        # Base query to join Employee and Attendance data
        query = """
            SELECT A.check_in_time, A.check_out_time, A.total_hours, A.status, E.name
            FROM Attendance A
            JOIN Employees E ON A.employee_id = E.employee_id
            WHERE A.employee_id = ?
        """
        params = [employee_id]
        
        # Add date filtering if dates are provided
        if start_date:
            query += " AND DATE(A.check_in_time) >= ?"
            params.append(start_date)
        if end_date:
            query += " AND DATE(A.check_in_time) <= ?"
            params.append(end_date)
            
        query += " ORDER BY A.check_in_time DESC"

        try:
            cursor.execute(query, tuple(params))
            records = cursor.fetchall()
            
            # Get column names for easier data handling
            columns = [desc[0] for desc in cursor.description]
            
            # Convert list of tuples to list of dictionaries
            history = [dict(zip(columns, row)) for row in records]
            return history
            
        except sqlite3.Error as e:
            print(f"Error retrieving attendance history for ID {employee_id}: {e}")
            return []
        finally:
            conn.close()

