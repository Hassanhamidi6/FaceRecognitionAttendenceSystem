import sqlite3
import os 
from datetime import datetime 

class DatabaseHandler:
    def __init__(self, db_name = "Attendace.db"):
        self.db_name = db_name

    def _connect(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        return conn ,cursor

    # DB creation 
    def create_table(self):
        conn, cursor = self._connect()
        print("Creating Tables ...")

        queries  = [
            """
        -- ===========================
        -- EMPLOYEES TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS Employees (
            employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            position TEXT,
            email TEXT UNIQUE,
            phone_number TEXT,
            salary REAL,
            office_join_date DATE,
            shift_id INTEGER,
            status TEXT DEFAULT 'Active'
        );

        -- ===========================
        -- SHIFTS TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS Shifts (
            shift_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shift_name TEXT,
            start_time TIME,
            end_time TIME,
            working_days TEXT DEFAULT 'Mon,Tue,Wed,Thu,Fri,Sat'
        );

        -- ===========================
        -- WEEKENDS TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS Weekends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_name TEXT UNIQUE CHECK(day_name IN ('Saturday','Sunday'))
        );

        -- Default weekends
        INSERT OR IGNORE INTO Weekends (day_name) VALUES ('Saturday');
        INSERT OR IGNORE INTO Weekends (day_name) VALUES ('Sunday');

        -- ===========================
        -- LEAVE TYPES TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS LeaveTypes (
            leave_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
            leave_type TEXT UNIQUE NOT NULL CHECK(leave_type IN ('Sick','Casual','Annual','Maternity','others'))
        );

        -- ===========================
        -- LEAVE RECORDS TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS LeaveRecords (
            leave_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            leave_type_id INTEGER NOT NULL,
            start_date DATE,
            end_date DATE,
            total_days INTEGER,
            status TEXT DEFAULT 'Pending',
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id),
            FOREIGN KEY (leave_type_id) REFERENCES LeaveTypes(leave_type_id)
        );

        -- ===========================
        -- ATTENDANCE TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS Attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            check_in_time DATETIME,
            check_out_time DATETIME,
            total_hours REAL,   
            status TEXT DEFAULT 'Absent',
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id)
        );

        -- ===========================
        -- SALARY TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS Salary (
            salary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            base_salary REAL,
            overtime_hours REAL DEFAULT 0,
            overtime_rate REAL DEFAULT 0,
            total_salary REAL,
            month_year TEXT,
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id)
        );

        -- ===========================
        -- EMPLOYEE SUMMARY TABLE
        -- ===========================
        CREATE TABLE IF NOT EXISTS EmployeeSummary (
            employee_id INTEGER PRIMARY KEY,
            total_attendance INTEGER DEFAULT 0,
            last_check_in DATETIME,
            last_check_out DATETIME,
            last_attendance_date DATETIME,
            leaves_taken INTEGER DEFAULT 0,
            FOREIGN KEY (employee_id) REFERENCES Employees(employee_id)
        );
        """
        ]
        for q in queries:
            cursor.execute(q)
        conn.commit()
        conn.close()

    # ---------------- fetch employee by id ----------------
    def get_employees(self):
        conn, cursor = self._connect()
        cursor.execute("SELECT * FROM Employees")
        employees = cursor.fetchall()
        conn.close()
        return employees
    
    # ---------------- fetch employee by id ----------------
    def get_employee(self, employee_id):
        conn, cursor = self._connect()
        cursor.execute("SELECT * FROM Employees WHERE employee_id = ?", (employee_id,))
        employee = cursor.fetchone()
        conn.close()    
        return employee
    # ---------------- fetch attendance history ----------------
    def get_attendance_history(self, employee_id):
        conn, cursor = self._connect()
        cursor.execute("SELECT * FROM Attendance WHERE employee_id = ? ORDER BY check_in_time DESC", (employee_id,))
        history = cursor.fetchall()
        conn.close()    
        return history
    # ---------------- mark check-in ----------------
    def mark_check_in(self, employee_id):
        conn, cursor = self._connect()
        now = datetime.now()
        day_name = now.strftime("%A")
        cursor.execute("INSERT INTO Attendance (employee_id, check_in_time, day_name) VALUES (?, ?, ?)", (employee_id, now, day_name))
        conn.commit()
        conn.close()