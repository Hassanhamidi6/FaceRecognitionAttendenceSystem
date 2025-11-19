from databasehandler import DatabaseHandler
from datetime import datetime

def print_attendance_report():
    db = DatabaseHandler()
    
    # 1. Get Employee ID
    while True:
        try:
            employee_id = int(input("Enter Employee ID to view attendance: "))
            employee = db.get_employee(employee_id)
            if employee:
                employee_name = employee[1]
                break
            else:
                print(f"Employee ID {employee_id} not found.")
        except ValueError:
            print("Invalid input. Please enter a valid integer ID.")

    # 2. Get Date Range (optional)
    print("\n--- Optional Date Range (Leave blank for all time) ---")
    start_date_input = input("Enter Start Date (YYYY-MM-DD) or press Enter: ")
    end_date_input = input("Enter End Date (YYYY-MM-DD) or press Enter: ")

    start_date = start_date_input if start_date_input else None
    end_date = end_date_input if end_date_input else None
    
    # 3. Fetch Data
    print(f"\n--- Generating Attendance Report for {employee_name} (ID: {employee_id}) ---")
    
    history = db.get_employee_attendance_history(employee_id, start_date, end_date)
    
    if not history:
        print("No attendance records found for this employee in the specified range.")
        return

    # 4. Print Report
    print("-------------------------------------------------------------------------")
    print(f"{'Date':<12}{'Check-In':<10}{'Check-Out':<10}{'Status':<10}{'Total Hours':>15}")
    print("-------------------------------------------------------------------------")

    total_hours_sum = 0
    for record in history:
        # Extract and format date/time
        check_in_dt = datetime.strptime(record['check_in_time'].split(".")[0], "%Y-%m-%d %H:%M:%S")
        date = check_in_dt.strftime("%Y-%m-%d")
        check_in_time = check_in_dt.strftime("%H:%M")
        
        # Handle null checkout time (currently checked-in)
        if record['check_out_time']:
            check_out_dt = datetime.strptime(record['check_out_time'].split(".")[0], "%Y-%m-%d %H:%M:%S")
            check_out_time = check_out_dt.strftime("%H:%M")
            total_hours = record['total_hours']
            total_hours_sum += total_hours
            total_hours_str = f"{total_hours:.2f}"
        else:
            check_out_time = "N/A"
            total_hours_str = "N/A"
            
        print(f"{date:<12}{check_in_time:<10}{check_out_time:<10}{record['status']:<10}{total_hours_str:>15}")
    
    print("-------------------------------------------------------------------------")
    print(f"{'TOTAL HOURS (Completed Sessions):':<42}{total_hours_sum:>15.2f}")
    print("-------------------------------------------------------------------------")


if __name__ == '__main__':
    print_attendance_report()