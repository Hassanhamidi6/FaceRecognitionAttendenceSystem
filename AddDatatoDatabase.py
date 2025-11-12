import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("C:\\Users\\User\\Downloads\\faceattendanceinrealtime.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceattendanceinrealtime-1278c-default-rtdb.firebaseio.com/'
})  

ref = db.reference('Employees')
data = {
    '01':{
        'name': 'Shahbaz Hamidi',
        'position': 'Software Engineer',
        'starting_date': '2022-01-15',
        'phone': '123-456-7890',
        'email': 'shahbaz.hamidi`@example.com',
        'last_attendance': '2025-11-12 09:00:00'
    },  

    '02':{
        'name': 'Ameer Hamza',
        'position': 'Backend Developer',
        'starting_date': '2023-03-22',
        'phone': '987-654-3210',
        'email': 'ameer.hamza@example.com',
        'last_attendance': '2025-11-12 09:05:00'
    },

    '03':{
        'name': 'Hassan Hamidi',
        'position': 'Data Scientist',
        'starting_date': '2021-07-30',
        'phone': '555-123-4567',
        'email': 'hassan.hamidi@example.com',
        'last_attendance': '2025-11-12 09:10:00'
    },
}

for key, value in data.items():
    ref.child(key).set(value)