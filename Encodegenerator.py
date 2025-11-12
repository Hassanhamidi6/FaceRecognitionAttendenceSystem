import cv2 
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db


cred = credentials.Certificate("C:\\Users\\User\\Downloads\\faceattendanceinrealtime.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://faceattendanceinrealtime-1278c-default-rtdb.firebaseio.com/',
    'storageBucket': 'faceattendanceinrealtime-1278c.appspot.com'  #here we will write storage bucket URL of our firebase project
})  


# importing the Employee images

folderPath = "EmployeeImages"
imgList = []
EmpIds = []

for path in os.listdir(folderPath):
    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)
    EmpIds.append(os.path.splitext(path)[0])
    # print(path.split('.')[0])

    # Upload images to Firebase Storage
    filename = os.path.join(folderPath, path)
    bucket = storage.bucket()
    blob = bucket.blob(f'StudentImages/{os.path.basename(filename)}')
    blob.upload_from_filename(filename)
    print(f'Uploaded {filename} to Firebase Storage.')

print(EmpIds) 

# Function to find encodings
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownwithIds = [EmpIds, encodeListKnown]
# print(encodeListKnown)
print("Encoding Complete")

# Save the encodings and IDs using pickle
file = open("EncodeFile.p", 'wb') 
pickle.dump(encodeListKnownwithIds, file)
file.close()

print("File Saved")