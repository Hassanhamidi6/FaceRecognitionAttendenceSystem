import cv2
import os
import numpy as np
import pickle
import face_recognition
from firebase_admin import credentials
from firebase_admin import db

cap = cv2.VideoCapture(0)

# importing the ui images 
folderPath = "Resources"
imgList = []

for path in os.listdir(folderPath):
    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)

print(f"Total Images Loaded: {len(imgList)}")

# loading the encodings file from pickle

print("Loading Encodings file...")
file = open("EncodeFile.p", 'rb')
encodeListKnownwithIds = pickle.load(file)
file.close()
EmpIds, encodeListKnown = encodeListKnownwithIds
# print(StdIds)
print("Encodings Loaded")

modetype = 0
counter = 0
id = -1

while True:
    success, img = cap.read()
    if not success:
        break
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    imgS = cv2.flip(imgS, 1)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    cv2.imshow("Face Attendance", img)  

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("faceDis", faceDis)
        print("matches", matches)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            empId = EmpIds[matchIndex]
            print(f"Employee ID: {empId}")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # '*4' rescaling back to original size because we scaled down img above
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {empId}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            id = empId
            # print(f"Matched ID: {id}")

            if counter == 0:
                counter = 1
                # Here, you can add code to mark attendance in the database
                print(f"Attendance marked for ID: {empId}")

    if counter != 0:
        if counter ==1: 
            employeeinfo = db.reference(f'Employees/{id}').get()
            print(employeeinfo)
        


        counter +=1
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()