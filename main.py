import cv2
import os
import numpy as np
import pickle
import face_recognition


cap = cv2.VideoCapture(0)

# importing the ui images 
folderPath = "Resources"
imgList = []

for path in os.listdir(folderPath):
    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)

print(f"Total Images Loaded: {len(imgList)}")

# loading the encodings file

print("Loading Encodings file...")
file = open("EncodeFile.p", 'rb')
encodeListKnownwithIds = pickle.load(file)
file.close()
StdIds, encodeListKnown = encodeListKnownwithIds
# print(StdIds)
print("Encodings Loaded")

while True:
    success, img = cap.read()
    if not success:
        break
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("faceDis", faceDis)
        print("matches", matches)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            stdId = StdIds[matchIndex]
            print(f"Student ID: {stdId}")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # '*4' rescaling back to original size because we scaled down img above
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {stdId}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        

    cv2.imshow("Face Attendance", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()