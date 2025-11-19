import cv2 
import face_recognition
import pickle
import os

# Path to the folder containing employee images
folderPath = "EmployeeImages"
imgList = []
EmpIds = []

for path in os.listdir(folderPath):
    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)
    EmpIds.append(os.path.splitext(path)[0])
    # print(path.split('.')[0])

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
print(encodeListKnownwithIds)
print("Encoding Complete")

# Save the encodings and IDs in pickle file
file = open("EncodeFile.p", 'wb') 
pickle.dump(encodeListKnownwithIds, file)
file.close()

print("File Saved")


