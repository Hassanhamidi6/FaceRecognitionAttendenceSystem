import cv2 
import face_recognition
import pickle
import os

# importing the Student images

folderPath = "StudentImages"
imgList = []
StdIds = []

for path in os.listdir(folderPath):
    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)
    StdIds.append(os.path.splitext(path)[0])
    # print(path.split('.')[0])

print(StdIds) 

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
encodeListKnownwithIds = [StdIds, encodeListKnown]
# print(encodeListKnown)
print("Encoding Complete")

# Save the encodings and IDs using pickle
file = open("EncodeFile.p", 'wb') 
pickle.dump(encodeListKnownwithIds, file)
file.close()

print("File Saved")