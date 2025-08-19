'''
Tasks which will be perform in this 

Detecting single face
Detecting facial landmarks
Detecting multiple face
Dretecting face in a vedio

'''

from mtcnn import MTCNN
import cv2

detect= MTCNN()

img = cv2.imread("celebs.jpg")

output= detect.detect_faces(img)
# print(output)

for i in output:
    x,y,w,h= i['box']

    left_eyeX,left_eyeY = i['keypoints']['left_eye']
    right_eyeX,right_eyeY = i['keypoints']['right_eye']
    noseX,noseY = i['keypoints']['nose']
    mouth_leftX,mouth_leftY = i['keypoints']['mouth_left']
    mouth_rightX,mouth_rightY = i['keypoints']['mouth_right']

    cv2.circle(img,center=(left_eyeX,left_eyeY),color=(255,0,0),thickness=3,radius=2)
    cv2.circle(img,center=(right_eyeX,right_eyeY),color=(255,0,0),thickness=3,radius=2)
    cv2.circle(img,center=(noseX,noseY),color=(255,0,0),thickness=3,radius=2)
    cv2.circle(img,center=(mouth_leftX,mouth_leftY),color=(255,0,0),thickness=3,radius=2)
    cv2.circle(img,center=(mouth_rightX,mouth_rightY),color=(255,0,0),thickness=3,radius=2)
    
    cv2.rectangle(img,pt1=(x,y), pt2=(x+w,y+h), color= (255, 0, 0), thickness=3) 

cv2.imshow("image", img)
cv2.waitKey(0)
