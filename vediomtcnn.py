import cv2
from mtcnn import MTCNN

cap= cv2.VideoCapture(0)

detect= MTCNN()

while True:
    success, img = cap.read()

    output = detect.detect_faces(img)

    for single_output in output:
        x,y,w,h= single_output['box']
        cv2.rectangle(img, pt1=(x,y), pt2=(x+w ,y+h), color=(255,0,0), thickness=3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF ==ord('x'):
        break
cv2.destroyAllWindows()