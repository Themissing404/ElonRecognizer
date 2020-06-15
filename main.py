import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
opened_yet = False

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        id_ ,conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            if id_ == 0 and not opened_yet:
                opened_yet = True
                os.startfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),'video.mp4'))
        img_item = 'myimage.png'
        #cv2.imwrite(img_item,roi_color)
        color = (255,0,0)
        stroke = 2
        width = x + w
        hight = y + h
        cv2.rectangle(frame,(x,y),(width,hight),color,stroke)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()