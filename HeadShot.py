import cv2 

from cvzone.FaceDetectionModule import FaceDetector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
detector = FaceDetector()

while True:
    ret,frame=cap.read()
    if not ret:
        break 
    
    frame, bboxs = detector.findFaces(frame, draw=False)

    if bboxs:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.1,6)
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 4)
            posx=roi_color.shape[1]//2
            posy=roi_color.shape[0]//3 - 20
            cv2.circle(roi_color,(posx,posy ),30,(0,0,0),2)
            cv2.circle(roi_color,(posx,posy ),15,(255,255,255),1)
            cv2.line(roi_color,(posx-15,posy),(posx+15,posy),(0,0,255),1)
            cv2.line(roi_color,(posx,posy-15),(posx,posy+15),(0,0,255),1)
            cv2.putText(frame,'Targets In Sight : ' + str(len(faces)),(0,100),cv2.FONT_ITALIC,0.75,(0,0,0),2)
            cv2.putText(frame,'Co-ord = '+'['+str(posx)+','+str(posy)+']',(0,120),cv2.FONT_ITALIC,0.50,(0,0,0),2)
    
        
    else:
        
        
        posx=frame.shape[1]//2
        posy=frame.shape[0]//2
        cv2.circle(frame,(posx,posy ),30,(0,0,0),2)
        cv2.circle(frame,(posx,posy ),15,(255,255,255),2)
        cv2.line(frame,(posx-15,posy),(posx+15,posy),(0,0,255),1)
        cv2.line(frame,(posx,posy-15),(posx,posy+15),(0,0,255),1)
        cv2.putText(frame,'Targets In Sight : 0',(0,100),cv2.FONT_ITALIC,0.75,(0,0,0),2)
    cv2.imshow('TARGET',frame) 
    if cv2.waitKey(1) == ord('q'):
        break
            
             
cap.release()
cv2.destroyAllWindows()
