import numpy as np
import cv2
from playsound import playsound


cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

timeAllowed = 5
focusBreakTime = 0
playing = False

while True:
    ret, frame = cap.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.05, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        roi_gray = grayscale[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        if len(eyes) > 1:
            focusBreakTime += 1
        else:
            focusBreakTime = 0

        if focusBreakTime >= timeAllowed:
            playsound('assets/alarm.mp3', False)
            print("GET BACK TO WORK")

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)

    cv2.imshow('Me', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
