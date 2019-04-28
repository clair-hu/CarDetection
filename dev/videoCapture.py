import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('frontface.xml')
eye_cascade = cv2.CascadeClassifier('eyes.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    faces =face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi = img[y:y+h, x:x+h]
        eyes = eye_cascade.detectMultiScale(roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex + ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(0)
# # fourcc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 200, (640,480))
#
# while True:
#   ret, frame = cap.read()
#   out.write(frame)
#   cv2.imshow('frame', frame)
#
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()
