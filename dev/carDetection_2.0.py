import cv2
from matplotlib import pyplot as plt

car_cascade = cv2.CascadeClassifier('cascade.xml')
fn = r"C:\Users\clair\dev\Parking UFO\video\DJI_0005.MOV"
print(fn)
cap = cv2.VideoCapture(fn)
# img = cv2.imread('car3.jpg', 1)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw border
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        # ncars = ncars + 1

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
