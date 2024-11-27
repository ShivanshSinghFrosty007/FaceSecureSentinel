import cv2
import time

cap = cv2.VideoCapture('D:\demo.mp4')

car_cascade = cv2.CascadeClassifier("D:\PycharmProjects\HSS\\vech.xml")
carTracker = {}
coord = [[637, 352], [904, 352], [631, 512], [952, 512]]

while True:
    counter = 0
    success, img = cap.read()
    img = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    imgS = img
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    cars = car_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.8, 13)
    image = cv2.line(img, (0, 150), (1000, 150), (0, 255, 0), 2)
    image = cv2.line(img, (0, 350), (1000, 350), (0, 255, 0), 2)

    for x, y, width, height in cars:
        # if y > 410 and y < 420:
        # print(y)
        if (x >= 100 and y > 400 and y < 430):
            cv2.line(image, (0, 150), (1000, 150), (0, 255, 0),
                     2)  # Changes color of the line
            tim1 = time.time()  # Initial time
            print("Car Entered.")
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 3)
        centerX = int((x + x + width) / 2)
        centerY = int((y + y + height) / 2)
        cv2.circle(image, (centerX, centerY), 3, (255, 0, 0), 3)

    cv2.putText(image, "counter", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Display", img)

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.waitKey(1)
