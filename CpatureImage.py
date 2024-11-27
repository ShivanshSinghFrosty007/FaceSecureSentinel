import copy
import time

import cv2

import EncodeGenetrator

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def capture():
    pause = False
    while True:
        result, image = cam.read()

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.3, 5)
        faces = list(faces)
        newimage = copy.copy(image)

        for (x, y, width, height) in faces:
            cv2.rectangle(newimage, (x, y), (x + width, y + height), (255, 0, 0), 3)
            if (cv2.waitKey(1) == ord('d')):
                named_tuple = time.localtime()  # get struct_time
                time_string = time.strftime("%H%M%S", named_tuple)
                cv2.imwrite('D:\PycharmProjects\HSS\Images\\' + time_string + ".png", image)
                return

        if not pause:
            cv2.imshow("Display", newimage)

        if cv2.waitKey(1) == ord('q'):
            print("exit")
            return


capture()
EncodeGenetrator.main()
