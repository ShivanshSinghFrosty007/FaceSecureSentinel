import pickle

import cv2
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 414)

file = open('D:\PycharmProjects\HSS\EncodeFile.P', 'rb')
listWithId = pickle.load(file)
file.close()
encodeListKnown, Ids = listWithId
print(encodeListKnown)
print(Ids)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    success, img = cap.read()

    # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = img
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    faces = face_cascade.detectMultiScale(cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY), 1.3, 5)

    for encodeFace, faceLoc, (x, y, width, height) in zip(encodeCurFrame, faceCurFrame, faces):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # cv2.rectangle(img, (x + 200, y + 100), (x + width + 320, y + height + 200), (255, 0, 0), 3)
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 3)
        cv2.putText(img, str(matches[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        print(matches, faceDis)

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow("Display", img)
    cv2.waitKey(1)
