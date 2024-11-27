from flask import Flask, render_template, Response
import cv2

import pickle

import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 414)

file = open('D:\PycharmProjects\HSS\EncodeFile.P', 'rb')
listWithId = pickle.load(file)
file.close()
encodeListKnown, Ids = listWithId


app = Flask(__name__)
# camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def facetest():
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
            foundMatch = False
            for match in matches:
                if(match == True):
                    foundMatch = True
                    break
                else:
                    foundMatch = False
            cv2.putText(img, str(foundMatch), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            print(matches, faceDis)

        cv2.imshow("Display", img)

        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cv2.waitKey(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(facetest(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0")