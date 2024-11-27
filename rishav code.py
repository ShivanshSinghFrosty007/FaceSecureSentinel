from flask import Flask, render_template, Response
import cv2
import time
import datetime
import pickle
import face_recognition

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 414)

# Load known faces and IDs
with open('D:\PycharmProjects\HSS\EncodeFile.P', 'rb') as file:
    encodeListKnown, Ids = pickle.load(file)

app = Flask(__name__)

# Initialize face and body cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Variables for detection and recording
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

# Frame size and codec for video writer
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

def facetest():
    global detection, detection_stopped_time, timer_started
    out = None
    while True:
        success, img = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

        faceCurFrame = face_recognition.face_locations(img)
        encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)

        # video recoding part
        if len(faces) + len(bodies) > 0:
            if not detection:
                detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
                print("Started Recording!")
            timer_started = False
        elif detection:
            if not timer_started:
                timer_started = True
                detection_stopped_time = time.time()
            elif time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                out = None
                print('Stopped Recording!')

        if detection and out is not None:
            out.write(img)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            (top, right, bottom, left) = faceLoc
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
            foundMatches = any(matches)
            matchText = "Not Match"
            if(foundMatches ==True):
                matchText = "Match Found"
            cv2.putText(img, str(matchText), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            print(matches, faceDis)

        cv2.imshow("Display", img)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(facetest(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0")