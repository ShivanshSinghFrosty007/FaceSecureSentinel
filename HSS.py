import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def captureStart():
    while True:
        success, img = cap.read()
        _, frame = cap.read()

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(captureStart(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
