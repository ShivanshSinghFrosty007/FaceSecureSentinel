import cv2
import dlib

image = cv2.imread("D:\PycharmProjects\HSS\Images\\1.jpg")
image = cv2.resize(image, (600, 400))

cnnDetector = dlib.cnn_face_detection_model_v1('D:\PycharmProjects\HSS\mmod_human_face_detector.dat')
print("1")
faceDetections = cnnDetector(image, 1)
print('2')

for face in faceDetections:
    left, top, right, bottom = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
    confidence = face.confidence
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
cv2.imshow('display', image)
