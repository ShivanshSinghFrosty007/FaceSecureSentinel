import os
import pickle

import cv2
import face_recognition

def findEncodings(imagesList):
    encodeList = []

    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def main():
    folderPath = 'D:\PycharmProjects\HSS\Images'
    modePathList = os.listdir(folderPath)
    imgList = []
    Ids = []
    for path in modePathList:
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        Ids.append(os.path.splitext(path)[0])

    encodeListKnown = findEncodings(imgList)
    listWithId = [encodeListKnown, Ids]

    file = open('D:\PycharmProjects\HSS\EncodeFile.P', 'wb')
    pickle.dump(listWithId, file)
    file.close()
    print("done")

if __name__ == "__main__":
    main()