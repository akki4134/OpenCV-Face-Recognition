import cv2
import face_recognition
import numpy as np
import os

path = 'Images'
imgs = []
classNames = []
list = os.listdir(path)

print(list)

for cl in list:
    curImg = cv2.imread(f'{path}/{cl}')
    imgs.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncoding(imgs):
    encodeList = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncoding(imgs)
print('encoding completed')

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        match = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if cv2.waitKey(10) == ord('q'):
        break
    cv2.imshow('AIS', img)

cam.release()
cv2.destroyAllWindows()
