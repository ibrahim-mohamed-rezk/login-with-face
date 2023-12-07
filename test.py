import pickle
import numpy as np
import cv2
import face_recognition


print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

imgStudent = []
img =cv2.imread("598741.jpeg")
imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

faceCurFrame = face_recognition.face_locations(imgS)
print("faceCurFrame", faceCurFrame)
encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
print("encodeCurFrame", encodeCurFrame)

if faceCurFrame:
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)
        matchIndex = np.argmin(faceDis)
        print("Match Index", matchIndex)

        if matches[matchIndex]:
            print("Known Face Detected")
            print(studentIds[matchIndex])
            id = studentIds[matchIndex]
else:
    print("not user")
