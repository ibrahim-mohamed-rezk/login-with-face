from flask import Flask, request, jsonify
import cv2
import os
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import numpy as np

app = Flask(__name__)

cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-reco-3ceed-default-rtdb.firebaseio.com/",
    'storageBucket': "face-reco-3ceed.appspot.com"
})

bucket = storage.bucket()

folderPath = 'Images'

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def saveImage(image, studentId):
    fileName = f'{folderPath}/{studentId}.jpg'
    cv2.imwrite(fileName, image)
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

def loadEncodeFile():
    file = open("EncodeFile.p", 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    print("encodeListKnownWithIds", encodeListKnownWithIds)
    return encodeListKnownWithIds

@app.route('/register', methods=['POST'])
def register():
    studentId = request.form['studentId']
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    saveImage(image, studentId)
    imagesList = [cv2.imread(os.path.join(folderPath, f)) for f in os.listdir(folderPath)]
    encodeListKnown = findEncodings(imagesList)
    encodeListKnownWithIds = [encodeListKnown, [f.split(".")[0] for f in os.listdir(folderPath)]]
    file = open("EncodeFile.p", 'wb')
    pickle.dump(encodeListKnownWithIds, file)
    file.close()
    return jsonify({'message': 'Image registered successfully'})

@app.route('/test', methods=['POST'])
def test():
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    imagesList = [cv2.imread(os.path.join(folderPath, f)) for f in os.listdir(folderPath)]
    encodeListKnown = findEncodings(imagesList)
    encodeListKnownWithIds = loadEncodeFile()
    encodeListKnown = encodeListKnownWithIds[0]
    studentIds = encodeListKnownWithIds[1]
    matches = face_recognition.compare_faces(encodeListKnown, face_recognition.face_encodings(image)[0])
    if True in matches:
        matchIndex = matches.index(True)
        print("matched ", matches, matchIndex, studentIds)
        return jsonify({'message': 'User exists', 'studentId': studentIds[matchIndex]})
    else:
        return jsonify({'message': 'User does not exist'})

if __name__ == '__main__':
    app.run(debug=True)
