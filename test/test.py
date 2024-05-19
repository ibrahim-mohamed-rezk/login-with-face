from flask import Flask, request, jsonify
import cv2
import os
import pickle
import firebase_admin
from firebase_admin import credentials, db, storage
import numpy as np

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-reco-3ceed-default-rtdb.firebaseio.com/",
    'storageBucket': "face-reco-3ceed.appspot.com"
})

bucket = storage.bucket()

folderPath = 'Images'

def saveImage(image, studentId):
    fileName = f'{folderPath}/{studentId}.jpg'
    cv2.imwrite(fileName, image)
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

def compute_face_encoding(face_img):
    # Placeholder function to compute the face encoding.
    # This is just a placeholder; replace with actual face recognition logic.
    return [0.1, 0.2, 0.3]  # Replace with real values.

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (128, 128))
            
            # Compute face encoding (replace with actual encoding logic)
            encode = compute_face_encoding(face_img)
            encodeList.append(encode)
    
    return encodeList

# ... [rest of your imports and setup code]

def compare_encodings(unknown_encoding, known_encodings, threshold=0.6):
    """
    Compare the unknown face encoding with known encodings.
    Return the corresponding student ID if a match is found.
    """
    for idx, known_encoding in enumerate(known_encodings):
        # Compute the distance between the unknown encoding and known encoding.
        # The lower the distance, the better the match.
        distance = np.linalg.norm(np.array(unknown_encoding) - np.array(known_encoding))
        
        # If the distance is below the threshold, consider it a match.
        if distance < threshold:
            return idx  # Return the index or ID that matches.
    return None  # Return None if no match is found.





@app.route('/register', methods=['POST'])
def register():
    studentId = request.form['studentId']
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    saveImage(image, studentId)
    
    imagesList = [cv2.imread(os.path.join(folderPath, f)) for f in os.listdir(folderPath)]
    encodeListKnown = findEncodings(imagesList)
    
    db_ref = db.reference(f'Students/{studentId}')
    db_ref.set({
        'name': request.form['name'],
        'total_attendance': 0,
        'last_attendance_time': ""
    })
    
    file = open("EncodeFile.p", 'wb')
    pickle.dump(encodeListKnown, file)
    file.close()
    
    return jsonify({'message': 'Image registered successfully'})

@app.route('/test', methods=['POST'])
def test():
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    imagesList = [cv2.imread(os.path.join(folderPath, f)) for f in os.listdir(folderPath)]
    encodeListKnown = findEncodings(imagesList)
    
    # Placeholder logic for face recognition
    unknown_encoding = compute_face_encoding(image)
    
    user_id = compare_encodings(unknown_encoding, encodeListKnown)
    
    if user_id is not None:
        # Here, user_id is the index or ID of the matched user.
        # You can return additional details or just the ID based on your requirement.
        return jsonify({'message': 'User recognized', 'studentId': user_id})
    else:
        return jsonify({'message': 'User not recognized'})

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
