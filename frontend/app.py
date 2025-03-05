from flask import Flask, render_template, Response
import cv2
import numpy as np
import requests
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='smile_detection.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Dummy API URL
API_URL = "https://jsonplaceholder.typicode.com/posts"

# Load pre-trained models for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Fine-tuned parameters for face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Slightly reduce scale factor for better accuracy
        minNeighbors=5,   # Increase neighbors to reduce false positives
        minSize=(30, 30)  # Ignore very small faces
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Fine-tuned parameters for smile detection
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,  # Increase scale factor for faster processing
            minNeighbors=25,  # Increase neighbors to reduce false positives
            minSize=(25, 25)   # Ignore very small smiles
        )

        if len(smiles) > 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            logging.info("Smile detected!")
            call_dummy_api("Smile detected!")
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Not Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            logging.info("No smile detected.")

    return frame

def call_dummy_api(message):
    data = {"title": "Smile Detection", "body": message, "userId": 1}
    response = requests.post(API_URL, json=data)
    if response.status_code == 201:
        logging.info(f"API call successful: {response.json()}")
    else:
        logging.error(f"API call failed: {response.status_code}")

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_smile(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)