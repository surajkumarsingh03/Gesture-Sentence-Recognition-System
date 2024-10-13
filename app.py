
from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

app = Flask(__name__)

# Initialize the video capture, hand detector, and classifier
cap = cv2.VideoCapture(0)

# Set the desired frame width and height
frame_width = 540  # Set this to the desired width
frame_height = 480  # Set this to the desired height (you can adjust this as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")

offset = 20
imgSize = 300
labels = ["Have you Eaten", "Goodbye! Have a good day", "Hello, How are you!", "Help", "you", "No", "Do you want to Play", 
          "Please, Do me a Favour", "Thank you!!!", "Yes, Offcourse"]

detected_gestures = []
sentence = ""
sentence_printed = False

start_time = time.time()
pause_duration = 2  # Seconds to wait before assuming the gesture sequence is complete

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/feedback.html')
def feedback():
    return render_template('feedback.html')

@app.route('/backend.html')
def back():
    return render_template('backend.html')

def generate_frames():
    global sentence, sentence_printed, start_time, detected_gestures
    
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure the coordinates do not exceed image dimensions
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                detected_gestures.append(labels[index])

                current_time = time.time()
                if current_time - start_time > pause_duration:
                    if not sentence_printed:
                        sentence = " ".join(detected_gestures)
                        sentence_printed = True
                    else:
                        sentence = labels[index]

                    detected_gestures = []
                    start_time = current_time

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

        cv2.putText(imgOutput, sentence, (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()

