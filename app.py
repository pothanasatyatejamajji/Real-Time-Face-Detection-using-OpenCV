from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import os
from datetime import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Global frame for snapshot capture
current_frame = None

def generate_frames():
    global current_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            current_frame = frame.copy()  # Save latest frame

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', message=None)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global current_frame
    if current_frame is not None:
        if not os.path.exists('faces'):
            os.makedirs('faces')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'faces/captured_face_{timestamp}.jpg'
        cv2.imwrite(filename, current_frame)
        print(f"Saved snapshot to {filename}")
        return render_template('index.html', message=f"Snapshot saved as {filename}")
    else:
        return render_template('index.html', message="❌ No frame available.")

if __name__ == '__main__':
    app.run(debug=True)
