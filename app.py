import os
import cv2
import time
import firebase_admin
from flask import Flask, Response, jsonify, request
from firebase_admin import credentials, firestore
from inference import InferencePipeline
from threading import Thread

app = Flask(__name__)

# --- CONFIG & STATE ---
stream_active = True  # Global toggle
last_frame = None
last_upload_time = 0

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def fod_sync_sink(predictions, video_frame):
    global last_frame, last_upload_time, stream_active
    if not stream_active:
        return 

    img = video_frame.image.copy()
    detections = [p for p in predictions["predictions"] if p['confidence'] > 0.60]
    
    for pred in detections:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 3)
        
        # Throttled Firebase Upload
        current_time = time.time()
        if current_time - last_upload_time > 3:
            sector = "Sector A" if x < (img.shape[1]/3) else "Sector B" if x < (2*img.shape[1]/3) else "Sector C"
            db.collection("fod_alerts").add({
                "type": "FOD", "sector": sector, "confidence": float(pred['confidence']),
                "timestamp": firestore.SERVER_TIMESTAMP,
                "status": "Active"
            })
            last_upload_time = current_time

    last_frame = img

# --- VIDEO GENERATOR ---
def generate_frames():
    global last_frame, stream_active
    while True:
        if stream_active and last_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Send a "Station Idle" placeholder when stopped
            idle_img = cv2.imread('idle.jpg') # Or create a black frame
            if idle_img is None: idle_img = os.urandom(500) # Fallback junk bytes
            time.sleep(0.5)

# --- ROUTES ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    global stream_active
    stream_active = not stream_active
    return jsonify({"status": "success", "active": stream_active})

if __name__ == "__main__":
    Thread(target=lambda: InferencePipeline.init(
        model_id="fod-runway-dataset-a9r6k/1",
        video_reference=0,
        on_prediction=fod_sync_sink,
        api_key="uArQmW8LPw2yqYtPI1et"
    ).start(), daemon=True).start()
    app.run(host='0.0.0.0', port=5001, threaded=True)