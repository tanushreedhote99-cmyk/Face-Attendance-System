from flask import Flask, render_template, Response, jsonify
import cv2
from attendance import recognize_and_mark_attendance
from datetime import datetime

app = Flask(__name__)

camera = None
camera_on = False
attendance_done = False
last_attendance = None

def get_camera():
    for i in [0,1,0]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    return None

def gen_frames():
    global camera, camera_on, attendance_done, last_attendance
    while camera_on:
        success, frame = camera.read()
        if not success:
            continue

        if not attendance_done:
            name, face_box = recognize_and_mark_attendance(frame)
            if name:
                last_attendance = {
                    "name": name,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S")
                }
                attendance_done = True
                camera_on = False  # auto stop

            if face_box:
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                if name:
                    cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_on, attendance_done, last_attendance
    attendance_done = False
    last_attendance = None
    if not camera_on:
        camera = get_camera()
        if camera is None:
            return jsonify({"status":"No camera found"})
        camera_on = True
    return jsonify({"status":"camera started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_on, camera
    camera_on = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status":"camera stopped"})

@app.route('/video_feed')
def video_feed():
    if camera is None:
        return "Camera not started", 404
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_last_attendance')
def get_last_attendance():
    return jsonify(last_attendance)

if __name__ == "__main__":
    app.run(debug=True)
