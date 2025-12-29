from flask import Flask, render_template, Response, jsonify
import cv2
from attendance import detect_and_mark, attendance_data

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def generate_frames():
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        frame = detect_and_mark(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Stop camera if at least 1 attendance recorded
        if attendance_data:
            camera.release()
            break

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_attendance")
def get_attendance():
    return jsonify(attendance_data)

if __name__ == "__main__":
    app.run(debug=True)
