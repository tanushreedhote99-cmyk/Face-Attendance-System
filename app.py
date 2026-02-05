import face_recognition
import cv2
import os
import gspread
from flask import Flask, render_template, Response, jsonify
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

app = Flask(__name__)

# --- CONFIG ---
DATASET_PATH = 'dataset'
SHEET_ID = "1kMbG_96D552CcXQcijLVH8ggFQxN8qLRbZdL1N-6oTc"
attendance_done = False
current_user = "Scanning..."
last_time = "--:--:--"
last_date = "--/--/----"

# Google Sheets Setup
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    print("✅ Google Sheets Connected")
except Exception as e:
    print(f"⚠️ Sheets Error: {e}")

# Pre-load Dataset
known_encodings = []
known_names = []

print("🚀 Loading Dataset... Thoda sabr rakhein.")
for name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, name)
    if not os.path.isdir(person_dir): continue
    for img_name in os.listdir(person_dir):
        img = face_recognition.load_image_file(os.path.join(person_dir, img_name))
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(name)
print(f"✅ AI Ready! Total {len(known_names)} faces loaded.")

def gen_frames():
    global attendance_done, current_user, last_time, last_date
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success or attendance_done: break

        # SPEED BOOST: Process frame at 1/4 size (Hang nahi hoga)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                
                if not attendance_done:
                    current_user = name
                    now = datetime.now()
                    last_time = now.strftime('%H:%M:%S')
                    last_date = now.strftime('%d-%m-%Y')
                    try:
                        sheet.append_row([name, last_time, last_date])
                        attendance_done = True
                    except: print("⚠️ Sheet Upload Failed")

            # Draw Box and Name (Scale back to original size)
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    return jsonify({
        "name": current_user,
        "time": last_time,
        "date": last_date
    })

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)