import cv2
import os
import numpy as np
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ================= PATHS =================
BASE_DIR = r"C:\Users\Admin\Desktop\faceproject"
DATASET_DIR = os.path.join(BASE_DIR, "students_faces")
MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")
CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")

# ================= SETTINGS =================
CONFIDENCE_THRESHOLD = 60        # balanced & practical
STABLE_FRAMES_REQUIRED = 6       # same face confirmation

# ================= GOOGLE SHEET =================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
client = gspread.authorize(creds)
sheet = client.open("Face_Attendance").sheet1

def save_to_gsheet(name):
    now = datetime.now()
    sheet.append_row([
        name,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S")
    ])
    print("✅ Attendance Saved:", name)

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= FACE RECOGNIZER =================
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# ================= TRAIN MODEL =================
def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    print("🔁 Training model...")

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(
                gray, 1.3, 5
            )

            for (x, y, w, h) in faces_detected:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face = cv2.equalizeHist(face)

                faces.append(face)
                labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)
    np.save(LABELS_PATH, label_map)

    print("✅ Model trained successfully")
    return label_map

# ================= LOAD / TRAIN =================
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()
    print("✅ Model loaded")
else:
    label_map = train_model()

# ================= CAMERA =================
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_name = None
stable_frames = 0
attendance_marked = False

print("📷 Camera ON | Look straight")

# ================= MAIN LOOP =================
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=7,
        minSize=(120, 120)
    )

    if len(faces) == 0:
        current_name = None
        stable_frames = 0

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        label, confidence = recognizer.predict(face)

        name = "UNKNOWN"
        color = (0, 0, 255)

        # ===== STRICT & SAFE LOGIC =====
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_name = label_map[label]

            if current_name == predicted_name:
                stable_frames += 1
            else:
                current_name = predicted_name
                stable_frames = 1

            if stable_frames >= STABLE_FRAMES_REQUIRED:
                name = predicted_name
                color = (0, 255, 0)

        else:
            current_name = None
            stable_frames = 0

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame,
                    f"{name}  ({confidence:.1f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        cv2.putText(frame,
                    f"Stable: {stable_frames}/{STABLE_FRAMES_REQUIRED}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # ===== FINAL ATTENDANCE =====
        if (name != "UNKNOWN" and
            stable_frames >= STABLE_FRAMES_REQUIRED and
            not attendance_marked):

            save_to_gsheet(name)
            attendance_marked = True

            cv2.putText(frame, "Attendance Saved",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Face Attendance System", frame)
            cv2.waitKey(1500)

            # 🔒 AUTO CAMERA OFF
            cam.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
