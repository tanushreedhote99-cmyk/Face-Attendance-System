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

CONFIDENCE_THRESHOLD = 55
STABLE_FRAMES = 6

# ================= GOOGLE SHEET =================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
client = gspread.authorize(creds)
sheet = client.open("Face_Attendance").sheet1

def save_attendance(name):
    now = datetime.now()
    sheet.append_row([
        name,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S")
    ])
    print("✅ Attendance saved:", name)

# ================= FACE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# ================= TRAIN =================
def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            gray = cv2.resize(gray, (200,200))
            gray = cv2.equalizeHist(gray)

            faces.append(gray)
            labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)
    np.save(LABELS_PATH, label_map)
    print("✅ Model trained successfully")
    return label_map

if not os.path.exists(MODEL_PATH):
    label_map = train_model()
else:
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()
    print("✅ Model loaded")

# ================= CAMERA =================
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3,640)
cam.set(4,480)

current_name = None
stable = 0
marked = False

print("📷 Camera ON")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=6, minSize=(120,120)
    )

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))
        face = cv2.equalizeHist(face)

        label, conf = recognizer.predict(face)

        name = "UNKNOWN"
        color = (0,0,255)

        if conf < CONFIDENCE_THRESHOLD:
            predicted = label_map[label]

            if predicted == current_name:
                stable += 1
            else:
                current_name = predicted
                stable = 1

            if stable >= STABLE_FRAMES:
                name = predicted
                color = (0,255,0)
        else:
            current_name = None
            stable = 0

        # ===== DRAW RECTANGLE & NAME =====
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,
                    f"{name} ({conf:.1f})",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,color,2)

        # ===== STABILITY INFO =====
        cv2.putText(frame,
                    f"Stable: {stable}/{STABLE_FRAMES}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,0),2)

        # ===== FINAL ATTENDANCE =====
        if name != "UNKNOWN" and stable >= STABLE_FRAMES and not marked:
            save_attendance(name)
            marked = True
            print("📌 Attendance recorded, camera will close in 2s")

    cv2.imshow("Face Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or marked:  # ESC or attendance done
        break

cam.release()
cv2.destroyAllWindows()
