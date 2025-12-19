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
CONFIDENCE_THRESHOLD = 55     # balance (strict + practical)
STABLE_FRAMES_REQUIRED = 6

current_name = None
stable_frames = 0
attendance_marked = False

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
    sheet.append_row([name,
                      now.strftime("%Y-%m-%d"),
                      now.strftime("%H:%M:%S")])
    print("📝 Attendance Saved:", name)

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= FACE RECOGNIZER =================
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model():
    faces, labels = [], []
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

            gray = cv2.resize(gray, (200, 200))
            gray = cv2.equalizeHist(gray)

            faces.append(gray)
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
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("📷 Camera ON | Waiting for clear face...")

# ================= MAIN LOOP =================
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(90, 90)
    )

    if len(faces) == 0:
        stable_frames = 0
        current_name = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        roi = cv2.equalizeHist(roi)

        name = "UNKNOWN"
        color = (0, 0, 255)

        try:
            label, confidence = recognizer.predict(roi)

            if label in label_map and confidence < CONFIDENCE_THRESHOLD:
                name = label_map[label]
                color = (0, 255, 0)
            else:
                name = "UNKNOWN"

        except:
            name = "UNKNOWN"

        # ===== STABILITY CHECK =====
        if name == current_name and name != "UNKNOWN":
            stable_frames += 1
        else:
            stable_frames = 0
            current_name = name

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame,
                    f"Stable: {stable_frames}/{STABLE_FRAMES_REQUIRED}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        # ===== FINAL ATTENDANCE =====
        if (stable_frames >= STABLE_FRAMES_REQUIRED and
            name != "UNKNOWN" and
            not attendance_marked):

            save_to_gsheet(name)
            attendance_marked = True

            cv2.putText(frame, "Attendance Saved",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Face Attendance System", frame)
            cv2.waitKey(1500)
            cam.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
