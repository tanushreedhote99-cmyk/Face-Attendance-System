import cv2
import os
import numpy as np
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import hashlib

# ===== PATHS =====
BASE_DIR = r"C:\Users\Admin\Desktop\faceproject"
DATASET_DIR = os.path.join(BASE_DIR, "students_faces")
MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")
CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")
HASH_PATH = os.path.join(BASE_DIR, "dataset_hash.txt")  # dataset state check

# ===== SETTINGS =====
CONFIDENCE_THRESHOLD = 55
STABLE_FRAMES = 6

# ===== GOOGLE SHEET =====
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
client = gspread.authorize(creds)
sheet = client.open("Face_Attendance").sheet1

def save_attendance(name):
    now = datetime.now()
    sheet.append_row([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print(f"✅ Attendance saved: {name}")

# ===== FACE DETECTOR & RECOGNIZER =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ===== HELPER: dataset hash =====
def get_dataset_hash():
    files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for f in filenames:
            files.append(os.path.join(root,f))
    files.sort()
    hash_md5 = hashlib.md5()
    for f in files:
        hash_md5.update(f.encode())
    return hash_md5.hexdigest()

# ===== TRAIN MODEL =====
def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR,f))]
    if len(folders) == 0:
        print("⚠️ Dataset is empty!")
        return {}

    for person in folders:
        person_path = os.path.join(DATASET_DIR, person)
        label_map[label_id] = person
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img,(200,200))
            img = cv2.equalizeHist(img)
            faces.append(img)
            labels.append(label_id)
        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)
    np.save(LABELS_PATH, label_map)
    # save dataset hash
    dataset_hash = get_dataset_hash()
    with open(HASH_PATH, "w") as f:
        f.write(dataset_hash)

    print("✅ Model trained successfully")
    return label_map

# ===== LOAD OR TRAIN =====
retrain = False
dataset_hash = get_dataset_hash()

# check previous dataset hash
if os.path.exists(HASH_PATH):
    with open(HASH_PATH, "r") as f:
        old_hash = f.read()
    if old_hash != dataset_hash:
        retrain = True
else:
    retrain = True

# train if model missing or dataset changed
if retrain or not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
    label_map = train_model()
else:
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()
    print("✅ Model loaded")

# ===== CAMERA & ATTENDANCE =====
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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(120, 120))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        try:
            label, conf = recognizer.predict(face)
        except:
            continue

        name = "UNKNOWN"
        color = (0, 0, 255)

        if conf < CONFIDENCE_THRESHOLD:
            predicted = label_map[label]
            if predicted == current_name:
                stable += 1
            else:
                current_name = predicted
                stable = 1
            if stable >= STABLE_FRAMES:
                name = predicted
                color = (0, 255, 0)
        else:
            current_name = None
            stable = 0

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({conf:.1f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Stable: {stable}/{STABLE_FRAMES}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Attendance save
        if name != "UNKNOWN" and stable >= STABLE_FRAMES and not marked:
            save_attendance(name)
            marked = True
            print("📌 Attendance recorded, camera will close automatically")

    cv2.imshow("Face Attendance System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or marked:
        break

cam.release()
cv2.destroyAllWindows()
