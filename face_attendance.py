import cv2
import os
import numpy as np
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from collections import deque

# ================= PATHS =================
BASE_DIR = r"C:\Users\Admin\Desktop\faceproject"
DATASET_DIR = os.path.join(BASE_DIR, "students_faces")
MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")
CREDS_PATH = os.path.join(BASE_DIR, "credentials.json")
os.makedirs(DATASET_DIR, exist_ok=True)

# ================= SETTINGS =================
CONFIDENCE_SOFT = 65
FRAME_CHECK = 8
LOCK_FRAMES = 6

face_buffer = deque(maxlen=FRAME_CHECK)
current_identity = None
stable_frames = 0

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
    sheet.append_row([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print("📝 Attendance Saved:", name)

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise Exception("❌ Haarcascade not loaded")

# ================= FACE RECOGNIZER =================
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_model():
    faces, labels = [], []
    label_map = {}
    label_id = 0

    for person in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(folder):
            continue

        label_map[label_id] = person

        for img in os.listdir(folder):
            path = os.path.join(folder, img)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, (200, 200))
            gray = cv2.equalizeHist(gray)
            faces.append(gray)
            labels.append(label_id)

        label_id += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.write(MODEL_PATH)
        np.save(LABELS_PATH, label_map)
        print("✅ Model trained")

    return label_map

# ================= LOAD MODEL =================
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()
else:
    label_map = train_model()

# ================= CAMERA =================
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

final_identity_at_exit = None
unknown_face_img = None

print("📷 Camera ON | Press Q to Exit & Save Attendance")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(120, 120))

    if len(faces) == 0:
        face_buffer.clear()
        current_identity = None
        stable_frames = 0

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        roi = cv2.equalizeHist(roi)

        predicted = "UNKNOWN"

        if label_map:
            try:
                label, confidence = recognizer.predict(roi)
                if confidence < CONFIDENCE_SOFT:
                    predicted = label_map[label]
            except:
                pass

        # --- BUFFER & STABILITY ---
        face_buffer.append(predicted)
        name = max(set(face_buffer), key=face_buffer.count)

        # Agar UNKNOWN aa gaya toh force UNKNOWN hi rakho (purana naam leak na ho)
        if name == "UNKNOWN":
            current_identity = None
            stable_frames = 0
            display_name = "UNKNOWN"
        else:
            stable_frames += 1
            if stable_frames >= LOCK_FRAMES:
                current_identity = name
            display_name = current_identity if current_identity else name

        color = (0, 255, 0) if display_name != "UNKNOWN" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, display_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if display_name == "UNKNOWN":
            unknown_face_img = roi.copy()

    cv2.imshow("Face Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Q dabate waqt jo dikh raha hai wahi final
        if current_identity:
            final_identity_at_exit = current_identity
        else:
            final_identity_at_exit = "UNKNOWN"
        break

cam.release()
cv2.destroyAllWindows()

# ================= SAVE TO GSHEET AFTER Q =================
print("📤 Saving attendance to Google Sheet...")

if final_identity_at_exit == "UNKNOWN":
    print("❌ UNKNOWN face detected. Attendance not marked.")
elif final_identity_at_exit:
    save_to_gsheet(final_identity_at_exit)
else:
    print("⚠ No face detected.")

# ================= HANDLE UNKNOWN (REGISTER) =================
if final_identity_at_exit == "UNKNOWN" and unknown_face_img is not None:
    new_name = input("Unknown face detected. Enter name to register (or press Enter to skip): ").strip()
    if new_name:
        safe_name = "".join(c for c in new_name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not safe_name:
            print("❌ Invalid name. Skipping save.")
        else:
            folder = os.path.join(DATASET_DIR, safe_name)
            os.makedirs(folder, exist_ok=True)
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            cv2.imwrite(os.path.join(folder, filename), unknown_face_img)
            print("📸 Image saved for", safe_name)
            label_map = train_model()
            print("✅ Registration done. Next time this face will be recognized.")
