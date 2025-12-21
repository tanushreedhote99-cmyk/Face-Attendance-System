import cv2
import numpy as np
import os
from datetime import datetime

# ==== LOAD MODEL & LABELS ====
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()
CONFIDENCE_THRESHOLD = 60  # adjust if needed

ATTENDANCE_FILE = "attendance.csv"

def save_attendance(name):
    # Check if already marked today
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # create file if not exists
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    # Check if already exists today
    with open(ATTENDANCE_FILE, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.strip() and line.split(",")[0] == name and line.split(",")[1] == date_str:
                return  # already marked

    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")
    print(f"✅ Attendance saved: {name}")

def recognize_and_mark_attendance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    best_name = None
    best_box = None

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        try:
            label, confidence = recognizer.predict(face)
        except:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(label, "UNKNOWN")
        else:
            name = "UNKNOWN"

        if name != "UNKNOWN":
            save_attendance(name)

        best_name = name
        best_box = (x, y, w, h)
        break  # only first face

    return best_name, best_box
