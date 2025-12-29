import cv2
from datetime import datetime
import numpy as np

CONFIDENCE_THRESHOLD = 45
attendance_done = set()  # set of names already marked
attendance_data = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")
label_map = dict(enumerate(np.load("labels.npy", allow_pickle=True).item().values()))

def detect_and_mark(frame):
    global attendance_done, attendance_data

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label, confidence = recognizer.predict(face)

        if confidence > CONFIDENCE_THRESHOLD:
            name = "Unknown"
        else:
            name = label_map[label]

        # Draw green box + name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Attendance mark once per student
        if name != "Unknown" and name not in attendance_done:
            attendance_done.add(name)
            attendance_data.append({
                "name": name,
                "date": datetime.now().strftime("%d-%m-%Y"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": f"✅ Attendance Marked for {name}"
            })

    return frame
