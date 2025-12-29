import cv2
import os
import numpy as np

DATASET_PATH = "students_faces"
MODEL_PATH = "face_model.yml"
LABEL_PATH = "labels.npy"

faces_data = []
labels = []
label_map = {}

current_label = 0
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            faces_data.append(face)
            labels.append(current_label)

    current_label += 1

faces_data = np.array(faces_data)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_data, labels)

recognizer.save(MODEL_PATH)
np.save(LABEL_PATH, label_map)

print("✅ Training completed successfully")
