import cv2
import os
import numpy as np
import pickle

# Path settings
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    label_map = {}
    current_id = 0

    for name in os.listdir(path):
        person_path = os.path.join(path, name)
        if os.path.isdir(person_path):
            if name not in label_map:
                label_map[current_id] = name
                current_id += 1
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                faces = detector.detectMultiScale(img)
                for (x, y, w, h) in faces:
                    faceSamples.append(img[y:y+h, x:x+w])
                    ids.append(current_id - 1)
                    
    return faceSamples, ids, label_map

print("🔄 Training faces... Isme thoda time lag sakta hai.")
faces, ids, labels = getImagesAndLabels(path)

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    with open("labels.pickle", 'wb') as f:
        pickle.dump(labels, f)
    print("✅ Training Complete! 'trainer.yml' aur 'labels.pickle' ban gayi hain.")
else:
    print("❌ Error: Dataset folder mein images nahi mili!")