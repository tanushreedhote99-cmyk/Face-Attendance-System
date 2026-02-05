import cv2
import os

person_name = input("Enter your name: ").strip()
dataset_path = f"dataset/{person_name}"

os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0

print("Look at camera. Capturing faces...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"Captured: {count}", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Capture", frame)

    if count >= 80:   # 80 images enough
        break

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Face dataset created successfully!")
