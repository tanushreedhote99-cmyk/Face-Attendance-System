import cv2
import os

BASE_DIR = r"C:\Users\Admin\Desktop\faceproject"
DATASET_DIR = os.path.join(BASE_DIR, "students_faces")

name = input("Enter name EXACT (folder name): ").strip()
save_path = os.path.join(DATASET_DIR, name)
os.makedirs(save_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

count = 0
MAX_IMAGES = 30

print("📷 Look straight | Good light | No mask")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 1.3, 6, minSize=(120,120)
    )

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))

        cv2.imwrite(
            os.path.join(save_path, f"{count}.jpg"),
            face
        )
        count += 1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{count}/{MAX_IMAGES}",
                    (20,40),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

    cv2.imshow("Dataset Capture", frame)

    if cv2.waitKey(1) == 27 or count >= MAX_IMAGES:
        break

cam.release()
cv2.destroyAllWindows()
print("✅ Dataset created for:", name)
