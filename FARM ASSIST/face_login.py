import cv2
import os
import numpy as np
import webbrowser
import requests
import time

# Directory to store face images
FACE_DATA_DIR = "face_data"
STATIC_DIR = "static"  # Folder to store recognized user image

if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_face(user_name):
    """Capture and save face images for a new user."""
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 1:  # Capture 50 images for training
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Only capture if one face is clearly detected
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            img_path = f"{FACE_DATA_DIR}/{user_name}_{count}.jpg"
            cv2.imwrite(img_path, face)
            count += 1
            print(f"📸 Captured image {count}/50")

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Add a small delay between captures
            time.sleep(0.2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Face data saved for {user_name}")

def recognize_face():
    """Recognize a face and verify login."""
    cap = cv2.VideoCapture(0)
    
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("❌ OpenCV contrib module not found! Install it using:")
        print("   pip install opencv-contrib-python")
        return None

    faces, labels = [], []
    label_map = {}
    user_list = os.listdir(FACE_DATA_DIR)

    if not user_list:
        print("❌ No face data found. Please capture a face first.")
        return None

    for idx, file in enumerate(user_list):
        img = cv2.imread(f"{FACE_DATA_DIR}/{file}", cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(idx)
        label_map[idx] = file.split("_")[0]

    recognizer.train(faces, np.array(labels))

    print("🔍 Please look into the camera for login...")
    retry_attempts = 3  # Allow 3 retries
    while retry_attempts > 0:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 70:
                user_name = label_map[label]
                print(f"✅ Login Successful! Welcome, {user_name}")

                # Save recognized face as "static/user.jpg"
                cv2.imwrite(f"{STATIC_DIR}/user.jpg", frame)

                cap.release()
                cv2.destroyAllWindows()

                # Send user data to Flask
                try:
                    response = requests.get(f"http://127.0.0.1:5000/set_user?name={user_name}", timeout=3)
                    if response.status_code == 200:
                        print("✅ User data sent to Flask successfully!")
                except requests.exceptions.RequestException:
                    print("⚠️ Unable to send user data to Flask server.")

                webbrowser.open("http://127.0.0.1:5000/dashboard")  
                return user_name  

            else:
                print("❌ Face not recognized. Retrying...")
                retry_attempts -= 1

        cv2.imshow("Capturing Faces", frame)
        cv2.setWindowProperty("Capturing Faces", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("❌ Failed to recognize face after multiple attempts.")
    cap.release()
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    print("1️⃣ Capture New Face")
    print("2️⃣ Login with Face Recognition")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        user_name = input("Enter your name: ")
        capture_face(user_name)
    elif choice == "2":
        recognize_face()
    else:
        print("❌ Invalid choice. Exiting.")
