import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.8, min_tracking_confidence=0.8)  # ✅ Increased accuracy

# EAR Threshold & Frame Count
EAR_THRESHOLD = 0.25 # Adjust as needed
EYE_CLOSED_FRAMES = 7  # Number of frames before alarm
closed_counter = 0  # ✅ Keeps track of eye closure duration

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_EAR(eye):
    """Calculate the Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def detect_sleep(frame_data):
    """Detect sleep from a Base64-encoded image frame."""
    global closed_counter

    # Convert Base64 to image
    image_data = base64.b64decode(frame_data)
    image = Image.open(io.BytesIO(image_data))
    frame = np.array(image)

    # Convert image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        print("❌ No face detected! Ensure good lighting and proper positioning.")
        closed_counter = 0  # ✅ Reset counter if no face is detected
        return False  # No face detected

    for face_landmarks in results.multi_face_landmarks:
        # Extract eye landmarks
        left_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1], 
                              face_landmarks.landmark[i].y * frame.shape[0]] for i in LEFT_EYE])
        right_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1], 
                               face_landmarks.landmark[i].y * frame.shape[0]] for i in RIGHT_EYE])

        # Compute EAR for both eyes
        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # ✅ Debugging: Print EAR values live
        print(f"👀 EAR: {avg_EAR:.3f} (Threshold: {EAR_THRESHOLD})")

        if avg_EAR < EAR_THRESHOLD:
            closed_counter += 1
            print(f"🔴 Eyes closed! Frame {closed_counter}/{EYE_CLOSED_FRAMES}")  
            if closed_counter >= EYE_CLOSED_FRAMES:
                print("🚨 ALARM TRIGGERED! 🚨")
                return True  # Eyes are closed
        else:
            closed_counter = 0  # ✅ Instantly reset counter when eyes open
            print("✅ Eyes Opened - ALARM STOPPED")

    return False  # Eyes are open
