import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Face Detection Attendance", layout="centered")
st.title("📸 Face Detection Attendance System")
st.markdown("Real-time face detection with attendance logging. Works on mobile and desktop browsers.")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

attendance_log = {}
face_counter = 1

def mark_attendance(face_id):
    if face_id not in attendance_log:
        attendance_log[face_id] = datetime.now().strftime("%H:%M:%S")

# Webcam toggle
run = st.toggle("Enable Webcam")

FRAME_WINDOW = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Unable to access webcam. Please allow camera access in your browser settings.")
    else:
        st.info("✅ Webcam active. Rotate your phone to landscape for best experience.")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame not received. Try refreshing the page.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):
                face_id = f"Face_{i+1}"
                mark_attendance(face_id)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, face_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

# Attendance log
st.subheader("📝 Attendance Log")
if attendance_log:
    for face_id, time in attendance_log.items():
        st.success(f"{face_id} detected at {time}")
else:
    st.info("No faces detected yet.")

# Mobile instructions
st.markdown("""
### 📱 Mobile Setup Instructions:
- Use **Google Chrome** or **Microsoft Edge** on Android.
- When prompted, **allow camera access**.
- If not prompted, go to browser settings → Site settings → Camera → Allow.
- Rotate your phone to **landscape** for best experience.
""")