# app.py

import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Face Detection Attendance", layout="centered")
st.title("📸 Face Detection Attendance System")
st.markdown("Real-time face detection with attendance logging. No known faces required.")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

attendance_log = {}
face_counter = 1

def mark_attendance(face_id):
    if face_id not in attendance_log:
        attendance_log[face_id] = datetime.now().strftime("%H:%M:%S")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not accessible.")
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

st.subheader("📝 Attendance Log")
if attendance_log:
    for face_id, time in attendance_log.items():
        st.success(f"{face_id} detected at {time}")
else:
    st.info("No faces detected yet.")

st.markdown("""
**Instructions:**
- Click "Start Webcam" to begin detection.
- The app will detect faces and log attendance with unique IDs.
- No known face images required — works for any user.
""")
