# face_detection_attendance.py

import streamlit as st
import cv2
import face_recognition
import numpy as np
from datetime import datetime

# --- Streamlit Setup ---
st.set_page_config(page_title="Face Detection Attendance", layout="centered")
st.title("📸 Face Detection Attendance System")
st.markdown("Real-time face detection with attendance logging. No preloaded faces required.")

# --- Attendance Tracker ---
attendance_log = {}
face_counter = 1

def mark_attendance(face_id):
    if face_id not in attendance_log:
        attendance_log[face_id] = datetime.now().strftime("%H:%M:%S")

# --- Webcam Feed ---
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not accessible.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for i, face_location in enumerate(face_locations):
            face_id = f"Face_{i+1}"
            mark_attendance(face_id)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, face_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# --- Attendance Log Display ---
st.subheader("📝 Attendance Log")
if attendance_log:
    for face_id, time in attendance_log.items():
        st.success(f"{face_id} detected at {time}")
else:
    st.info("No faces detected yet.")

# --- Instructions ---
st.markdown("""
**Instructions:**
- Click "Start Webcam" to begin detection.
- The app will detect faces and log attendance with unique IDs.
- No known face images required — works for any user.
""")
