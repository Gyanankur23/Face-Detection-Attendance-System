# app.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Face Detection Attendance", layout="wide")

# --- Session State Setup ---
if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = []

if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["Live Detection", "Attendance Records", "History Tracker"])

# --- Face Detection Function ---
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# --- Page 1: Live Detection ---
if page == "Live Detection":
    st.title("Face Detection Attendance System")
    st.markdown("Use webcam or upload an image to detect faces and log attendance.")

    use_webcam = st.checkbox("Use Webcam (desktop only)")
    FRAME_WINDOW = st.empty()

    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible. Try uploading an image instead.")
        else:
            st.info("Webcam active. Press Stop to end session.")
            stop = st.button("Stop")
            while not stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame not received.")
                    break

                faces = detect_faces(frame)
                for i, (x, y, w, h) in enumerate(faces):
                    face_id = f"Face_{i+1}"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.attendance_log.append({"Face ID": face_id, "Time": timestamp})
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, face_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            faces = detect_faces(image)
            for i, (x, y, w, h) in enumerate(faces):
                face_id = f"Face_{i+1}"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.attendance_log.append({"Face ID": face_id, "Time": timestamp})
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, face_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Faces")

# --- Page 2: Attendance Records ---
elif page == "Attendance Records":
    st.title("Attendance Records")
    if st.session_state.attendance_log:
        df = pd.DataFrame(st.session_state.attendance_log)
        st.dataframe(df)
        if st.button("Save to History"):
            st.session_state.history.extend(st.session_state.attendance_log)
            st.session_state.attendance_log = []
            st.success("Attendance saved to history.")
    else:
        st.info("No attendance recorded yet.")

# --- Page 3: History Tracker ---
elif page == "History Tracker":
    st.title("Attendance History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.info("No history available.")