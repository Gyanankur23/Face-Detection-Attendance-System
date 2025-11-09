import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import plotly.express as px

st.set_page_config(page_title="Face Detection Attendance", layout="wide")

# --- Constants ---
CSV_FILE = "attendance_history.csv"
COLLEGE_CSV = "Mumbai_University_Colleges.csv"

# --- Load College List ---
college_df = pd.read_csv(COLLEGE_CSV)
college_list = college_df["College Name"].tolist()

# --- Load History from CSV ---
if os.path.exists(CSV_FILE):
    history_df = pd.read_csv(CSV_FILE)
else:
    history_df = pd.DataFrame(columns=[
        "Name", "Roll Number", "College", "Department", "Year", "Division", "Date", "Timestamp"
    ])

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["Live Detection", "Attendance Records", "Dashboard"])

# --- Face Detection Function ---
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# --- Page 1: Live Detection ---
if page == "Live Detection":
    st.title("Face Detection Attendance System")

    use_webcam = st.checkbox("Use Webcam (desktop only)")
    FRAME_WINDOW = st.empty()

    selected_college = st.selectbox("Select your college", college_list)
    if selected_college == "Others (mention)":
        selected_college = st.text_input("Enter your college name")

    attendance_date = st.date_input("Select attendance date", datetime.today())
    department = st.selectbox("Department", ["IT", "CS", "Commerce", "Arts", "Science", "Law", "Other"])
    year = st.selectbox("Year", ["FY", "SY", "TY", "Final Year", "PG", "PhD"])
    division = st.selectbox("Class Division", ["A", "B", "C", "D"])

    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible.")
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
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face_{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if len(faces) > 0:
                    st.markdown("Face detected. Enter details below:")
                    name = st.text_input("Name")
                    roll = st.text_input("Roll Number")
                    if name and roll:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        new_entry = {
                            "Name": name,
                            "Roll Number": roll,
                            "College": selected_college,
                            "Department": department,
                            "Year": year,
                            "Division": division,
                            "Date": attendance_date.strftime("%Y-%m-%d"),
                            "Timestamp": timestamp
                        }
                        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
                        history_df.to_csv(CSV_FILE, index=False)
                        st.success("Attendance recorded.")
                        break
            cap.release()

    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            faces = detect_faces(image)
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"Face_{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Faces")

            if len(faces) > 0:
                st.markdown("Face detected. Enter details below:")
                name = st.text_input("Name")
                roll = st.text_input("Roll Number")
                if name and roll:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_entry = {
                        "Name": name,
                        "Roll Number": roll,
                        "College": selected_college,
                        "Department": department,
                        "Year": year,
                        "Division": division,
                        "Date": attendance_date.strftime("%Y-%m-%d"),
                        "Timestamp": timestamp
                    }
                    history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
                    history_df.to_csv(CSV_FILE, index=False)
                    st.success("Attendance recorded.")

# --- Page 2: Attendance Records ---
elif page == "Attendance Records":
    st.title("Attendance Records")
    if not history_df.empty:
        st.dataframe(history_df)
    else:
        st.info("No attendance records found.")

# --- Page 3: Dashboard ---
elif page == "Dashboard":
    st.title("Attendance Dashboard")

    filter_college = st.selectbox("Filter by College", ["All"] + sorted(history_df["College"].unique()))
    filter_year = st.multiselect("Filter by Year", sorted(history_df["Year"].unique()))
    filter_division = st.multiselect("Filter by Division", sorted(history_df["Division"].unique()))

    filtered_df = history_df.copy()
    if filter_college != "All":
        filtered_df = filtered_df[filtered_df["College"] == filter_college]
    if filter_year:
        filtered_df = filtered_df[filtered_df["Year"].isin(filter_year)]
    if filter_division:
        filtered_df = filtered_df[filtered_df["Division"].isin(filter_division)]

    st.subheader("Filtered Attendance Data")
    st.dataframe(filtered_df)

    if not filtered_df.empty:
        st.subheader("Attendance by Year")
        fig1 = px.pie(filtered_df, names="Year", title="Attendance Distribution by Year")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Attendance by Division")
        fig2 = px.bar(filtered_df, x="Division", title="Attendance Count by Division")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Daily Attendance Trend")
        fig3 = px.line(filtered_df.groupby("Date").size().reset_index(name="Count"), x="Date", y="Count", title="Attendance Over Time")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No data to visualize.")