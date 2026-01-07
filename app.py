import streamlit as st
import cv2
import numpy as np
from PIL import Image

# App title
st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("🧑 Human Face Identification App")
st.write("Upload an image and adjust parameters to detect human faces.")

# Sidebar controls
st.sidebar.header("Model Parameters")
scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5)
min_face_size = st.sidebar.slider("Minimum Face Size", 30, 150, 50)

# Load Haar Cascade
@st.cache_resource
def load_model():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = load_model()

# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("📸 Image Preview")
    st.image(image, use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_face_size, min_face_size)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    st.subheader("🟩 Detection Result")
    st.image(img_array, use_column_width=True)

    st.success(f"Total faces detected: {len(faces)}")

else:
    st.info("Please upload an image to start face detection.")
