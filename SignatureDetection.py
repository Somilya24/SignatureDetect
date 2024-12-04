import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os


# ============================
# YOLO Training Function
# ============================
def train_yolo(data_yaml, epochs=10, img_size=640):
    """
    Train YOLO model with specified data and configuration.
    :param data_yaml: Path to data.yaml file for training.
    :param model_name: Pre-trained YOLO model to use.
    :param epochs: Number of training epochs.
    :param img_size: Image size for training.
    """
    st.write("Starting YOLO model training...")
    model = YOLO("yolo11n.pt")  # Load pre-trained model
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, verbose=True)
    st.success("Training complete! Check runs/detect/train/ for results.")


# ============================
# YOLO Inference Function
# ============================
def detect_signatures(image_path, model_path="runs/detect/train4/weights/best.pt"):
    """
    Run YOLO inference on an image.
    :param image_path: Path to the image for detection.
    :param model_path: Path to the trained YOLO model weights.
    """
    st.write("Running YOLO inference...")
    model = YOLO(model_path)  # Load trained model
    results = model(image_path)  # Run inference
    return results


# ============================
# Streamlit UI
# ============================
def main():
    st.title("YOLO Signature Detection")

    # Mode Selection
    mode = st.selectbox("Select Mode", ["Train", "Detect"])

    if mode == "Train":
        st.subheader("Train YOLO Model")

        # Inputs for training
        data_yaml = st.text_input("Path to signature.yaml", r"C:\Users\u461547\PycharmProjects\SignDetect\datasets\signature\data.yaml")
        model_name = st.selectbox("Pre-trained YOLO Model", ["yolov11n.pt", "yolov8s.pt", "yolov5n.pt"])
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=10)
        img_size = st.number_input("Image Size", min_value=320, max_value=1280, value=640)

        if st.button("Start Training"):
            train_yolo(data_yaml, epochs=epochs, img_size=img_size)

    elif mode == "Detect":
        st.subheader("Detect Signatures")
        uploaded_file = st.file_uploader("Upload a Document Image", type=["png", "jpg", "jpeg"])
        model_path = st.text_input("Path to Trained Model", "runs/detect/train/weights/best.pt")

        if uploaded_file is not None:
            # Save uploaded image temporarily
            image_path = "uploaded_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())

            # Run YOLO detection
            results = detect_signatures(image_path, model_path)

            # Display results
            result_image = results[0].plot()
            st.image(result_image, caption="Detected Signatures", use_column_width=True)


if __name__ == "__main__":
    main()