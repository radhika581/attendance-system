"""
face_utils.py
-------------
Face detection and face recognition using OpenCV LBPH.

Pipeline:
    1. Detect face using Haar Cascade.
    2. Crop and preprocess the face.
    3. Resize to a fixed size.
    4. Normalize lighting using histogram equalization.
    5. Train an LBPH face recognizer.
    6. Save/load the trained model.
    7. Predict the registered user.
"""

import os
import cv2
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FACE_SIZE = (200, 200)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "db"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lbph_model.yml"
)

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# ---------------------------------------------------------
# Haar Cascade
# ---------------------------------------------------------

_face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)

if _face_cascade.empty():
    raise RuntimeError(
        "Could not load Haar Cascade classifier."
    )


# ---------------------------------------------------------
# Face Detection
# ---------------------------------------------------------

def detect_faces(image_bgr):
    """
    Detect faces in a BGR OpenCV image.

    Returns:
        faces -> list of (x, y, w, h)
        gray  -> grayscale image
    """

    if image_bgr is None:
        raise ValueError(
            "Input image is None."
        )

    if len(image_bgr.shape) != 3:
        raise ValueError(
            "Expected a BGR color image."
        )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Slightly improve contrast
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    return faces, gray


# ---------------------------------------------------------
# Face Extraction / Preprocessing
# ---------------------------------------------------------

def extract_face(gray_image, box):
    """
    Crop and normalize a detected face.

    Output:
        200 x 200 grayscale image
    """

    if gray_image is None:
        raise ValueError(
            "Gray image is None."
        )

    x, y, w, h = box

    # Safety checks
    if w <= 0 or h <= 0:
        raise ValueError(
            "Invalid face bounding box."
        )

    height, width = gray_image.shape[:2]

    x = max(0, x)
    y = max(0, y)

    x2 = min(width, x + w)
    y2 = min(height, y + h)

    face = gray_image[
        y:y2,
        x:x2
    ]

    if face.size == 0:
        raise ValueError(
            "Could not extract face."
        )

    # Normalize size
    face = cv2.resize(
        face,
        FACE_SIZE
    )

    # Normalize lighting
    face = cv2.equalizeHist(
        face
    )

    return face


# ---------------------------------------------------------
# LBPH Recognizer
# ---------------------------------------------------------

def get_recognizer():
    """
    Create an OpenCV LBPH recognizer.
    """

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is unavailable. "
            "Install opencv-contrib-python."
        )

    return cv2.face.LBPHFaceRecognizer_create()


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

def train_model(face_samples, labels):
    """
    Train LBPH recognizer and save it to disk.

    face_samples:
        List of grayscale face images.

    labels:
        List of corresponding user IDs.
    """

    if not face_samples:
        raise ValueError(
            "No face samples provided for training."
        )

    if len(face_samples) != len(labels):
        raise ValueError(
            "Number of face samples and labels "
            "must be equal."
        )

    processed_samples = []

    for face in face_samples:

        if face is None:
            continue

        # Make sure every training image
        # has the same size.
        face = cv2.resize(
            face,
            FACE_SIZE
        )

        if len(face.shape) == 3:
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

        face = cv2.equalizeHist(
            face
        )

        processed_samples.append(
            face
        )

    if not processed_samples:
        raise ValueError(
            "No valid face samples available."
        )

    labels_array = np.asarray(
        labels,
        dtype=np.int32
    )

    if len(processed_samples) != len(labels_array):
        raise ValueError(
            "Valid samples and labels "
            "count do not match."
        )

    # Create recognizer
    recognizer = get_recognizer()

    # Train
    recognizer.train(
        processed_samples,
        labels_array
    )

    # Create model directory
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save model
    recognizer.write(
        MODEL_PATH
    )

    print(
        f"LBPH model saved to: {MODEL_PATH}"
    )

    return recognizer


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def load_model():
    """
    Load previously trained LBPH model.

    Returns:
        recognizer if model exists
        None otherwise
    """

    if not os.path.exists(
        MODEL_PATH
    ):
        return None

    try:

        recognizer = get_recognizer()

        recognizer.read(
            MODEL_PATH
        )

        return recognizer

    except Exception as e:

        print(
            "Could not load LBPH model:",
            e
        )

        return None


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(
    recognizer,
    face_gray,
    confidence_threshold=70.0
):
    """
    Predict the identity of a face.

    Important:
        LBPH confidence is actually a
        distance value.

        LOWER = better match.

    Returns:
        (user_id, confidence)

        or

        (None, confidence)
        if confidence is above threshold.
    """

    if recognizer is None:
        raise ValueError(
            "Recognizer is not loaded."
        )

    if face_gray is None:
        raise ValueError(
            "Face image is None."
        )

    # Ensure same size as training images
    face_gray = cv2.resize(
        face_gray,
        FACE_SIZE
    )

    if len(face_gray.shape) == 3:
        face_gray = cv2.cvtColor(
            face_gray,
            cv2.COLOR_BGR2GRAY
        )

    # Same preprocessing used during training
    face_gray = cv2.equalizeHist(
        face_gray
    )

    label, confidence = recognizer.predict(
        face_gray
    )

    confidence = float(
        confidence
    )

    if confidence <= confidence_threshold:
        return int(label), confidence

    return None, confidence