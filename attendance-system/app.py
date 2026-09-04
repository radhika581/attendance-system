"""
app.py
------
Flask REST API for the Automated Attendance System.

Endpoints
---------
GET  /
GET  /health

POST /api/register
     multipart form:
     - roll_no
     - name
     - images[] (2-10 face photos recommended)

POST /api/mark-attendance
     multipart form:
     - image

GET  /api/attendance?date=YYYY-MM-DD
GET  /api/attendance/summary
GET  /api/users
"""

import os
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template
from PIL import Image

import database as db
import face_utils


# ---------------------------------------------------------
# Flask application
# ---------------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACES_DIR = os.path.join(BASE_DIR, "data", "faces")


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def read_image_from_filestorage(file_storage):
    """
    Convert uploaded Flask/Werkzeug FileStorage
    into an OpenCV BGR image.
    """

    try:
        pil_img = Image.open(file_storage.stream).convert("RGB")
        rgb = np.array(pil_img)

        if rgb.size == 0:
            raise ValueError("Empty image")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return bgr

    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")


def retrain_model_from_disk():
    """
    Rebuild the LBPH model using all saved face samples.

    Face samples are stored as:

    data/
        faces/
            <user_id>/
                0.png
                1.png
                2.png
    """

    samples = []
    labels = []

    if not os.path.isdir(FACES_DIR):
        return False

    # Go through every registered user
    for user_id_str in os.listdir(FACES_DIR):

        user_dir = os.path.join(FACES_DIR, user_id_str)

        if not os.path.isdir(user_dir):
            continue

        try:
            user_id = int(user_id_str)
        except ValueError:
            continue

        # Read every face sample
        for filename in os.listdir(user_dir):

            file_path = os.path.join(user_dir, filename)

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            face = cv2.imread(
                file_path,
                cv2.IMREAD_GRAYSCALE
            )

            if face is None:
                continue

            samples.append(face)
            labels.append(user_id)

    # No training data
    if not samples:
        return False

    try:
        # Train and save model
        face_utils.train_model(samples, labels)

        return True

    except Exception as e:
        print("Model training error:", e)
        return False


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """
    Serve the web dashboard.
    """

    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """

    return jsonify({
        "status": "ok"
    })


# ---------------------------------------------------------
# Register user
# ---------------------------------------------------------

@app.route("/api/register", methods=["POST"])
def register():

    roll_no = request.form.get("roll_no", "").strip()
    name = request.form.get("name", "").strip()

    images = request.files.getlist("images")

    # Validate basic fields
    if not roll_no:
        return jsonify({
            "error": "roll_no is required"
        }), 400

    if not name:
        return jsonify({
            "error": "name is required"
        }), 400

    if not images:
        return jsonify({
            "error": "at least one face image is required"
        }), 400

    # Optional recommended limit
    if len(images) > 10:
        return jsonify({
            "error": "maximum 10 images are allowed"
        }), 400

    # Check duplicate roll number
    existing = db.get_user_by_roll(roll_no)

    if existing:
        return jsonify({
            "error": f"roll_no '{roll_no}' is already registered"
        }), 409

    # Create database user
    try:
        user_id = db.create_user(
            roll_no,
            name
        )
    except Exception as e:
        return jsonify({
            "error": f"could not create user: {str(e)}"
        }), 500

    # Create user's face directory
    user_dir = os.path.join(
        FACES_DIR,
        str(user_id)
    )

    os.makedirs(
        user_dir,
        exist_ok=True
    )

    saved_count = 0

    # Process uploaded images
    for idx, file_storage in enumerate(images):

        try:
            # Convert uploaded image
            bgr = read_image_from_filestorage(
                file_storage
            )

            # Detect faces
            faces, gray = face_utils.detect_faces(
                bgr
            )

            # No face
            if len(faces) == 0:
                continue

            # Choose largest face
            box = max(
                faces,
                key=lambda b: b[2] * b[3]
            )

            # Extract face
            face_crop = face_utils.extract_face(
                gray,
                box
            )

            if face_crop is None:
                continue

            # Save face sample
            output_path = os.path.join(
                user_dir,
                f"{idx}.png"
            )

            success = cv2.imwrite(
                output_path,
                face_crop
            )

            if success:
                saved_count += 1

        except Exception as e:
            print(
                f"Error processing image {idx}:",
                e
            )

    # No usable face images
    if saved_count == 0:

        try:
            db.delete_user(user_id)
        except Exception as e:
            print(
                "Database rollback failed:",
                e
            )

        # Remove empty directory
        if os.path.isdir(user_dir):

            try:
                if not os.listdir(user_dir):
                    os.rmdir(user_dir)
            except Exception:
                pass

        return jsonify({
            "error": (
                "no faces detected in the uploaded "
                "images; registration rolled back"
            )
        }), 422

    # Retrain model
    trained = retrain_model_from_disk()

    if not trained:

        return jsonify({
            "error": (
                "user was registered but model "
                "training failed"
            ),
            "user_id": user_id,
            "saved_samples": saved_count
        }), 500

    return jsonify({
        "message": (
            f"Registered {name} ({roll_no}) "
            f"with {saved_count} face sample(s)"
        ),
        "user_id": user_id,
        "face_samples": saved_count
    }), 201


# ---------------------------------------------------------
# Mark attendance
# ---------------------------------------------------------

@app.route("/api/mark-attendance", methods=["POST"])
def mark_attendance():

    # Check image
    if "image" not in request.files:

        return jsonify({
            "error": "image file is required"
        }), 400

    # Load trained model
    try:
        recognizer = face_utils.load_model()
    except Exception as e:

        return jsonify({
            "error": f"could not load recognition model: {str(e)}"
        }), 500

    # No model
    if recognizer is None:

        return jsonify({
            "error": (
                "no trained model yet; "
                "register at least one user first"
            )
        }), 400

    # Read image
    try:
        bgr = read_image_from_filestorage(
            request.files["image"]
        )
    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

    # Detect face
    try:
        faces, gray = face_utils.detect_faces(
            bgr
        )
    except Exception as e:

        return jsonify({
            "error": f"face detection failed: {str(e)}"
        }), 500

    if len(faces) == 0:

        return jsonify({
            "error": "no face detected in the image"
        }), 422

    # Choose largest face
    box = max(
        faces,
        key=lambda b: b[2] * b[3]
    )

    # Extract face
    try:
        face_crop = face_utils.extract_face(
            gray,
            box
        )
    except Exception as e:

        return jsonify({
            "error": f"could not extract face: {str(e)}"
        }), 500

    # Recognize
    try:
        user_id, confidence = face_utils.predict(
            recognizer,
            face_crop
        )
    except Exception as e:

        return jsonify({
            "error": f"face recognition failed: {str(e)}"
        }), 500

    # Unknown face
    if user_id is None:

        return jsonify({
            "error": "face not recognized (confidence too low)",
            "confidence": round(
                float(confidence),
                2
            )
        }), 404

    # Find user
    user = db.get_user_by_id(user_id)

    if user is None:

        return jsonify({
            "error": (
                "recognized label has no "
                "matching user record"
            )
        }), 500

    # Mark attendance
    try:

        success, message = db.mark_attendance(
            user_id
        )

    except Exception as e:

        return jsonify({
            "error": f"attendance database error: {str(e)}"
        }), 500

    status_code = 200 if success else 409

    return jsonify({

        "user": {
            "roll_no": user["roll_no"],
            "name": user["name"]
        },

        "confidence": round(
            float(confidence),
            2
        ),

        "marked": success,

        "message": message

    }), status_code


# ---------------------------------------------------------
# Attendance by date
# ---------------------------------------------------------

@app.route("/api/attendance", methods=["GET"])
def attendance_by_date():

    date_str = request.args.get("date")

    if not date_str:

        return jsonify({
            "error": (
                "date query param required, "
                "format YYYY-MM-DD"
            )
        }), 400

    # Basic date validation
    try:
        from datetime import datetime

        datetime.strptime(
            date_str,
            "%Y-%m-%d"
        )

    except ValueError:

        return jsonify({
            "error": (
                "invalid date format; "
                "use YYYY-MM-DD"
            )
        }), 400

    try:

        records = db.get_attendance_by_date(
            date_str
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    return jsonify({
        "date": date_str,
        "count": len(records),
        "records": records
    })


# ---------------------------------------------------------
# Attendance summary
# ---------------------------------------------------------

@app.route("/api/attendance/summary", methods=["GET"])
def attendance_summary():

    try:

        summary = db.get_attendance_summary()

        return jsonify(summary)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# List users
# ---------------------------------------------------------

@app.route("/api/users", methods=["GET"])
def users():

    try:

        return jsonify(
            db.list_users()
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Application startup
# ---------------------------------------------------------

if __name__ == "__main__":

    # Initialize database
    db.init_db()

    # Create face directory
    os.makedirs(
        FACES_DIR,
        exist_ok=True
    )

    print("=" * 50)
    print("Automated Attendance System")
    print("=" * 50)
    print("Health: http://127.0.0.1:5000/health")
    print("API:    http://127.0.0.1:5000/")
    print("=" * 50)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )