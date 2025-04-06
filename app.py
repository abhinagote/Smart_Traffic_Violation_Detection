import os
import cv2
import pymysql
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from helmet_detection import detect_helmet, detect_people  # ✅ Correct function name
from one_way_detection import predict_image  # ✅ Import One-Way Detection
from triple_seat_detection import predict_image as predict_triple_seat


app = Flask(__name__)

# ✅ MySQL Configuration
db = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="detection_system",
    cursorclass=pymysql.cursors.DictCursor
)
cursor = db.cursor()

# ✅ Set upload & result folders
UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ✅ Dashboard Route (First Page)
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ✅ Helmet Detection Route
@app.route("/helmet", methods=["GET", "POST"])
def helmet_detection():
    result_image = None
    message = ""  

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        # ✅ Save uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # ✅ Run YOLO detection
        result_path, detections = detect_helmet(file_path)

        # ✅ Determine message
        if any(label == "Without Helmet" for label, _ in detections):
            message = "No Helmet Detected! 🚨"
        elif any(label == "With Helmet" for label, _ in detections):
            message = "Helmet Detected ✅"
        else:
            message = "No helmet detected in the image."

        result_image = result_path

    return render_template("index.html", result_image=result_image, message=message)

@app.route("/multi_rider", methods=["POST", "GET"])
def multi_rider_detection():
    result_image = None
    message = ""

    if request.method == "POST":
        print("✅ POST request received")  # Debugging step

        if "file" not in request.files:
            print("🚨 No file part in request!")  # Debugging step
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            print("🚨 No selected file!")  # Debugging step
            return "No selected file!", 400

        print(f"✅ File received: {file.filename}")  # Debugging step

        # ✅ Save uploaded file correctly
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        print(f"✅ File saved at: {file_path}")  # Debugging step

        # ✅ Detect multiple riders
        result_path, message = detect_people(file_path)  

        print(f"✅ Detection done. Result at: {result_path}")  # Debugging step

        result_image = result_path  

    return render_template("multi_rider_detection.html", result_image=result_image, message=message)

@app.route("/one_way", methods=["POST", "GET"])
def one_way_detection():
    result_image = None
    message = ""

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # ✅ Call prediction function
        prediction, probability = predict_image(file_path)

        if prediction is None or probability is None:
            return "Error processing image!", 500  # ❌ Handle None case

        message = f"Detected: {prediction} (Confidence: {probability:.2f})"
        result_image = file.filename  # ✅ Save result image filename

    return render_template("one_way_detection.html", result_image=result_image, message=message)

@app.route("/triple_seat", methods=["POST", "GET"])
def triple_seat_detection():
    if request.method == "GET":
        # ✅ Reset variables on page load
        return render_template(
            "triple_seat_detection.html",
            violation_label=None,
            result_image=None,
            message="Upload an image to check for violations.",
            image_path=None
        )

    # ✅ Process image on POST request
    if "file" not in request.files:
        return "No file uploaded!", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file!", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # ✅ Get only the prediction label
    violation_label, _ = predict_triple_seat(file_path)  

    message = "🚨 Triple Seat Violation Detected!" if violation_label == "Triple Seat Detected" else "✅ No Violation Detected."

    return render_template(
        "triple_seat_detection.html",
        violation_label=violation_label,
        result_image=filename,
        message=message,
        image_path=file_path
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # 🔥 Disable auto-reload
