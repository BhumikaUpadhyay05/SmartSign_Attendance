from flask import Flask, render_template, request, send_file
from ultralytics import YOLO  # Correct import from ultralytics package
from PIL import Image
import numpy as np
import io
import os
import base64

# Initialize Flask app
app = Flask(__name__)

# Path to YOLO model 
MODEL_PATH = 'best.pt'  # Path to  trained YOLOv5 model

# Load the YOLO model
model = YOLO(MODEL_PATH)  # This loads the model trained and saved as best.pt

# List of all students
students = [
    "Aarav Patel", "Anjali Gupta", "Ayesha Khan", "Dev Joshi", 
    "Dhruv Mehta", "Kartik Agarwal", "Meera Pillai", "Soham Singh", 
    "Shruti Patel", "Tanya Sharma"
]

# Directory for uploading files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route: Display the image upload form
@app.route('/')
def index():
    return render_template('index.html')  # Form to upload image

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file is provided in the request
    if 'file' not in request.files:
        return "No file part", 400  # Return 400 Bad Request if no file is uploaded
    
    file = request.files['file']
    
    # Ensure a file is actually selected
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded image to the upload folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Run YOLOv5 model on the uploaded image
    results = model(image_path)
    
    # Get the result image with bounding boxes (returns numpy array)
    output_image = results[0].plot()  # This returns a NumPy array
    
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(output_image)  # Convert ndarray to PIL Image
    
    # Resize the image to a smaller size (adjust the size as needed)
    pil_image = pil_image.resize((400, 600))  # Resize to 800x600, adjust size as needed
    
    # Convert the PIL image to a BytesIO object so we can send it as a response
    img_io = io.BytesIO()
    pil_image.save(img_io, 'JPEG')  # Save it as JPEG (or another format if you prefer)
    img_io.seek(0)  # Seek to the beginning of the BytesIO object

    # Convert image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Extract the detected labels (class names)
    labels = results[0].names  # Class ID to label mapping
    detections = results[0].boxes  # List of detected bounding boxes
    
    detected_labels = []
    for box in detections:
        class_id = int(box.cls[0])  # Class ID of the detected object
        class_label = labels[class_id]  # Get the label corresponding to the class ID
        if class_label not in detected_labels:
            detected_labels.append(class_label)  # Add to list if not already present
    
    # Create a list of student attendance (P or A)
    student_attendance = []
    for student in students:
        if student in detected_labels:
            student_attendance.append({'name': student, 'attendance': 'Present'})
        else:
            student_attendance.append({'name': student, 'attendance': 'Absent'})
    
    # Return the image with detected labels and the list of labels to the result page
    return render_template('result.html', image_data=img_base64, student_attendance=student_attendance)

if __name__ == '__main__':
    app.run(debug=True)
