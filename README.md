# SmartSign_Attendance
This project Develops an Efficient Attendance System using computer vision which recognises Handwritten Signatures in the attendance sheet automating the process of marking students as present.

## Project Overview

The current attendance process often involves students manually signing an attendance sheet, which is then processed manually to mark attendance. This approach is not only inefficient but also time-consuming. This project automates the entire process of marking attendance by using **Computer Vision** and **Signature Recognition** techniques.

The system allows users to upload images of attendance sheets and automatically detects the presence of individuals based on their **handwritten signatures** and stores the attendance record in a form of a table. 

## Project Structure

The project contains the following important folders and files:

### 1. **`dataset3/`**  
This folder contains the images and labels used for training and validation:

- **`train3/`**: Contains around 60 images along with their corresponding YOLO-formatted label files (created using the `LabelImg` tool) for training.
- **`val3/`**: Contains 2 images with their YOLO-formatted label files used for validation.
- **`data3.yaml`**: Contains paths to the training and validation data folders, the number of classes, and the class names. This file is required for training the YOLOv8 model.

### 2. **`app2.py`**  
A **Flask** web application that serves as the interface for the attendance system.  
- **Functionality**: Allows users to upload scanned attendance sheet images and receive the processed output:
    - An image with detected signatures highlighted.
    - A table displaying who is present and who is absent based on the detected signatures.

### 3. **`train.py`**  
A script to train the **YOLOv8** model on the provided dataset.  
- **Key Parameters**:  
    - `data`: Path to the `data3.yaml` file.
    - `epochs=75`: Number of training epochs.
    - `imgsz=640`: Image size.
    - `batch=8`: Batch size.
    - `name='custom_yolov8_model'`: The name assigned to the model during training.

### 4. **`runs/`**  
This folder contains performance metrics and logs of the trained YOLO models.  
- After training, you can find detailed results, including confusion matrix and performance graphs.

### 5. **`templates/`**  
Contains the HTML templates used by the Flask web application.  
- **`index.html`**: The landing page for uploading the attendance sheet.
- **`result.html`**: Displays the processed result, including the output image and attendance table.

### 6. **`yolov5/`**  
Contains the implementation of the **YOLOv5** model (if used for any preliminary work or comparisons).  
- This folder includes the pre-trained model weights, configuration files, and other related files.

### 7. **`uploads/`**  
This folder is where the uploaded images from the Flask web interface are stored.  
- Images uploaded through the website are temporarily stored here for processing.

---

## How It Works

### Image Upload:
Users upload a scanned or photographed image of the attendance sheet.

### YOLO-based Signature Detection:
The **YOLO (You Only Look Once)** model, a real-time object detection system, is used to detect and identify handwritten signatures in the image. YOLO uses a modified **Convolutional Neural Network (CNN)** to process and classify objects (in this case, signatures) in an image.

1. **CNN Layers**: These layers perform convolutions on the input image to extract lower-level features (such as edges, textures, and patterns) and higher-level features (such as shapes and objects).
2. **Strides and Pooling**: Strides and pooling operations help reduce the spatial dimensions of the image, making the model more computationally efficient.
3. **Feature Map**: After processing the image, the backbone CNN network produces a feature map, which is passed to further stages for object classification (signature detection).

### Signature Matching:
Once a signature is detected, it is compared against a pre-registered list of authorized signatures. If a match is found, the individual is marked as **present** in the attendance table.

## Requirements

- Python 3.x
- Labelimg (for labelling the images)
- YOLO (You Only Look Once) pre-trained model

![Processed Output Image](Desktop/1.jpg)



