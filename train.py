# Step 1: Install necessary libraries
# !pip install ultralytics

# Import necessary libraries
from ultralytics import YOLO
import os

# Define paths for the dataset
dataset_path = r"/Users/bhumika/Documents/Bhumika_Upadhyay_CV_assignment/CV_project/dataset3"  
train_data = os.path.join(dataset_path, 'train3')  # Training data directory
val_data = os.path.join(dataset_path, 'val3')  # Validation data directory
yaml_path = os.path.join(dataset_path, 'data3.yaml')  # YAML file describing dataset

#  Configure and train the YOLOv8 model
model = YOLO('yolov8n.pt') 

model.train(
    data=yaml_path,
    epochs=75,  # Number of epochs
    imgsz=640,  # Image size
    batch=8,  # Batch size
    name='custom_yolov8_model'  # Name for the training run
)

#  Evaluate the model
metrics = model.val(data=yaml_path)

# Save the trained model
model_path = 'best.pt'  # Path where the model will be saved
model.save(model_path)

print(f"Model saved to {model_path}")
print(f"Metrics: {metrics}")
