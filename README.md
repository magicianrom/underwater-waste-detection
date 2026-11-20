# Underwater Waste Detection

An AI-powered computer vision project that identifies, detects, and segments underwater waste using deep learning.

# URL

https://underwater-waste-detection.streamlit.app

# Video link 

https://drive.google.com/file/d/1cV3d4p3lkRXe6UxZtwb3-cAMiIjpD4I3/view?usp=sharing

# Video Demo

https://github.com/user-attachments/assets/74985107-eb38-4e25-b3ac-2b2008d9554b

## Project Overview

This project uses two deep learning models:
- **MobileNetV2:** Image classification.
- **YOLO11n-seg:** Object detection and instance segmentation.

The project recognises eight underwater waste categories:
- Plastic
- Metal
- Wood
- Fabric
- Paper
- Fishing Gear
- Rubber
- Other Waste

## Models and Results

| Model | Task | Result |
|---|---|---:|
| MobileNetV2 | Classification | 57.30% Accuracy |
| YOLO11n-seg | Object Detection | 58.0% mAP50 |
| YOLO11n-seg | Segmentation | 57.3% mAP50 |

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Ultralytics YOLO11
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyCOCOtools
- Streamlit
- Google Colab

## Project Workflow

1. Explore and prepare the dataset.
2. Balance and resize classification images.
3. Train the MobileNetV2 classification model.
4. Fine-tune and evaluate MobileNetV2.
5. Convert COCO annotations into YOLO segmentation format.
6. Train YOLO11n-seg for object detection and segmentation.
7. Evaluate model performance.
8. Deploy both models using Streamlit.

## Streamlit Application

The application allows users to upload an underwater image and:

- Classify the waste category.
- Detect waste objects using bounding boxes.
- View segmentation masks outlining detected objects.
- View prediction confidence scores.

## Dataset Preparation

The original dataset uses COCO annotations.

For classification, images were organised into category folders and balanced for training.

For YOLO, COCO annotations were converted into YOLO segmentation format by:
- Mapping category IDs.
- Generating binary segmentation masks.
- Extracting object contours.
- Normalising polygon coordinates.
- Creating YOLO label files.

## Limitations

- Underwater lighting and visibility can affect predictions.
- Similar-looking waste categories can be difficult to distinguish.
- Small or partially hidden objects may be harder to detect.
- Model performance can be improved with more training data and further tuning.

## Future Improvements

- Expand the dataset with more underwater images.
- Improve classification and detection accuracy.
- Improve recognition of similar waste materials.
- Test the models under different underwater conditions.

## Conclusion

This project demonstrates the use of deep learning for underwater waste recognition by combining image classification, object detection, and instance segmentation in one Streamlit application.
