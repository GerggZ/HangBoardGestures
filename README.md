# Hand Gesture Recognition System

## Overview
This project implements a **real-time hand gesture recognition system** using **OpenCV**, **MediaPipe**, and **Keras**. The system detects hand landmarks, classifies gestures, and includes tools for training a custom gesture recognition model. It is designed to work with **both Windows and Raspberry Pi**, allowing users to train and deploy their own gesture models efficiently.

## Features
- **Live Hand Tracking** using MediaPipe's hand detection model  
- **Gesture Recognition** with a trained CNN model  
- **Gesture Recording** to collect training data  
- **Multi-Platform Camera Support** (Windows & Raspberry Pi)
  - Raspberry Pi not supported yet
- **Custom Model Training Pipeline** with Keras  


## Installation
### Prerequisites
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- TensorFlow / Keras
- scikit-learn

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Gesture Recognition System
To start live hand tracking and gesture recognition:
```bash
python main.py
```

## Training a Custom Gesture Model
To train the gesture recognition model using recorded CSV data:
```bash
cd training
python train.py
```

## Camera Support
By default, the system uses the **Windows camera module**. To explicitly use it:
```python
from camera.windows_camera import WindowsCamera
camera = WindowsCamera()
```

If support for **Raspberry Pi** is needed, implement `raspberry_pi_camera.py` using the `BaseCamera` interface.

## Screenshots
Below is an example screenshot of the hand tracking interface detecting hand landmarks in real time:

![Hand Tracking Output](docs/screenshot.png)

_(Ensure you place your screenshot inside the `docs/` folder and update the path accordingly)_

## Acknowledgments
- **MediaPipe Hands** for hand tracking
- **OpenCV** for computer vision processing
- **TensorFlow / Keras** for deep learning

## License
This project is licensed under the MIT License.





# Hand Gesture Recognition System

## Overview
This project implements a **real-time hand gesture recognition system** using **OpenCV**, **MediaPipe**, and **Keras**. The system detects hand landmarks, classifies gestures, and includes tools for training a custom gesture recognition model. It is designed to work with **both Windows and Raspberry Pi**, allowing users to train and deploy their own gesture models efficiently.

## Features
✅ **Live Hand Tracking** using MediaPipe's hand detection model  
✅ **Gesture Recognition** with a trained CNN model  
✅ **Custom Gesture Recording** to collect training data  
✅ **Flash Screen Effect** for visual feedback  
✅ **Multi-Platform Camera Support** (Windows & Raspberry Pi)  
✅ **Custom Model Training Pipeline** with Keras  
✅ **Modular Code Structure** for easy expansion and modifications  

## Installation
### Prerequisites
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- TensorFlow / Keras
- scikit-learn

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Gesture Recognition System
The system operates in two modes: **Training** and **Tracking**. The mode is selected in `main.py`.

### **1. Running Gesture Tracking**
To start live hand tracking and gesture recognition:
```bash
python main.py
```
By default, `mode = 'Tracking'` in `main.py`. This initializes the `HandTrackingApp` with the correct paths.

### **2. Training a Custom Gesture Model**
To train the gesture recognition model using recorded CSV data, set `mode = 'Training'` in `main.py`, then run:
```bash
python main.py
```
This initializes `TrainModelApp`, which will train a model using the dataset stored in `hand_landmarks_data`.

## Configuration & File Paths
The system dynamically sets paths based on a `version_name`. The default structure is defined in `config.py`, and `main.py` formats these paths at runtime:

### **Default Paths in `config.py` (Do Not Modify Directly)**
```python
MODEL_PATH = os.path.join("data", "{}_cnn_model.keras")
LANDMARK_DATA_FOLDER = os.path.join("data", "hand_landmarks_data", "{}")
TRAINING_LOGS_FOLDER = os.path.join("data", "logs", "{}")
```

### **Dynamically Configured in `main.py`**
```python
mode = 'Training'  # <-- either `Training` or `Tracking`
version_name = 'example'  # <-- the current version name, multiple different modesl can be trained
```

## Camera Support
By default, the system uses the **Windows camera module**. To explicitly use it:
```python
from camera.windows_camera import WindowsCamera
camera = WindowsCamera()
```

If support for **Raspberry Pi** is needed, implement `raspberry_pi_camera.py` using the `BaseCamera` interface.
But it isn't implemented yet

## Screenshots
Below is an example screenshot of the hand tracking interface detecting hand landmarks in real time:

![Hand Tracking Output](docs/example_screenshots_1.png)

_(Ensure you place your screenshot inside the `docs/` folder and update the path accordingly)_

## Acknowledgments
- **MediaPipe Hands** for hand tracking
- **OpenCV** for computer vision processing
- **TensorFlow / Keras** for deep learning

## License
This project is licensed under the MIT License.



