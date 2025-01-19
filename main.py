# main.py


from utils.config import LANDMARK_DATA_FOLDER, TRAINING_LOGS_FOLDER, MODEL_PATH
from app.gesture_training_app import TrainModelApp
from app.hand_tracking_app import HandTrackingApp
from camera.windows_camera import WindowsCamera

if __name__ == "__main__":
    mode = 'Tracking'
    version_name = 'generic_gestures'

    # Configure paths
    model_path = MODEL_PATH.format(version_name)
    training_logs_folder = TRAINING_LOGS_FOLDER.format(version_name)
    landmark_data_folder = LANDMARK_DATA_FOLDER.format(version_name)

    mode_options = ['Training', 'Tracking']
    if mode not in mode_options:
        raise ValueError(f"mode of `{mode} not supported. Select from {mode_options}")

    if mode == 'Training':
        app = TrainModelApp(model_path, landmark_data_folder, training_logs_folder)
    elif mode == 'Tracking':
        camera = WindowsCamera(camera_source=0, width=640, height=480)
        app = HandTrackingApp(model_path, landmark_data_folder, camera=camera)

    app.run()
