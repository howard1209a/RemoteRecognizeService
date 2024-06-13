import grpc
import mediapipe as mp
from mediapipe.tasks import python
from concurrent import futures
from PIL import Image
import io
import matplotlib.pyplot as plt

import server_pb2
import server_pb2_grpc

model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
recognizer = GestureRecognizer.create_from_options(options)

if __name__ == '__main__':
    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file('myplot.png')

    recognize_result = recognizer.recognize(mp_image)
    print(recognize_result)
