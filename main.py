import grpc
import mediapipe as mp
import numpy as np
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


class RemoteRecognizeService(server_pb2_grpc.RemoteRecognizeServiceServicer):
    def recognize(self, request, context):
        # 使用PIL将字节数据转换为图像
        image = Image.open(io.BytesIO(request.bitmap_data))
        # 将图像转换为NumPy数组
        image_np = np.array(image)
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

        gesture_recognition_result = recognizer.recognize(mp_image)
        if len(gesture_recognition_result.handedness) == 0:
            return server_pb2.RecognizeResponse()
        else:
            return server_pb2.RecognizeResponse(gesture_recognition_result.handedness[0],
                                                gesture_recognition_result.gestures[0])


def server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_RemoteRecognizeServiceServicer_to_server(RemoteRecognizeService(), server)
    server.add_insecure_port('0.0.0.0:50051')  # [::]:50051
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    server()
