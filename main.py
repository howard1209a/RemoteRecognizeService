import csv
import os
import time

import grpc
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from concurrent import futures
from PIL import Image
import io

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


def is_jpeg(data: bytes) -> bool:
    # 检查文件的开头和结尾是否是JPEG格式的标志
    return data[:2] == b'\xFF\xD8' and data[-2:] == b'\xFF\xD9'


class RemoteRecognizeService(server_pb2_grpc.RemoteRecognizeServiceServicer):
    def recognize(self, request, context):
        recieve_time = (int)(time.time() * 1000)

        # 使用PIL将字节数据转换为图像
        image = Image.open(io.BytesIO(request.bitmap_data))
        # 将图像转换为NumPy数组
        image_np = np.array(image)
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

        gesture_recognition_result = recognizer.recognize(mp_image)

        sendback_time = (int)(time.time() * 1000)

        if len(gesture_recognition_result.handedness) == 0:
            return server_pb2.RecognizeResponse(
                handedness="",
                gesture="",
                x1=0,
                y1=0,
                x2=0,
                y2=0,
                recieve_time=recieve_time,
                sendback_time=sendback_time
            )
        else:
            x1, y1, x2, y2 = detectRectangle(gesture_recognition_result.hand_landmarks[0])
            return server_pb2.RecognizeResponse(handedness=gesture_recognition_result.handedness[0][0].category_name,
                                                gesture=gesture_recognition_result.gestures[0][0].category_name, x1=x1,
                                                y1=y1, x2=x2, y2=y2, recieve_time=recieve_time,
                                                sendback_time=sendback_time)

    def logReport(self, request, context):
        # 定义列名
        columns = [
            "deviceSerialNumber", "taskId", "unloadEnd", "startTime", "endTime",
            "posExist", "copyTime", "preprocessTime", "recognizeTime",
            "renderTime", "transfer2RemoteTime", "computeRemoteTime", "transfer2LocalTime",
            "transTime", "allTime"
        ]

        # 创建目录（如果不存在）
        log_dir = f"log/{request.deviceSerialNumber}"
        os.makedirs(log_dir, exist_ok=True)

        # 定义CSV文件路径
        log_file_path = f"{log_dir}/log-{request.deviceSerialNumber}.csv"

        # 准备数据行
        log_data = [
            request.deviceSerialNumber, request.taskId, request.unloadEnd,
            request.startTime, request.endTime, request.posExist,
            request.copyTime, request.preprocessTime, request.recognizeTime,
            request.renderTime, request.transfer2RemoteTime, request.computeRemoteTime,
            request.transfer2LocalTime,
            int(request.transfer2LocalTime) + int(request.transfer2RemoteTime),
            int(request.endTime) -int(request.startTime)
        ]

        # 判断文件是否存在，若不存在则写入列名
        file_exists = os.path.exists(log_file_path)

        with open(log_file_path, mode='a', newline='') as log_file:
            writer = csv.writer(log_file)

            # 如果文件是新创建的，写入列名
            if not file_exists:
                writer.writerow(columns)

            # 写入数据行
            writer.writerow(log_data)

        return server_pb2.EmptyResponse(success="true")

    def systemState(self, request, context):
        system_state_data = (
            f"{request.deviceSerialNumber} "
            f"{request.timestamp} "
            f"{request.cpuUsage} "
            f"{request.memUsage} "
            f"{request.batteryLevel}\n"
        )

        # 创建目录（如果不存在）
        log_dir = f"log/{request.deviceSerialNumber}"
        os.makedirs(log_dir, exist_ok=True)

        with open("log/" + request.deviceSerialNumber + "/" + "systemState-" + request.deviceSerialNumber + ".txt",
                  "a") as log_file:
            log_file.write(system_state_data)

        return server_pb2.EmptyResponse(success="true")


def server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_RemoteRecognizeServiceServicer_to_server(RemoteRecognizeService(), server)
    server.add_insecure_port('0.0.0.0:50051')  # [::]:50051
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()


def detectRectangle(normalizedLandmarks):
    x1 = 1
    y1 = 1
    x2 = 0
    y2 = 0

    for landmark in normalizedLandmarks:
        x1 = min(x1, landmark.x)
        y1 = min(y1, landmark.y)
        x2 = max(x2, landmark.x)
        y2 = max(y2, landmark.y)

    return x1, y1, x2, y2


if __name__ == '__main__':
    server()
