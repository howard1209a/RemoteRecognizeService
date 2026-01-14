import torch
from utils.general import non_max_suppression


class YoloDetector:
    def __init__(self):
        # 加载拆分后的模型
        self.models = [torch.load(f'yolo/sub_model/sub_model{i}.pth') for i in range(25)]

        # 将模型设置为评估模式
        for model in self.models:
            model.eval()

        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.max_det = 1000  # maximum detections per image

    def detect(self, input_tensor, index):
        cache = []

        # 按顺序计算每个模型
        with torch.no_grad():
            for i in range(len(self.models)):
                if i == index - 1:
                    cache.append(input_tensor)
                    continue
                elif i < index - 1:
                    cache.append(None)
                    continue

                sequential = self.models[i]
                model = sequential[0]
                depend_list = model.f

                if isinstance(model.f, int):
                    intput = cache[model.f]
                else:
                    intput = []
                    for depend in depend_list:
                        intput.append(cache[depend])

                output = sequential(intput)
                cache.append(output)

        pred = cache[-1][0]

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        return pred[0]
