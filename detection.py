import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, config_path, classes_path):
        """
        初始化对象检测器
        :param model_path: 模型文件路径
        :param config_path: 配置文件路径
        :param classes_path: 类别文件路径
        """
        with open(classes_path, 'r') as f:
            self.class_names = f.read().split('\n')
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.model = cv2.dnn.readNet(model=model_path, config=config_path, framework='TensorFlow')

    def detect(self, image_path, confidence_threshold=.4):
        """
        对指定图片进行目标检测
        :param image_path: 图片路径
        :param confidence_threshold: 置信度阈值
        :return: 检测结果
        """
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        self.model.setInput(blob)
        output = self.model.forward()

        results = []
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > confidence_threshold:
                class_id = detection[1]
                class_name = self.class_names[int(class_id) - 1]
                color = self.colors[int(class_id)]
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                results.append((class_name, confidence, (box_x, box_y, box_width, box_height), color))

        return results

    def draw_boxes(self, image_path, results):
        """
        在图片上绘制检测结果
        :param image_path: 图片路径
        :param results: 检测结果
        """
        image = cv2.imread(image_path)
        for result in results:
            class_name, confidence, bbox, color = result
            box_x, box_y, box_width, box_height = bbox
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_x + box_width), int(box_y + box_height)), color, thickness=2)
            cv2.putText(image, f"{class_name}: {confidence:.2f}", (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return image

# 使用示例
if __name__ == '__main__':
    detector = ObjectDetector(model_path='./input/frozen_inference_graph.pb',
                              config_path='./input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                              classes_path='./input/object_detection_classes_coco.txt')
    results = detector.detect(image_path='./input/image_2.jpg')
    image = detector.draw_boxes(image_path='./input/image_2.jpg', results=results)
    cv2.imshow('image', image)
    cv2.imwrite('outputs/image_detection_result.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()