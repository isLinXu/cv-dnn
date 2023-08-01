
import cv2
import numpy as np


def detect_objects(image_path):
    """
    加载模型和类别标签，对图像进行预处理，将预处理后的图像输入到网络中进行推理，解析网络输出并绘制边界框，最终返回绘制了边界框的图像。

    参数：
    image_path: str，待检测图像的路径。

    返回值：
    img: ndarray，绘制了边界框的图像。
    """
    # 加载模型和类别标签
    model = cv2.dnn.readNetFromDarknet('ckpts/yolov3.cfg',
                                       'ckpts/yolov3.weights')
    classes = []
    with open('data/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # 加载图像并进行预处理
    img = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # 将预处理后的图像输入到网络中进行推理
    model.setInput(blob)
    output_layers = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers)

    # 解析网络输出并绘制边界框
    conf_threshold = 0.5
    nms_threshold = 0.4
    class_ids = []
    confidences = []
    boxes = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        class_id = class_ids[i]
        label = f"{classes[class_id]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (left, top), (left + width, top + height), color, 2)
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

if __name__ == '__main__':
    # 加载图像并检测物体
    img = detect_objects('input/000000012670.jpg')

    # 显示结果图像
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()