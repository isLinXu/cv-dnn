
import cv2
import time
import numpy as np

def detect_objects_video(video_path, model_path, config_path, class_names_path, confidence_threshold=0.4):
    """
    检测视频中的物体，并在物体周围绘制矩形框和类别标签，同时输出检测后的视频

    :param video_path: str, 视频路径
    :param model_path: str, 模型文件路径
    :param config_path: str, 配置文件路径
    :param class_names_path: str, 类别名称文件路径
    :param confidence_threshold: float, 置信度阈值，默认为0.4
    """
    # 加载类别名称
    with open(class_names_path, 'r') as f:
        class_names = f.read().split('\n')

    # 为每个类别获取不同的颜色
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # 加载模型
    model = cv2.dnn.readNet(model=model_path, config=config_path, framework='TensorFlow')

    # 捕获视频
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧的宽度和高度以便于保存视频
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 创建 `VideoWriter()` 对象
    out = cv2.VideoWriter('video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # 检测视频中的物体
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = frame
            image_height, image_width, _ = image.shape

            # 创建blob
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)

            # 计算FPS
            start = time.time()
            model.setInput(blob)
            output = model.forward()
            end = time.time()
            fps = 1 / (end - start)

            # 遍历每个检测结果
            for detection in output[0, 0, :, :]:
                # 获取置信度
                confidence = detection[2]

                # 仅在置信度超过阈值时绘制矩形框和类别标签
                if confidence > confidence_threshold:
                    # 获取类别ID
                    class_id = detection[1]

                    # 映射类别ID到类别名称
                    class_name = class_names[int(class_id) - 1]
                    color = COLORS[int(class_id)]

                    # 获取矩形框的坐标
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height

                    # 获取矩形框的宽度和高度
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height

                    # 绘制矩形框
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)

                    # 绘制类别名称
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # 绘制FPS
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('image', image)
            out.write(image)

            # 按下q键退出
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_objects_video(video_path='/Users/gatilin/PycharmProjects/cv-dnn/input/video_1.mp4',
                        model_path='/Users/gatilin/PycharmProjects/cv-dnn/input/frozen_inference_graph.pb',
                        config_path='/Users/gatilin/PycharmProjects/cv-dnn/input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        class_names_path='/Users/gatilin/PycharmProjects/cv-dnn/input/object_detection_classes_coco.txt',
                        confidence_threshold=0.4)
