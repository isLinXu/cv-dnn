

import cv2
import numpy as np

def dnn_classify_image(label_txt, model_path, config_path, input_path, output_path, framework, visualize=True):
    """
    使用 OpenCV DNN 进行图像分类
    :param label_txt:   标签文件
    :param model_path:  模型文件
    :param config_path: 配置文件
    :param input_path:  输入图像
    :param output_path: 输出图像
    :param framework:   框架
    :param visualize:   是否可视化
    :return:
    """
    # 读取图像标签
    with open(label_txt, 'r') as f:
        image_net_names = f.read().split('\n')
    # 仅获取标签的第一个词
    class_names = [name.split(',')[0] for name in image_net_names]

    # 读深度神经网络模型
    model = cv2.dnn.readNet(model=model_path,
                            config=config_path,
                            framework=framework)

    # 读取图像
    image = cv2.imread(input_path)
    # 生成图像 Blob
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
    # 设置神经网络模型的输入 Blob
    model.setInput(blob)
    # 通过模型对 Blob 进行分类
    outputs = model.forward()

    # 获取分类结果
    final_outputs = outputs[0]
    # 将所有结果转换为一维形式
    final_outputs = final_outputs.reshape(1000, 1)
    # 获取分类编号
    # label = np.argmax(final_outputs)
    label_id = np.argmax(final_outputs)
    # 将输出分数转化为 Softmax 概率
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    # 获取最终最大的概率分数
    final_prob = np.max(probs) * 100
    # 根据分类编号获取分类标签名称
    out_name = class_names[label_id]


    if visualize:
        # 在图像上标出分类结果
        out_text = f"{out_name}, {final_prob:.3f}"
        cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.imwrite(output_path, image)
    return out_name, final_prob


if __name__ == '__main__':
    label_txt = './input/classification_classes_ILSVRC2012.txt'
    model_path = './input/DenseNet_121.caffemodel'
    config_path = './input/DenseNet_121.prototxt'
    framework = 'Caffe'
    input_path = './images/dog.jpg'
    output_path = './outputs/result_class_image.jpg'
    out_name, final_prob = dnn_classify_image(label_txt, model_path, config_path, input_path, output_path, framework, visualize=True)
    print(f"Label: {out_name}, probability: {final_prob:.3f}")