# 预测手写数字
import base64

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from cnn_mnist import CnnNet
from MLP_DNN import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def findBorderContours(img, maxArea=50):

    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('converted_img', img)
    #processed_img = denoise_demo(img)
    #cv2.imshow('processed_img', processed_img)

    img = accessBinary(converted_img)
    #cv2.imshow('accessBinary', img)

    contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)
    return borders

# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img
'''
#照片的数字可使用这个
           if img[i][j]>100:
               img[i][j]=0
           else:
               img[i][j]=255
'''

def denoise_demo(src):
    #src = cv2.imread("D:/javaopencv/lenanoise2.png")
    #cv2.imshow("input", src)
    # 相似窗口大小5， 搜索窗口大小25
    # h = 10, h 越大表示去噪声效果越好，细节越丢失
    dst = cv2.fastNlMeansDenoisingColored(src, None, 15, 15, 7, 21)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gret = cv2.fastNlMeansDenoising(gray, None, 15, 8, 25)
    return gret


def salt_pepper_noise(src):
    # ksize必须是大于1 奇数3\5\7\9\11
    dst = cv2.medianBlur(src, 5)
    return dst

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img

# 根据边框转换为MNIST格式
def transMNIST(img, borders, size=(28, 28)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)

        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

def predict(my_mnist_model, imgData):

    #my_mnist_model = NeuralNet(input_size,hidden_size,output_size)
    #my_mnist_model.load_state_dict(torch.load(modelpath))#'model.ckpt'
    my_mnist_model.eval()
    with torch.no_grad():
        # print(my_mnist_model.summary())
        img = imgData.astype('float32') / 255


        for i in range(len(img)):
            plt.figure("Image")  # 图像窗口名称
            plt.imshow(img[i],cmap='gray')
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title('hand write digit rec')  # 图像题目
            plt.show()
            

        img = torch.tensor(img.reshape(-1, 28 * 28))
        results = my_mnist_model(img).numpy()
        result_number = []
        for result in results:
            result_number.append(np.argmax(result))

        return result_number

def mnist_digit_rec(str_img):
    data = {'numbers': [],'marked_img':''}
    str_img = str_img.strip().replace("data:image/png;base64,", '').replace("data:image/jpg;base64,", '')
    print(str_img)
    img_data = base64.b64decode(str_img)
    #img_data = base64.b64decode(str_img)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 识别
    data['numbers'],data['marked_img']=ndarrayImg2Numbers(img_np);
    return data

def ndarrayImg2Numbers(ndarray_img):
    #my_mnist_model = CnnNet()
    #my_mnist_model.load_state_dict(torch.load('cnn_model.ckpt'))#'model.ckpt'
    input_size = 784
    hidden_size = 500
    output_size = 10

    my_mnist_model = NeuralNet(input_size,hidden_size,output_size)
    my_mnist_model.load_state_dict(torch.load('model.ckpt'))#'model.ckpt'

    borders = findBorderContours(ndarray_img)
    imgData = transMNIST(ndarray_img, borders)
    results = predict(my_mnist_model, imgData)
    marked_img = showResults(ndarray_img, 'post_request_img.jpg', borders, results)
    return results,marked_img
# 显示结果及边框

def showResults(img,save_path, borders, results=None):
    #img = cv2.imread(path)

    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)

    #cv2.imshow('test', img)
    cv2.imwrite(save_path, img)
    #cv2.waitKey(0)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('hand write digit rec')  # 图像题目
    plt.show()


    encoded = cv2.imencode('.jpg',img)
    #image_code = str(base64.b64encode(encoded[1]))

    base64_data = base64.b64encode(encoded[1])
    print(type(base64_data))
    # print(base64_data)
    # 如果想要在浏览器上访问base64格式图片，需要在前面加上：data:image/jpeg;base64,
    base64_str = "data:image/jpg;base64,"+str(base64_data, 'utf-8')

    #print(base64_str)
    return base64_str



if __name__ == '__main__':
    path = 'test_imgs/test10.jpg'
    save_path = path + '_result.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    model_path = 'model.ckpt'

    borders = findBorderContours(img)
    imgData = transMNIST(img, borders)
    results = predict(model_path, imgData)
    showResults(img, save_path, borders, results)
