import os
from PIL import Image
import numpy as np

def read_img(data_file):
    """
    读取指定文件夹下的所有图像并按数字顺序排序
    
    该函数用于批量读取结构光条纹图像，确保图像按照正确的拍摄顺序读取，
    这对于相位恢复算法非常重要。函数假设文件名以数字编号，例如1.jpg, 2.jpg等。
    
    参数:
        data_file: 图像文件夹路径，包含待处理的相移图像
        
    返回:
        img: 包含所有读取图像的列表，每个图像被转换为灰度图并转为numpy数组
             图像顺序与文件名数字顺序一致
    """
    # 按文件名中的数字排序，例如: 1.png, 2.png, 10.png 会被正确排序为 1.png, 2.png, 10.png
    data = sorted(os.listdir(data_file), key=lambda x: int(x.split('.')[0]))
    img_num = len(data)

    img = []
    
    # 遍历所有图像文件，按顺序读取
    for i in range(img_num):
        # 读取图像并转换为灰度图
        # 对于相位恢复，需要使用灰度图以获取亮度信息
        img_temp = Image.open(os.path.join(data_file, data[i])).convert('L')
        # 转换为numpy数组以便后续处理
        # numpy数组便于进行大规模数值计算
        img_temp = np.array(img_temp)
        img.append(img_temp)

    return img 