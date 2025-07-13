import os
import time
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from numpy.lib.function_base import append
from read_image import read_img
from read_calib import read_cali
from get_abs_phase import *
from get_3d_result import *

def get_fore(fore,black):
    """
    提取前景区域，使用亮图与暗图的差值来分割目标区域
    
    参数:
        fore: 前景图像(亮图)
        black: 背景图像(暗图)
        
    返回:
        flag: 二值掩码，标记有效的前景区域(1为前景，0为背景)
    """
    flag = fore-black
    flag[flag<10]=0  # 差值小于阈值的认为是背景
    flag[flag!=0]=1  # 差值大于阈值的认为是前景
    
    reveal = flag*255
    return flag

#初始化参数  
fx = [81,72,64]  # 水平方向的三个频率，从高到低
fy = [64,56,49]  # 垂直方向的三个频率，从高到低
pixel_size = 0.36375  # 像素物理尺寸(mm)
screen_width = 1920*pixel_size  # 屏幕宽度(mm)
screen_height = 1080*pixel_size  # 屏幕高度(mm)
phix0=0.5  # 水平方向相位偏移
phiy0=0.5  # 垂直方向相位偏移
phase_step=4  # 相移步数(4步相移)


# 读取标定参数
# 包括相机内参、畸变系数、相机与投影仪之间的变换关系等
M1,M2,D1,D2,Rc,Rs,Tc,Ts,plane_parameter = read_cali(r"F:\dataset\phase_shift_in_speebot\2022_11_23\renderImag1123_10\calibResult_regular1.xml")
if(M1 is None or M2 is None or D1 is None or D2 is None or Rc is None or Rs is None or Tc is None or Ts is None or plane_parameter is None):
    raise Exception("读取标定数据错误！")

# print("%d",plane_parameter[0][0:3])

# 读取映射表(可选)
# 映射表用于快速查找特定区域的深度范围估计
mapping = cv.imread(r"F:\dataset\phase_shift_in_speebot\test\cali\mapping.tiff",-1)#[:,:,0:3]
# plt.figure()
# plt.imshow(mapping)
# plt.show()

# 读取B样条插值表
# 用于在相位插值计算中提高精度
file_table = cv.FileStorage("./interpolation_table.xml",cv.FILE_STORAGE_READ)
Table_interpolation = file_table.getNode("interpolation_table").mat()
Table_interpolation = Table_interpolation.astype(np.float64)
Table_interpolation = np.transpose(Table_interpolation)

# 获取相位
# 下面注释的代码展示了如何从相移图像计算相位
# phase1 = multi_phase(fx,4,img1[0:24],ph0=0.5)
# phase2 = multi_phase(fx,4,img2[0:24],ph0=0.5)
# unwarp_x1,unwarp_y1,ratio1 = phase1.get_phase()
# unwarp_x2,unwarp_y2,ratio2 = phase2.get_phase()

# 读取已计算好的相位图文件
path = r"F:\dataset\phase_shift_in_speebot\2022_11_23\renderImag1123_10"
unwarp_x1 = cv.imread(path+"/VL.tiff",-1)  # 相机1垂直方向相位图
unwarp_y1 = cv.imread(path+"/HL.tiff",-1)  # 相机1水平方向相位图
unwarp_x2 = cv.imread(path+"/VR.tiff",-1)  # 相机2垂直方向相位图
unwarp_y2 = cv.imread(path+"/HR.tiff",-1)  # 相机2水平方向相位图

# 归一化相位图(注释部分)
# cv.normalize(unwarp_x1,unwarp_x1,0,1,cv.NORM_MINMAX)
# cv.normalize(unwarp_y1,unwarp_y1,0,1,cv.NORM_MINMAX)
# cv.normalize(unwarp_x2,unwarp_x2,0,1,cv.NORM_MINMAX)
# cv.normalize(unwarp_y2,unwarp_y2,0,1,cv.NORM_MINMAX)

# 计算相位图的有效区域掩码
# 对于值为0的区域(无相位信息)，将其设为0，有效区域设为1
ret,flag1x = cv.threshold(unwarp_x1,0,1,cv.THRESH_BINARY_INV)
ret,flag1y = cv.threshold(unwarp_y1,0,1,cv.THRESH_BINARY_INV)
ret,flag2x = cv.threshold(unwarp_x2,0,1,cv.THRESH_BINARY_INV)
ret,flag2y = cv.threshold(unwarp_y2,0,1,cv.THRESH_BINARY_INV)

# 反转掩码，使得有效区域为1，无效区域为0
flag1x = -1*flag1x+1
flag1y = -1*flag1y+1
flag2x = -1*flag2x+1
flag2y = -1*flag2y+1

# 同时满足水平和垂直方向相位有效的区域才是最终有效区域
flag1 = flag1x*flag1y  # 相机1的有效区域
flag2 = flag2x*flag2y  # 相机2的有效区域
# plt.figure()
# plt.imshow(flag1,cmap='gray')
# plt.figure()
# plt.imshow(flag2,cmap='gray')
# plt.show()



# plt.figure()
# plt.subplot(121)
# plt.title("VL")
# plt.imshow(unwarp_x1,cmap="gray")
# plt.subplot(122)
# plt.title("HL")
# plt.imshow(unwarp_y1,cmap="gray")


# plt.figure()
# plt.subplot(121)
# plt.title("VR")
# plt.imshow(unwarp_x2,cmap="gray")
# plt.subplot(122)
# plt.title("HR")
# plt.imshow(unwarp_y2,cmap="gray")
# plt.show()

# 获取深度信息
# 相机1坐标转换
# 相机2坐标转换

# 使用get_3dresult类进行3D重建
# 通过相机1和相机2的相位图和标定参数计算三维点云
result=get_3dresult(phase_x1=unwarp_x1,  phase_y1=unwarp_y1,phase_x2=unwarp_x2,phase_y2=unwarp_y2,flag1 = flag1,flag2 = flag2,screen_width=screen_width,screen_height=screen_height,Rs=Rs,Ts=Ts,Rc=Rc,Tc=Tc,M1=M1,D1=D1,M2=M2,D2=D2,plane_parameter=plane_parameter,mapping=mapping,table_interpolation=Table_interpolation)

# 执行三维重建计算
result_3d,flag_result = result.calculate()

# 将结果重新整形为点云数据
result_3d_point = np.reshape(result_3d,(-1,9,1))
point_matrix = np.squeeze(result_3d_point,axis=2)
# point_matrix_not_zeros = []
# for i in range(np.shape(point_matrix)[0]):
#     if(np.linalg.norm(point_matrix[i,2:5]!=0)):
#         point_matrix_not_zeros.append(point_matrix[i,:])
# point_matrix_not_zeros = np.array(point_matrix_not_zeros)
# point_matrix_not_zeros.shape = -1,3

# 保存三维重建结果到文本文件
save_txt = r"F:\dataset\phase_shift_in_speebot\2022_11_23\renderImag1123_10"
try:
    os.makedirs(save_txt)
except:
    pass
txt_name = "/resulttest.txt"
# txt_name = "/{}.txt".format(time.strftime('%H_%M',time.localtime(time.time())))
with open(save_txt+txt_name,'w') as f:
    np.savetxt(f,point_matrix)
    f.close()


# 可视化三维重建结果
# 创建3D散点图，展示重建的点云
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(point_matrix[:,2],point_matrix[:,3],point_matrix[:,4])
plt.show()


# plt.figure()
# plt.imshow(result_3d)
# plt.show()


# fig=plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.plot_surface(result_3d[:,:,0],result_3d[:,:,1],result_3d[:,:,2],cmap='rainbow')
# plt.show()

