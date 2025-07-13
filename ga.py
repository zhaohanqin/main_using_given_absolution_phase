#from cv2 import CAP_PROP_XI_LUT_VALUE, norm
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

from numpy.random import rand

class pos_g():
    """
    粒子群优化算法类，用于三维重建中的深度优化
    
    该类实现了粒子群优化算法，通过最小化相机投影误差来估计三维点的深度值。
    在结构光三维重建中，利用双目几何约束和已知的相位图来优化三维点的深度值。
    """
    def __init__(self,fig,M1,M2,D1,D2,Rs,Ts,Rc,Tc,plane_parameter,table_interpolation,point,point_screen,axis_sceen2,item_max,varible_num,min_varible,max_varible,w_ini=0.5,w_end=0.1,c1=2,c2=2,v_min = -2,v_max = 4,num_total=20):
        """
        初始化粒子群优化器
        
        参数:
            fig: 用于可视化的matplotlib图形对象
            M1, M2: 两个相机的内参矩阵
            D1, D2: 两个相机的畸变系数
            Rs, Ts: 从投影仪到相机1的旋转矩阵和平移向量
            Rc, Tc: 从相机2到相机1的旋转矩阵和平移向量
            plane_parameter: 投影平面参数
            table_interpolation: B样条插值表
            point: 待优化的三维点在相机1坐标系下的单位方向向量
            point_screen: 显示屏/投影仪上的对应点
            axis_sceen2: 相机2上的坐标对应显示屏的坐标(相机1坐标系下)
            item_max: 最大迭代次数
            varible_num: 变量数量(通常为1，即深度值)
            min_varible, max_varible: 变量范围(深度值的范围)
            w_ini, w_end: 惯性权重的初始值和终止值
            c1, c2: 加速常数
            v_min, v_max: 粒子速度范围
            num_total: 粒子数量
        """
        self.fig=fig
        self.M1 = M1
        self.M2 = M2
        self.D1 = D1
        self.D2 = D2
        self.Rs = Rs
        self.Ts = Ts
        self.Rc = Rc
        self.Tc = Tc
        self.plane_parameter = plane_parameter
        self.table_interpolation = table_interpolation
        self.R2 = np.linalg.inv(Rc)  # 相机1坐标到相机2坐标的旋转矩阵
        self.T2 = -1*np.dot(self.R2,Tc)  # 相机1坐标到相机2坐标的平移向量
        self.point = point
        self.point_screen = point_screen  # 二维坐标 
        self.axis_screen2 = axis_sceen2  # 相机2上的坐标对应显示屏的坐标（相机1坐标系下，深度值为3，[x,y,z]）索引为相机成像平面坐标索引

        self.num_total=num_total  # 种群数量
        self.item_max = item_max  # 最大迭代次数
        self.varible_num = varible_num  # 变量数量
        self.min_varible = min_varible  # 最小变量值
        self.max_varible = max_varible  # 最大变量值
        self.c1=c1  # 个体学习因子
        self.c2=c2  # 群体学习因子
        self.w_ini = w_ini  # 惯性权重初始值
        self.w_end = w_end  # 惯性权重终止值
        self.v_min = v_min  # 最小速度
        self.v_max = v_max  # 最大速度
        #self.fig = fig
        
        self.opt_line=[]  # 记录优化过程
        
        # 初始化粒子群，随机生成初始解
        answer=[]
        # for i in range(num_total):
        #     t=[]
        #     for n in range(self.varible_num):
        #         t.append(random.uniform(-10, 10))
        #     while t in answer:
        #         for n in range(self.varible_num):
        #             t.append(random.uniform(-10, 10))
        #     answer.append(t)
        for i in range(num_total):
            t=np.random.uniform(min_varible,max_varible,varible_num)
            while np.any(t == answer).all():
                t=np.random.uniform(min_varible,max_varible,varible_num)
            answer.append(t)
        self.answer = answer

        # 初始化粒子速度
        self.spd=[]
        # for i in range(num_total):
        #     self.spd.append([0. for i in range(self.varible_num)])
        for i in range(num_total):
            self.spd.append(np.zeros((1,self.varible_num),dtype='float'))
        self.spd = np.reshape(self.spd,[-1,self.varible_num])
        
        # 计算每个粒子的初始适应度值
        val = []
        for i in range(num_total):
            val.append(self.cal_val(answer[i]))
        # 当前解的值
        self.answer_val = val
        
        # 初始化个体最优位置和值
        self.local_best = answer
        self.local_best_val = val
        
        # 初始化全局最优位置和值
        min_index = self.local_best_val.index(min(self.local_best_val))
        self.total_best = answer[min_index]
        self.total_best_val = val[min_index]
        self.opt_line.append(self.total_best_val)
        # #初始化速度
        # self.spd = self.c1self.c2*random.uniform()*(self.total_best-self.answer)


    def cal_val(self,answer):
        """
        计算适应度函数值(投影误差)
        
        通过计算三维点投影到相机2上的位置与实际观测位置之间的误差来评估深度值的准确性
        
        参数:
            answer: 当前粒子的位置(深度值)
            
        返回:
            error: 投影误差值
        """
        # 根据深度值计算三维点在相机1坐标系下的坐标
        camera1_point = self.point*answer[0]
        camera1_point.shape = -1,1

        # 转换到相机2坐标系下
        camera2_point = np.dot(self.R2,camera1_point)+self.T2

        # 将三维点投影到相机2成像平面上
        image2_point = np.dot(self.M2,camera2_point)
        image2_point/=image2_point[2,0]  # 归一化齐次坐标

        #去畸变
        # image2_point_temp = np.array([image2_point[0],image2_point[1]],dtype=np.float32)
        # image2_point_temp.shape=-1,1
        # image2_point = cv.undistortPoints(np.expand_dims(image2_point_temp,axis=1),self.M2,self.D2,None,self.M2)
        # image2_point = np.squeeze(image2_point)
        
        image2_point = np.array([image2_point[0],image2_point[1],1.],dtype=np.float64)
        image2_point.shape=-1,1

        # 检查投影点是否在相机2的视野范围内
        if (image2_point[0,0]<0 or image2_point[0,0]>np.shape(self.axis_screen2)[1]-1 or image2_point[1,0]<0 or image2_point[1,0]>np.shape(self.axis_screen2)[0]-1):
            return 999  # 如果超出范围，返回一个较大的误差值

        # # #插值获取显示屏坐标

        # #双线性插值
        # lambda1=0
        # lambda2=0
        # distanceceilx = abs(image2_point[0,0]-np.ceil(image2_point[0,0]))
        # distancefloorx = abs(image2_point[0,0]-np.floor(image2_point[0,0]))
        # distanceceily = abs(image2_point[1,0]-np.ceil(image2_point[1,0]))
        # distancefloory = abs(image2_point[1,0]-np.floor(image2_point[1,0]))
        
        # if(distanceceilx<=1e-6 and distancefloorx<=1e-6):
        #     lambda1 = distancefloorx/(distanceceilx+distancefloorx)
        # if(distanceceily<=1e-6 and distancefloory<=1e-6):
        #     lambda2 = distancefloorx/(distanceceilx+distancefloorx)
            
            
        # # lambda1 = distancefloorx/(distanceceilx+distancefloorx)
        # # lambda2 = distancefloory/(distanceceily+distancefloory)
        # # print(np.floor(image2_point[1,0]))
        # image2_screen_axis_floor_floor = self.axis_screen2[int(np.floor(image2_point[1,0])),int(np.floor(image2_point[0,0])),:]
        # image2_screen_axis_floor_ceil = self.axis_screen2[int(np.ceil(image2_point[1,0])),int(np.floor(image2_point[0,0])),:]
        # image2_screen_axis_ceil_floor = self.axis_screen2[int(np.floor(image2_point[1,0])),int(np.ceil(image2_point[0,0])),:]
        # image2_screen_axis_ceil_ceil = self.axis_screen2[int(np.ceil(image2_point[1,0])),int(np.ceil(image2_point[0,0])),:]

        # result_temp1 = lambda1*image2_screen_axis_ceil_ceil+(1-lambda1)*image2_screen_axis_floor_ceil
        # result_temp2 = lambda1*image2_screen_axis_ceil_floor+(1-lambda1)*image2_screen_axis_floor_floor

        # image2_screen_axis = lambda2*result_temp1+(1-lambda2)*result_temp2
        # image2_screen_axis.shape = 3,1

        # B样条插值获取显示屏坐标
        # 首先计算插值的整数部分
        image2_round = np.floor(image2_point)
        
        # 确保插值区域在有效范围内
        if(image2_round[0,0]<3 or image2_round[0,0]>=np.shape(self.axis_screen2)[1]-3 or image2_round[1,0]<3 or image2_round[1,0]>=np.shape(self.axis_screen2)[0]-3):
            return 999  # 如果无法进行有效插值，返回一个较大的误差值
            
        # 检查插值区域内的所有点是否有效
        for i in range(-2,4):
            for j in range(-2,4):
                if(self.axis_screen2[int(image2_round[1,0])+i,int(image2_round[0,0])+j,0]==0 or self.axis_screen2[int(image2_round[1,0])+i,int(image2_round[0,0])+j,1]==0):
                    return 999  # 如果任一点无效，返回一个较大的误差值

        # 提取插值区域内的x、y、z坐标
        Cx = self.axis_screen2[int(image2_round[1,0]-2):int(image2_round[1,0]+4),int(image2_round[0,0]-2):int(image2_round[0,0]+4),0]
        Cy = self.axis_screen2[int(image2_round[1,0]-2):int(image2_round[1,0]+4),int(image2_round[0,0]-2):int(image2_round[0,0]+4),1]
        Cz = self.axis_screen2[int(image2_round[1,0]-2):int(image2_round[1,0]+4),int(image2_round[0,0]-2):int(image2_round[0,0]+4),2]

        # B样条插值参数
        Step=0.00005
        Delta_Y = image2_point[1,0]-image2_round[1,0]  # 小数部分，用于插值
        Delta_X = image2_point[0,0]-image2_round[0,0]  # 小数部分，用于插值

        # 使用预计算的插值表进行B样条插值
        Temp_Y = np.array(self.table_interpolation[round(Delta_Y/Step),:])
        Temp_X = np.array(self.table_interpolation[round(Delta_X/Step),:])
        Temp_Y.shape=1,6
        Temp_X.shape=1,6

        # 计算B样条插值结果
        resultx = np.dot(np.dot(Temp_Y,Cx),Temp_X.swapaxes(0,1))
        resulty = np.dot(np.dot(Temp_Y,Cy),Temp_X.swapaxes(0,1))
        resultz = np.dot(np.dot(Temp_Y,Cz),Temp_X.swapaxes(0,1))

        image2_screen_axis = np.array([[resultx],[resulty],[resultz]],dtype=np.float64)
        image2_screen_axis.shape = 3,1
        
        # 计算法线信息
        # 相机1的法线
        c1 = -camera1_point
        c1/=np.linalg.norm(c1)  # 归一化
        s1 = self.point_screen-camera1_point
        s1 /= np.linalg.norm(s1)  # 归一化

        n1 = c1+s1
        n1 /= np.linalg.norm(n1)  # 归一化后的法线

        # #相机2法线
        # mu2 = self.Tc-camera1_point
        # mu2 /= np.linalg.norm(mu2)
        # v2 = image2_screen_axis-camera1_point
        # v2 /= np.linalg.norm(v2)
        # n2 = mu2+v2
        # n2 /=np.linalg.norm(n2)
        # #误差
        # error = np.linalg.norm(n2-n1)


        # 使用相机1的法线计算相机2入射光线的反射方向
        c2 = self.Tc-camera1_point
        c2 /= np.linalg.norm(c2)
        len_temp = np.dot(c2.reshape(1,3),n1)
        s2 = c2+2*(len_temp*n1-c2)  # 反射光线方向
        s2/=np.linalg.norm(s2)

        # 计算反射光线与投影平面的交点
        t = abs(np.dot(self.plane_parameter[0][0:3],self.Ts-camera1_point))/(np.dot(self.plane_parameter[0][0:3],-s2))
        camera2_screen_n1 = camera1_point+s2*t

        # 计算预测的投影点与实际投影点之间的欧氏距离作为误差
        error = np.linalg.norm(camera2_screen_n1-image2_screen_axis)
        
      

        # #显示
        # ax = Axes3D(self.fig)
        # plot_n1 = n1+camera1_point
        # plot_n2 = n2+camera1_point

        # # # ax.scatter([0],[0],[0],c='r')
        # # # ax.scatter(self.Tc[0],self.Tc[1],self.Tc[2],c='g')

        # # # ax.plot([0,camera1_points[0,0]],[0,camera1_points[1,0]],[0,camera1_points[2,0]],'gray')
        # # # ax.plot([self.Tc[0,0],camera1_points[0,0]],[self.Tc[1,0],camera1_points[1,0]],[self.Tc[2,0],camera1_points[2,0]],'gray')
        

        # ax.scatter(camera1_point[0,0],camera1_point[1,0],camera1_point[2,0],c='b')
        # ax.plot([camera1_point[0,0],plot_n1[0,0]],[camera1_point[1,0],plot_n1[1,0]],[camera1_point[2,0],plot_n1[2,0]],color='red')
        # ax.plot([camera1_point[0,0],plot_n2[0,0]],[camera1_point[1,0],plot_n2[1,0]],[camera1_point[2,0],plot_n2[2,0]],color='green')
        # ax.set_title("error={:.5f}".format(error))

        # #plt.show()
        # plt.pause(0.000001)
        # plt.clf()
    
        # 返回误差值
        return error

    def run(self):
        """
        运行粒子群优化算法
        
        迭代执行粒子群优化过程，更新粒子位置和速度，找到全局最优解
        
        返回:
            self.total_best[0]: 最优深度值
            self.total_best_val: 最优解对应的误差值
        """
        # temp=[]
        # for i in range(100,1000):
        #     temp.append(self.cal_val([i/1000]))
        # plt.figure()
        # x = [i for i in range(len(temp))]
        # plt.plot(x,temp)
        # plt.show()
            
        for j in range(self.item_max):
            #print(j)
            for i in range(self.num_total):
                # 获取当前粒子位置
                answer_temp = self.answer[i]
                
                # 计算当前迭代的权重
                w = (self.w_ini-self.w_end)*(self.item_max-j)/self.item_max+self.w_end
                
                # 更新粒子速度
                # v = w*v + c1*rand*(pbest-x) + c2*rand*(gbest-x)
                self.spd[i] = w * self.spd[i]+self.c1*random.uniform(0,1)*(self.local_best[i]-answer_temp)+self.c2*random.uniform(0,1)*(self.total_best-answer_temp)
                
                # 更新粒子位置
                answer_temp = answer_temp+self.spd[i]

                # 限制粒子速度在有效范围内
                for n in range(self.varible_num):
                    if self.spd[i][n]<self.v_min:
                        self.spd[i][n] = self.v_min
                    elif self.spd[i][n]>self.v_max:
                        self.spd[i][n] = self.v_max
                
                    # 限制粒子位置在有效范围内
                    if answer_temp[n]<self.min_varible:
                        answer_temp[n] = self.min_varible
                    elif answer_temp[n]>self.max_varible:
                        answer_temp[n] = self.max_varible

                # 更新粒子位置
                self.answer[i] = answer_temp
                
                # 计算新位置的适应度值
                self.answer_val[i] = self.cal_val(self.answer[i])
                
                # 更新个体最优位置
                if(self.answer_val[i]<self.local_best_val[i]):
                    self.local_best_val[i] = self.answer_val[i]
                    self.local_best[i] = self.answer[i]
            
            # 更新全局最优位置
            min_index = self.answer_val.index(min(self.answer_val))
            if self.total_best_val>self.answer_val[min_index]:
                self.total_best = self.answer[min_index]
                self.total_best_val = self.answer_val[min_index]
            
            # 记录每次迭代的最优值
            self.opt_line.append(self.total_best_val)
                
            #画图
            #self.plot_image()
        
        # 返回最优深度值和对应的误差
        return self.total_best[0],self.total_best_val


    def plot_image(self):
        """
        可视化优化过程
        
        绘制优化曲线，显示最优值的变化过程
        """
        #ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(111)
        
        #ax1.scatter(self.total_best[0],self.total_best[1],c='r')
        
        # 绘制优化曲线
        x = [i for i in range(len(self.opt_line))]
        ax2.plot(x,self.opt_line,color='r')
        ax2.set_title('value = {:.2f},z={:.4f})'.format(self.total_best_val,self.total_best[0]))
        plt.pause(0.00001)
        plt.clf()
        


