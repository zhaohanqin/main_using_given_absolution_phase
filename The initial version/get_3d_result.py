import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ga import *

class get_3dresult():
    """
    三维重建类
    
    基于结构光相位图和标定信息，通过几何约束重建三维点云
    使用粒子群优化算法优化深度估计，提高三维重建的准确性
    """
    def __init__(self,phase_x1,phase_y1,phase_x2,phase_y2,flag1,flag2,screen_width,screen_height,
    Rs,Ts,Rc,Tc,M1,D1,M2,D2,plane_parameter,mapping,table_interpolation):
        """
        初始化三维重建对象
        
        参数:
            phase_x1, phase_y1: 相机1的水平和垂直方向相位图
            phase_x2, phase_y2: 相机2的水平和垂直方向相位图
            flag1, flag2: 相机1和相机2的有效区域掩码
            screen_width, screen_height: 投影屏幕的物理尺寸(mm)
            Rs, Ts: 从投影仪到相机1的旋转矩阵和平移向量
            Rc, Tc: 从相机2到相机1的旋转矩阵和平移向量
            M1, D1: 相机1的内参矩阵和畸变系数
            M2, D2: 相机2的内参矩阵和畸变系数
            plane_parameter: 投影平面参数
            mapping: 深度映射表(可选)
            table_interpolation: B样条插值表
        """
        self.phase_x1 = phase_x1  # 相机1水平相位
        self.phase_y1 = phase_y1  # 相机1垂直相位
        self.phase_x2 = phase_x2  # 相机2水平相位
        self.phase_y2 = phase_y2  # 相机2垂直相位
        self.flag1=flag1  # 相机1有效区域掩码
        self.flag2=flag2  # 相机2有效区域掩码
        
        self.Rs = Rs  # 从投影仪到相机1的旋转矩阵
        self.Ts = Ts  # 从投影仪到相机1的平移向量
        self.Rc = Rc  # 从相机2到相机1的旋转矩阵 
        self.Tc = Tc  # 从相机2到相机1的平移向量
        self.R2 = np.linalg.inv(Rc)  # 相机1坐标转换到相机2坐标的旋转矩阵
        self.T2 = -1*np.dot(self.R2,Tc)  # 相机1坐标转换到相机2坐标的平移向量
        self.M1 = M1  # 相机1内参矩阵
        self.D1 = D1  # 相机1畸变系数
        self.M2 = M2  # 相机2内参矩阵
        self.D2 = D2  # 相机2畸变系数
        self.plane_parameter = plane_parameter  # 投影平面参数
        self.mapping = mapping  # 深度映射表
        self.table_interpolation = table_interpolation  # B样条插值表
        
        self.width = screen_width  # 投影屏幕物理宽度
        self.height = screen_height  # 投影屏幕物理高度
    
    def transfer_to_world(self):
        """
        将归一化的相位值转换为实际物理坐标
        
        将相位值从[0,1]范围映射到投影屏幕的物理尺寸范围
        """
        self.phase_x1*=self.width
        self.phase_y1*=self.height
        self.phase_x2*=self.width
        self.phase_y2*=self.height
    
    def get_flag(self):
        """
        计算两个相机的共视区域
        
        找出两个相机都能看到的物理区域，确保三维重建的有效性
        
        返回:
            flag1, flag2: 更新后的两个相机有效区域掩码
        """
        # 处理水平方向共视区域
        flag1x = self.phase_x1.copy()
        flag2x = self.phase_x2.copy()

        # 计算水平相位的最大最小范围
        min_x1 = np.min(self.phase_x1)
        max_x1 = np.max(self.phase_x1)

        min_x2 = np.min(self.phase_x2)
        max_x2 = np.max(self.phase_x2)

        # 取两相机共同的水平相位范围
        max_x = min(max_x1,max_x2)
        min_x = max(min_x1,min_x2)

        # 将不在共视范围的区域置0
        flag1x[flag1x<min_x]=0
        flag1x[flag1x>max_x]=0
        flag1x[flag1x!=0]=1

        flag2x[flag2x<min_x]=0
        flag2x[flag2x>max_x]=0
        flag2x[flag2x!=0]=1

        # 处理垂直方向共视区域
        flag1y = self.phase_y1.copy()
        flag2y = self.phase_y2.copy()
        
        # 计算垂直相位的最大最小范围
        min_y1 = np.min(self.phase_y1)
        max_y1 = np.max(self.phase_y1)

        min_y2 = np.min(self.phase_y2)
        max_y2 = np.max(self.phase_y2)

        # 取两相机共同的垂直相位范围
        max_y = min(max_y1,max_y2)
        min_y = max(min_y1,min_y2)

        # 将不在共视范围的区域置0
        flag1y[flag1y<min_y]=0
        flag1y[flag1y>max_y]=0
        flag1y[flag1y!=0]=1

        flag2y[flag2y<min_y]=0
        flag2y[flag2y>max_y]=0
        flag2y[flag2y!=0]=1
        
        # 水平和垂直方向都有效的区域才是最终有效区域
        flag1 = flag1x*flag1y
        flag2 = flag2x*flag2y

        return flag1, flag2
    
    def transform_to_camera(self,flag,axis_world):
        """
        将世界坐标系下的点转换到相机坐标系
        
        参数:
            flag: 有效区域掩码
            axis_world: 世界坐标系下的点坐标
            
        返回:
            camera_axis: 相机坐标系下的点坐标(仅有效区域)
        """
        # 创建三通道掩码，用于筛选有效点
        temp = np.dstack((flag,flag,flag))

        # 将点坐标重新整形为适合矩阵运算的形式
        point_matrix = np.reshape(axis_world,(-1,3,1))
        point_matrix = np.squeeze(point_matrix,axis=2)

        # 应用旋转和平移变换到相机坐标系
        camera_axis = np.dot(self.Rs,np.transpose(point_matrix))+self.Ts

        # 转置结果并重新整形为原始形状
        camera_axis_trans = np.transpose(camera_axis)
        camera_axis = np.reshape(camera_axis_trans,(-1,np.shape(flag)[1],3))

        # 只保留有效区域的点
        camera_axis*=temp
        
        return camera_axis


    def calculate(self):
        """
        主要的三维重建计算流程
        
        基于相位图和相机标定信息，计算三维点云

        返回:
            result: 三维重建结果
            flag_result: 重建点的有效标记
        """
        # flag1,flag2 = self.get_flag()
        # 将归一化相位值转换为物理坐标
        self.transfer_to_world()

        # 获取公共视野
        # flag1,flag2 = self.get_flag()#主要用到flag1
        flag1 = self.flag1
        flag2 = self.flag2

        # 创建世界坐标系下的点
        r,c = np.shape(self.phase_x1)
        z = np.zeros((r,c))  # z初始值为0

        # 构建世界坐标系下的点坐标
        axis_world1 = np.dstack((self.phase_x1,self.phase_y1,z))
        axis_world2 = np.dstack((self.phase_x2,self.phase_y2,z))

        # x1_min = np.min(self.phase_x1)
        # x1_max = np.max(self.phase_x1)
        # x2_min = np.min(self.phase_x2)
        # x2_max = np.max(self.phase_x2)
        
        # y1_min = np.min(self.phase_y1)
        # y1_max = np.max(self.phase_y1)
        # y2_min = np.min(self.phase_y2)
        # y2_max = np.max(self.phase_y2)
        


        # 转换到相机坐标系下
        axis_camera1 = self.transform_to_camera(flag1,axis_world=axis_world1)
        axis_camera2 = self.transform_to_camera(flag2,axis_world=axis_world2)

        # 创建图形对象用于可视化
        fig2 = plt.figure()
        
        # 执行迭代优化计算三维点坐标
        result,flag_result = self.iterater_(axis_camera1=axis_camera1,axis_camera2=axis_camera2,flag1=flag1,flag2=flag2,fig=fig2)

        return result,flag_result

    def iterater_(self,axis_camera1,axis_camera2,flag1,flag2,fig):
        """
        迭代优化计算三维点坐标
        
        使用粒子群优化算法逐点优化深度值，生成三维点云
        
        参数:
            axis_camera1, axis_camera2: 两个相机坐标系下的点
            flag1, flag2: 有效区域掩码
            fig: 用于可视化的图形对象
            
        返回:
            result: 优化后的三维点坐标列表
            flag_iter: 标记哪些点被成功重建
        """
        
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(flag1)
        # plt.subplot(122)
        # plt.imshow(flag2)


        # #参数初始化
        # flag_iter = np.ones(np.shape(flag1))
        flag_iter=flag1
        # plt.figure()
        # plt.imshow(flag_iter,cmap='gray')
        # plt.show()
        
        # 设置深度搜索范围和参数
        max_z = 500.
        min_z=400.
        delta_z = 0.05
        iter_max = 40
        step=10 #稀疏度
        # result = [np.zeros((np.shape(flag1)[0],np.shape(flag1)[1],3))]
        result=[]


        # #使用连通域检索第一个点的位置
        # resval,flag1_thres = cv.threshold(flag1,0,255,cv.THRESH_BINARY)
        # flag1_thres = flag1_thres.astype(np.uint8)
        # contours, hierarchy = cv.findContours(flag1_thres,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
       
        # area=[]
        # for i in range(len(contours)):
        #     area.append(cv.contourArea(contours[i]))
        
        # max_idx = np.argmax(area)
        # for i in range(len(contours)):
        #     if i!=max_idx:
        #         cv.fillConvexPoly(flag_iter,contours[i],1)
        #     else:
        #         cv.fillConvexPoly(flag_iter,contours[i],0)
        
        # plt.figure()
        # plt.imshow(flag_iter*255)
        # plt.show()
        # rect = cv.minAreaRect(contours[max_idx])
        
        
        # #迭代
        # point_remain = [round(rect[0][0]),round(rect[0][1]),1.]#注意图像坐标和像素坐标的差别
        # point_remain = np.array(point_remain)
        # point_remain.shape = -1,3
       
        # num_remain=1

        
        
        # fig = plt.figure()
        # while(num_remain>0):
        #     point_remain_next=[]
        #     num_remain_next=0
        #     '''
        #     先判断边界，再进行计算
        #     '''
        #     for i in range(num_remain):
        #         #确定z的范围,（容易受噪声影响）
        #         # z_temp = result[int(point_remain[i,1]),int(point_remain[i,0]),2]
        #         z_temp=0
        #         if(z_temp!=0):
        #             max_z = z_temp+delta_z
        #             min_z = max(z_temp-delta_z,0)

        #         #当前点是否计算？主要是起始点
        #         if(flag_iter[int(point_remain[i,1]),int(point_remain[i,0])]==0):
        #             point = point_remain[i,:]
        #             point = np.array(point)
        #             point.shape=-1,1
        #             point_camera = np.dot(np.linalg.inv(self.M1),point)
        #             point_camera/=point_camera[2]
                    
        #             point_screen = axis_camera1[int(point[1,0]),int(point[0,0]),:]
        #             point_screen.shape = 3,-1

                    
        #             #使用粒子群算法获取深度值
        #             iter_me = pos_g(fig,self.M1,self.M2,self.D1,self.D2,self.Rs,self.Ts,self.Rc,self.Tc,self.plane_parameter,point_camera,point_screen,axis_camera2,iter_max,1,min_z,max_z)
        #             z=iter_me.run()
                    
        #             #更新参数
        #             # max_z = z+delta_z
        #             # min_z = z-delta_z
        #             # if min_z<0:
        #             #         min_z=0
        #             flag_iter[int(point[1,0]),int(point[0,0])]=1
                    
        #             #保存结果
        #             result[int(point[1,0]),int(point[0,0]),0]= point_camera[0,0]*z
        #             result[int(point[1,0]),int(point[0,0]),1]= point_camera[1,0]*z
        #             result[int(point[1,0]),int(point[0,0]),2]= z

        #             #print(z)
        #         #向上检索
        #         if (point_remain[i,1]-step>=0): #防止超出边界
        #             if(flag_iter[int(point_remain[i,1])-step,int(point_remain[i,0])]==0):
        #                 point = point_remain[i,:]
        #                 point = np.array(point)
        #                 point.shape = -1,1
        #                 point[1,0]-=step

        #                 point_camera = np.dot(np.linalg.inv(self.M1),point)
        #                 point_camera/=point_camera[2]

        #                 point_screen = axis_camera1[int(point[1,0]),int(point[0,0]),:]
        #                 point_screen.shape = 3,-1

        #                 iter_me = pos_g(fig,self.M1,self.M2,self.D1,self.D2,self.Rs,self.Ts,self.Rc,self.Tc,self.plane_parameter,point_camera,point_screen,axis_camera2,iter_max,1,min_z,max_z)
        #                 z=iter_me.run()

        #                 #更新参数
        #                 # max_z = z+delta_z
        #                 # min_z = z-delta_z
        #                 # if min_z<0:
        #                 #     min_z=0
        #                 flag_iter[int(point[1,0]),int(point[0,0])]=1
        #                 point_remain_next.append(np.transpose(point)[0,:])
        #                 num_remain_next+=1
                    
        #                 #保存结果
        #                 result[int(point[1,0]),int(point[0,0]),0]= point_camera[0,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),1]= point_camera[1,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),2]= z

        #                 #print(z)

        #         #向下检索
        #         if (point_remain[i,1]+step<np.shape(flag_iter)[0]): #防止超出边界
        #             if(flag_iter[int(point_remain[i,1])+step,int(point_remain[i,0])]==0):
        #                 point = point_remain[i,:] 
        #                 point = np.array(point)
        #                 point.shape = -1,1
        #                 point[1,0]+=step
        #                 point_camera = np.dot(np.linalg.inv(self.M1),point)
        #                 point_camera/=point_camera[2]

        #                 point_screen = axis_camera1[int(point[1,0]),int(point[0,0]),:]
        #                 point_screen.shape = 3,-1
        #                 iter_me = pos_g(fig,self.M1,self.M2,self.D1,self.D2,self.Rs,self.Ts,self.Rc,self.Tc,self.plane_parameter,point_camera,point_screen,axis_camera2,iter_max,1,min_z,max_z)
        #                 z=iter_me.run()

        #                 #更新参数
        #                 # max_z = z+delta_z
        #                 # min_z = z-delta_z
        #                 # if min_z<0:
        #                 #     min_z=0
        #                 flag_iter[int(point[1,0]),int(point[0,0])]=1
        #                 point_remain_next.append(np.transpose(point)[0,:])
        #                 num_remain_next+=1
                    
        #                 #保存结果
        #                 result[int(point[1,0]),int(point[0,0]),0]= point_camera[0,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),1]= point_camera[1,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),2]= z

        #                 #print(z)
                
        #         #向左检索
        #         if (point_remain[i,0]-step>=0): #防止超出边界
        #             if(flag_iter[int(point_remain[i,1]),int(point_remain[i,0])-step]==0):
        #                 point = point_remain[i,:]
        #                 point = np.array(point)
        #                 point.shape = -1,1
        #                 point[0,0]-=step

        #                 point_camera = np.dot(np.linalg.inv(self.M1),point)
        #                 point_camera/=point_camera[2]

        #                 point_screen = axis_camera1[int(point[1,0]),int(point[0,0]),:]
        #                 point_screen.shape = 3,-1
        #                 iter_me = pos_g(fig,self.M1,self.M2,self.D1,self.D2,self.Rs,self.Ts,self.Rc,self.Tc,self.plane_parameter,point_camera,point_screen,axis_camera2,iter_max,1,min_z,max_z)
        #                 z=iter_me.run()

        #                 #更新参数
        #                 # max_z = z+delta_z
        #                 # min_z = z-delta_z
        #                 # if min_z<0:
        #                 #     min_z=0
        #                 flag_iter[int(point[1,0]),int(point[0,0])]=1
        #                 point_remain_next.append(np.transpose(point)[0,:])
        #                 num_remain_next+=1
                    
        #                 #保存结果
                        
        #                 result[int(point[1,0]),int(point[0,0]),0]= point_camera[0,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),1]= point_camera[1,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),2]= z

        #                 #print(z)
        #         #向右检索
        #         if (point_remain[i,0]+step<np.shape(flag_iter)[1]): #防止超出边界
        #             if(flag_iter[int(point_remain[i,1]),int(point_remain[i,0])+step]==0):
        #                 point = point_remain[i,:]
        #                 point = np.array(point)
        #                 point.shape = -1,1
        #                 point[0,0]+=step

        #                 point_camera = np.dot(np.linalg.inv(self.M1),point)
        #                 point_camera/=point_camera[2]

        #                 point_screen = axis_camera1[int(point[1,0]),int(point[0,0]),:]
        #                 point_screen.shape = 3,-1
        #                 iter_me = pos_g(fig,self.M1,self.M2,self.D1,self.D2,self.Rs,self.Ts,self.Rc,self.Tc,self.plane_parameter,point_camera,point_screen,axis_camera2,iter_max,1,min_z,max_z)
        #                 z=iter_me.run()

        #                 #更新参数
        #                 # max_z = z+delta_z
        #                 # min_z = z-delta_z
        #                 # if min_z<0:
        #                 #     min_z=0
        #                 flag_iter[int(point[1,0]),int(point[0,0])]=1
        #                 point_remain_next.append(np.transpose(point)[0,:])
        #                 num_remain_next+=1
                    
        #                 #保存结果
        #                 result[int(point[1,0]),int(point[0,0]),0]= point_camera[0,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),1]= point_camera[1,0]*z
        #                 result[int(point[1,0]),int(point[0,0]),2]= z

        #                 #print(z)
        #     #更新参数
        #     point_remain = point_remain_next
        #     point_remain = np.array(point_remain)
        #     point_remain.shape = -1,3
        #     num_remain = num_remain_next
        # return result,flag_iter

        # 用于记录处理的点数量和临时深度范围
        num=0
        max_z_temp = max_z
        min_z_temp= min_z
        
        # 按步长采样图像点，减少计算量
        for i in range(0,flag1.shape[0],step):
            for j in range(0,flag1.shape[1],step):
                # 跳过无效区域
                if(flag_iter[i,j]==0):
                    continue
                
                # #映射表
                # if(self.mapping[i,j,0]!=0):
                #     min_z=self.mapping[i,j,0]-10
                # else:
                #     min_z=min_z_temp

                # if(self.mapping[i,j,1]!=0):
                #     max_z=self.mapping[i,j,1]+10
                # else:
                #     max_z = max_z_temp

                # 构建像素坐标点
                point =[j,i]
                
                point = np.array(point,dtype=np.float64)
                point.shape=-1,1

                #去畸变
                # point = cv.undistortPoints(np.expand_dims(point, axis=1),self.M1,self.D1,None,self.M1)
                # point = np.squeeze(point)


                # 构建齐次坐标
                point=[point[0],point[1],1]
                point = np.array(point,dtype=np.float64)
                point.shape=-1,1

                # 将像素坐标转换为相机坐标系下的射线方向
                point_camera=np.dot(np.linalg.inv(self.M1),point)
                point_camera/=point_camera[2]  # 归一化

                # 获取投影屏幕上的对应点
                point_screen = axis_camera1[int(i),int(j),:]
                point_screen.shape=3,-1
                
                # 使用粒子群优化算法估计深度值
                iter_me = pos_g(fig,self.M1,self.M2,self.D1,self.D2,self.Rs,self.Ts,self.Rc,self.Tc,self.plane_parameter,self.table_interpolation,point_camera,point_screen,axis_camera2,iter_max,1,min_z,max_z)
                z,qulity=iter_me.run()
                
                # 如果深度值有效(不是边界值)且质量足够好
                if(z!=0 and z!=max_z):# and qulity<0.5):
                    print('%d %f %f\n'%(num,z,qulity))

                    # 计算法线信息
                    c1  = -point_camera/np.linalg.norm(point_camera)  # 相机方向归一化
                    s1 = point_screen-point_camera*z  # 从点到屏幕的向量
                    s1/=np.linalg.norm(s1)  # 归一化
                    n1 = c1+s1  # 法线为相机方向和投影方向的和
                    n1/=np.linalg.norm(n1)  # 归一化法线
                    
                    # 创建结果数组，存储点的信息
                    # 包括：图像坐标、三维坐标、法线方向、质量评分
                    result_temp=np.zeros((1,9),dtype=np.double)

                    result_temp[0,0]=int(i)  # 图像y坐标
                    result_temp[0,1]=int(j)  # 图像x坐标
                    result_temp[0,2] =  point_camera[0]*z  # 三维点x坐标
                    result_temp[0,3] =  point_camera[1]*z  # 三维点y坐标
                    result_temp[0,4] =  z  # 三维点z坐标(深度)
                    result_temp[0,5] = n1[0]  # 法线x分量
                    result_temp[0,6] = n1[1]  # 法线y分量
                    result_temp[0,7] = n1[2]  # 法线z分量
                    result_temp[0,8] = qulity  # 重建质量评分
                    
                    result_temp = np.array(result_temp)
                    result.append(result_temp)
                    # result[int(i),int(j),0]= point_camera[0]*z
                    # result[int(i),int(j),1]= point_camera[1]*z
                    # result[int(i),int(j),2]= z
                    num+=1

        return result,flag_iter
    



    

        
        


        



