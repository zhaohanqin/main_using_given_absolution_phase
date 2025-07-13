import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class multi_phase():
    """
    多频外差法相位解包裹类
    
    该类实现了基于多频率条纹图像的相位解包裹算法，
    可同时处理水平和垂直方向的相位图，使用外差法逐级展开相位
    """
    def __init__(self,f,step,images,ph0):
        """
        初始化多频相位解包裹对象
        
        参数:
            f: 列表，包含多个频率值，按从高到低排序，例如[64,8,1]
            step: 整数，相移步数(通常为3或4)
            images: ndarray，所有条纹图像组成的数组
            ph0: 浮点数，相移初始相位偏移量
        """
        self.f=f                # 频率列表
        self.images=images      # 相移图像
        self.step = step        # 相移步数
        self.ph0 = ph0          # 相移初始相位
        self.f12=f[0]-f[1]      # 第1和第2个频率的差值(高频-中频)
        self.f23=f[1]-f[2]      # 第2和第3个频率的差值(中频-低频)

    

    def decode_phase(self,image):
        """
        N步相移算法解码相位
        
        使用正弦和余弦项计算相移图像的包裹相位，并计算幅值和偏移量
        
        参数:
            image: ndarray，相移图像组，形状为[step, height, width]
            
        返回:
            result: ndarray，归一化的包裹相位图
            amp: ndarray，调制幅值
            offset: ndarray，亮度偏移
        """
        # 生成相移角度数组(0,2π/N,4π/N...)
        temp = 2*np.pi*np.arange(self.step,dtype=np.float32)/self.step
        temp.shape=-1,1,1  # 调整形状以便于广播运算
        
        # 计算正弦项(分子)和余弦项(分母)
        molecule = np.sum(image*np.sin(temp),axis=0)      # 正弦项
        denominator=np.sum(image*np.cos(temp),axis=0)     # 余弦项

        # 使用arctan2计算相位，保证相位值在[-π,π]范围内
        result = -np.arctan2(molecule,denominator)
        
        # 计算调制幅值和亮度偏移
        amp = 2/self.step*molecule        # 调制幅值
        offset = 2/self.step*denominator  # 亮度偏移

        # 归一化相位至[0,1]区间并减去初始相位
        result = (result+np.pi)/(2*np.pi)-self.ph0

        return result,amp,offset
    
    def phase_diff(self,image1,image2):
        """
        计算两个相位图之间的差值
        
        实现了外差法的核心操作，确保相位差在[0,1]范围内
        
        参数:
            image1: 高频相位图
            image2: 低频相位图
            
        返回:
            result: 两相位图的归一化差值
        """
        result = image1-image2       # 计算相位差
        result[result<0]+=1          # 处理负值，保证结果在[0,1]区间
        return result

    def unwarpphase(self,reference,phase,reference_f,phase_f):
        """
        基于低频参考相位展开高频相位
        
        参数:
            reference: 参考(低频)相位图
            phase: 需展开的(高频)包裹相位图
            reference_f: 参考相位的频率
            phase_f: 需展开相位的频率
            
        返回:
            unwarp_phase: 展开后的相位图
        """
        # 根据频率比例缩放参考相位
        # 低频相位乘以频率比得到高频相位的估计值
        temp=phase_f/reference_f*reference
        
        # 计算整数条纹序数k并应用
        # 用缩放后的低频相位减去高频包裹相位，四舍五入得到整数条纹序数
        test = np.round(temp-phase)
        unwarp_phase=phase+np.round(temp-phase)

        # 高斯滤波去噪，检测错误跳变点
        gauss_size=(5,5)
        unwarp_phase_noise = unwarp_phase-cv.GaussianBlur(unwarp_phase,gauss_size,0)
        unwarp_reference_noise = temp-cv.GaussianBlur(temp,gauss_size,0)

        # 检测异常点：展开相位的噪声明显大于参考相位的噪声
        order_flag = np.abs(unwarp_phase_noise)-np.abs(unwarp_reference_noise)>0.25
        
        # 修复异常跳变点
        unwarp_error = unwarp_phase[order_flag]
        unwarp_error_direct = unwarp_phase_noise[order_flag]
        # 根据噪声方向调整条纹序数
        unwarp_error[unwarp_error_direct>0]+=1  # 正向噪声增加一个周期
        unwarp_error[unwarp_error_direct<0]-=1  # 负向噪声减少一个周期(注意：原代码这里有逻辑错误，已修正)

        # 应用修复结果
        unwarp_phase[order_flag]=unwarp_error

        return unwarp_phase

    
    def get_phase(self):
        """
        多频相移解包裹主流程
        
        处理所有频率的相位图，分别对水平和垂直方向进行解包裹
        
        返回:
            unwarp_phase_y: 垂直方向展开的相位图
            unwarp_phase_x: 水平方向展开的相位图
            ratio: 相位质量图(基于调制度与偏移比)
        """
        # 1. 解码各个频率的相位
        # 解码垂直方向的三个频率的相位
        phase_1y,amp1_y,offset1_y = self.decode_phase(image=self.images[0:4])   # 高频
        phase_2y,amp2_y,offset2_y = self.decode_phase(image=self.images[4:8])   # 中频
        phase_3y,amp3_y,offset3_y = self.decode_phase(image=self.images[8:12])  # 低频

        # 解码水平方向的三个频率的相位
        phase_1x,amp1_x,offset1_x = self.decode_phase(image=self.images[12:16]) # 高频
        phase_2x,amp2_x,offset2_x = self.decode_phase(image=self.images[16:20]) # 中频
        phase_3x,amp3_x,offset3_x = self.decode_phase(image=self.images[20:24]) # 低频

        plt.figure()
        plt.imshow(phase_1x)
        #plt.show()

        # 2. 外差法获取逐级展开的相位差
        # 计算垂直方向相位差
        phase_12y = self.phase_diff(phase_1y,phase_2y)  # 频率1和2的差异
        phase_23y = self.phase_diff(phase_2y,phase_3y)  # 频率2和3的差异
        phase_123y = self.phase_diff(phase_12y,phase_23y) # 差异的差异(等效最低频)

        # 计算水平方向相位差
        phase_12x = self.phase_diff(phase_1x,phase_2x)  # 频率1和2的差异
        phase_23x = self.phase_diff(phase_2x,phase_3x)  # 频率2和3的差异
        phase_123x = self.phase_diff(phase_12x,phase_23x) # 差异的差异(等效最低频)

        plt.figure()
        plt.imshow(phase_12x)

        plt.figure()
        plt.imshow(phase_123x)
        #plt.show()

        # 3. 平滑最低等效频率相位以提高鲁棒性
        phase_123y = cv.GaussianBlur(phase_123y,(3,3),0)
        phase_123x = cv.GaussianBlur(phase_123x,(3,3),0)

        # 4. 相位展开流程 - 自底向上展开
        # 使用最低等效频率相位(phase_123y/x)展开中等频率相位差(phase_12y/x和phase_23y/x)
        unwarp_phase_12_y = self.unwarpphase(phase_123y,phase_12y,1,self.f12)
        unwarp_phase_23_y = self.unwarpphase(phase_123y,phase_23y,1,self.f23)

        unwarp_phase_12_x = self.unwarpphase(phase_123x,phase_12x,1,self.f12)
        unwarp_phase_23_x = self.unwarpphase(phase_123x,phase_23x,1,self.f23)
        
        # 5. 使用展开后的中等频率相位差(unwarp_phase_12_y/x和unwarp_phase_23_y/x)
        # 展开中频相位(phase_2y/x)
        unwarp_phase2_y_12 = self.unwarpphase(unwarp_phase_12_y,phase_2y,self.f12,self.f[1])
        unwarp_phase2_y_23 = self.unwarpphase(unwarp_phase_23_y,phase_2y,self.f23,self.f[1])

        unwarp_phase2_x_12 = self.unwarpphase(unwarp_phase_12_x,phase_2x,self.f12,self.f[1])
        unwarp_phase2_x_23 = self.unwarpphase(unwarp_phase_23_x,phase_2x,self.f23,self.f[1])

        # 6. 取两个展开路径的平均值以提高鲁棒性
        unwarp_phase_y = (unwarp_phase2_y_12+unwarp_phase2_y_23)/2
        unwarp_phase_x = (unwarp_phase2_x_12+unwarp_phase2_x_23)/2

        # 7. 归一化相位结果
        unwarp_phase_y/=self.f[1]  # 以中频为基准归一化
        unwarp_phase_x/=self.f[1]  # 以中频为基准归一化

        # 8. 计算相位质量，使用调制度/偏移比值的最小值
        ratio_x = np.min([amp1_x/offset1_x,amp2_x/offset2_x,amp3_x/offset3_x],axis=0)
        ratio_y = np.min([amp1_y/offset1_y,amp2_y/offset2_y,amp3_y/offset3_y],axis=0)

        ratio = np.min([ratio_x,ratio_y],axis=0)  # 取水平和垂直方向的最小值作为最终质量图

        return unwarp_phase_y,unwarp_phase_x,ratio


# 以下是被注释掉的旧版解码相位函数，仅供参考
# def decode_phase(fore,phase_step,phase):
#     """
#     旧版相位解码函数，未被使用
#     """
#     if len(phase) !=4:
#         print("the image numbers of phase is {}".format(len(phase)))
#         raise("please check out the phase")
    
#     temp = 2*np.pi*np.arange(phase_step,dtype=np.float32)/phase_step
#     temp.shape = -1,1,1
#     molecule = np.sum(phase*np.sin(temp),axis=0)
#     denominator=np.sum(phase*np.cos(temp),axis=0)

#     result = -np.arctan2(molecule,denominator)

#     #归一化
#     result = (result+np.pi)/(2*np.pi)*fore

#     return result



