import cv2 as cv

def read_cali(file_path):
    """
    读取相机和投影仪标定参数
    
    从XML文件中读取立体视觉系统的标定参数，包括相机内参、畸变系数和相机/投影仪之间的转换关系。
    这些参数对于结构光三维重建至关重要，用于实现从二维相位到三维点云的转换。
    
    参数:
        file_path: 包含标定参数的XML文件路径
        
    返回:
        M1: 第一个相机的内参矩阵，3x3矩阵，包含焦距和主点坐标
        M2: 第二个相机/投影仪的内参矩阵
        D1: 第一个相机的畸变系数，包含径向和切向畸变参数
        D2: 第二个相机/投影仪的畸变系数
        Rc: 从相机2到相机1的旋转矩阵，描述两个相机坐标系的相对旋转
        Rs: 从显示屏/投影仪到相机1的旋转矩阵
        Tc: 从相机2到相机1的平移向量，描述两个相机坐标系的相对平移
        Ts: 从显示屏/投影仪到相机1的平移向量
        plane_parameter: 投影平面参数，用于三维重建中的光线交点计算
    """
    # 打开XML文件并读取标定参数
    file = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    
    # 读取相机内参矩阵
    M1 = file.getNode("M1").mat()  # 相机1内参矩阵
    M2 = file.getNode("M2").mat()  # 相机2/投影仪内参矩阵
    
    # 读取相机畸变系数
    D1 = file.getNode("Dist1").mat()  # 相机1畸变系数
    D2 = file.getNode("Dist2").mat()  # 相机2/投影仪畸变系数
    
    # 读取坐标系转换参数
    Rc = file.getNode("Rc").mat()  # 相机2到相机1的旋转矩阵
    Rs = file.getNode("Rs").mat()  # 显示屏/投影仪到相机1的旋转矩阵
    Tc = file.getNode("Tc").mat()  # 相机2到相机1的平移向量
    Ts = file.getNode("Ts").mat()  # 显示屏/投影仪到相机1的平移向量
    
    # 读取投影平面参数，用于光线与平面的交点计算
    plane_parameter = file.getNode("plane_parameter").mat()  # 投影平面参数

    return M1, M2, D1, D2, Rc, Rs, Tc, Ts, plane_parameter