import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Vector3
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations

from transforms3d.quaternions import quat2mat
from unitree_go.msg import SportModeState
import numpy as np
import yaml
import torch
import os

# 导入 process_cloud 中定义的 jit 加速函数

class Repuber(Node):
    def __init__(self):
        super().__init__('sensor_transformer')
        self.imu_sub = self.create_subscription(SportModeState, '/sportmodestate', self.imu_callback, 50)  # 创建IMU订阅者
        self.cloud_sub = self.create_subscription(PointCloud2, '/utlidar/cloud', self.cloud_callback, 50)
       
        self.imu_raw_pub = self.create_publisher(Imu, '/utlidar/transformed_raw_imu', 50)
        self.imu_pub = self.create_publisher(Imu, '/utlidar/transformed_imu', 50)
        self.cloud_pub = self.create_publisher(PointCloud2, '/utlidar/transformed_cloud', 50)
        
        # 声明参数
        self.declare_parameter('use_gpu', False)  # 默认不使用GPU加速
        self.declare_parameter('num_points', 150)  # 新增采样点数参数
        
        
        # 获取参数值
        self.use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.num_points = self.get_parameter('num_points').get_parameter_value().integer_value

        self.imu_stationary_list = []
        
        self.time_stamp_offset = 0
        self.time_stamp_offset_set = False
        
        self.cam_offset = 0.046825 

        # Load calibration data
        calib_data = calib_data = {
                'acc_bias_x': 0.0,
                'acc_bias_y': 0.0,
                'acc_bias_z': 0.0,
                'ang_bias_x': 0.0,
                'ang_bias_y': 0.0,
                'ang_bias_z': 0.0,
                'ang_z2x_proj': 0.15,
                'ang_z2y_proj': -0.28
            }
        try:
            home_path = os.path.expanduser('~')
            calib_file_path = os.path.join(home_path, '桌面/go2_imu_calib_data.yaml')
            calib_file = open(calib_file_path, 'r')
            calib_data = yaml.load(calib_file, Loader=yaml.FullLoader)
            print("imu_calib.yaml loaded")
            calib_file.close()
        except:
            print("imu_calib.yaml not found, using defualt values")
            
        self.acc_bias_x = calib_data['acc_bias_x']
        self.acc_bias_y = calib_data['acc_bias_y']
        self.acc_bias_z = calib_data['acc_bias_z']
        self.ang_bias_x = calib_data['ang_bias_x']
        self.ang_bias_y = calib_data['ang_bias_y']
        self.ang_bias_z = calib_data['ang_bias_z']
        self.ang_z2x_proj = calib_data['ang_z2x_proj']
        self.ang_z2y_proj = calib_data['ang_z2y_proj']
                
        self.body2cloud_trans = TransformStamped()
        self.body2cloud_trans.header.stamp = self.get_clock().now().to_msg()
        self.body2cloud_trans.header.frame_id = "body"
        self.body2cloud_trans.child_frame_id = "utlidar_lidar_1"
        self.body2cloud_trans.transform.translation.x = 0.0
        self.body2cloud_trans.transform.translation.y = 0.0
        self.body2cloud_trans.transform.translation.z = 0.0
        quat = tf_transformations.quaternion_from_euler(0, 2.87820258505555555556, 0)
        self.body2cloud_trans.transform.rotation.x = quat[0]
        self.body2cloud_trans.transform.rotation.y = quat[1]
        self.body2cloud_trans.transform.rotation.z = quat[2]
        self.body2cloud_trans.transform.rotation.w = quat[3]
        
        self.body2imu_trans = TransformStamped()  # 创建机体到IMU的变换
        self.body2imu_trans.header.stamp = self.get_clock().now().to_msg()  # 设置时间戳
        self.body2imu_trans.header.frame_id = "body"  # 设置父坐标系
        self.body2imu_trans.child_frame_id = "go2_imu_1"  # 设置子坐标系
        self.body2imu_trans.transform.translation.x = 0.0  # 设置X轴平移
        self.body2imu_trans.transform.translation.y = 0.0  # 设置Y轴平移
        self.body2imu_trans.transform.translation.z = 0.0  # 设置Z轴平移
        quat = tf_transformations.quaternion_from_euler(0, 0, 0)  # 计算欧拉角到四元数的转换
        self.body2imu_trans.transform.rotation.x = quat[0]  # 设置四元数X分量
        self.body2imu_trans.transform.rotation.y = quat[1]  # 设置四元数Y分量
        self.body2imu_trans.transform.rotation.z = quat[2]  # 设置四元数Z分量
        self.body2imu_trans.transform.rotation.w = quat[3]  # 设置四元数W分量
        
        self.x_filter_min = -0.7
        self.x_filter_max =  -0.1               # 原值0.0       
        self.y_filter_min = -0.3
        self.y_filter_max = 0.3
        self.z_filter_min = -0.6 - self.cam_offset
        self.z_filter_max = 0 - self.cam_offset

        rclpy.spin(self)
                
    def is_in_filter_box(self, point):
        # Check if the point is in the filter box
        is_in_box = point[0] > self.x_filter_min and \
                    point[0] < self.x_filter_max and \
                    point[1] > self.y_filter_min and \
                    point[1] < self.y_filter_max and \
                    point[2] > self.z_filter_min and \
                    point[2] < self.z_filter_max
        return is_in_box

    def cloud_callback(self, data):  # 定义云回调函数，接收数据
        
        
        if not self.time_stamp_offset_set:  # 如果时间戳偏移量未设置
            self.time_stamp_offset = self.get_clock().now().nanoseconds - Time.from_msg(data.header.stamp).nanoseconds  # 计算时间戳偏移量
            self.time_stamp_offset_set = True  # 设置时间戳偏移量为已设置

        # 读取点云数据
        cloud_arr = pc2.read_points_list(data)
        points = np.array(cloud_arr)
        
        if self.use_gpu:
            # GPU加速
         
            # 转换为 PyTorch 张量，并指定 float32 类型
            points_tensor = torch.from_numpy(points).float().to('cuda')  # <--- 明确设置为 float32

            # 提取坐标和强度
            points_xyz = points_tensor[:, :3]
            tags = points_tensor[:, 3:]

            # 获取变换参数
            transform = self.body2cloud_trans.transform
            quat = torch.tensor([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z],
                                device='cuda', dtype=torch.float32)  # <--- 设置 dtype
            translation = torch.tensor([transform.translation.x, transform.translation.y, transform.translation.z],
                                    device='cuda', dtype=torch.float32)  # <--- 设置 dtype

            # 应用四元数旋转 + 平移
            transformed_points = quat_rotate_inverse_torch(quat, points_xyz) + translation

            # 调整 Z 偏移
            transformed_points[:, 2] -= self.cam_offset
            transformed_points = torch.cat((transformed_points, tags), dim=1)
            num_points = self.num_points
            downsampled_tensor = farthest_point_sampling(transformed_points, num_points).squeeze(0)

            # 转换回 CPU 并发布
            downsampled_points = downsampled_tensor.cpu().numpy()
            
            elevated_cloud = pc2.create_cloud(data.header, data.fields, downsampled_points.tolist())
        
        else:
            # 常规处理
            
            transform = self.body2cloud_trans.transform  # 获取身体到点云的变换
            mat = quat2mat(np.array([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z]))  # 将四元数转换为旋转矩阵
            translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])  # 获取平移向量
            
            transformed_points = points  # 初始化变换后的点
            transformed_points[:, 0:3] = points[:, 0:3] @ mat.T + translation  # 应用旋转和位移变换
            transformed_points[:, 2] -= self.cam_offset  # 调整Z轴坐标，减去相机偏移
            i = 0  # 初始化索引
            remove_list = []  # 初始化待移除的点索引列表
            transformed_points = transformed_points.tolist()  # 将变换后的点转换为列表



            for i in range(len(transformed_points)):  # 遍历所有变换后的点
                transformed_points[i][4] = int(transformed_points[i][4])  # 将点的强度值转换为整数
                if self.is_in_filter_box(transformed_points[i]):  # 检查点是否在过滤框内
                    remove_list.append(i)  # 如果在过滤框内，添加到待移除列表


            remove_list.sort(reverse=True)  # 反向排序待移除列表，以便从后向前删除

            for id_to_remove in remove_list:  # 遍历待移除列表
                del transformed_points[id_to_remove]  # 删除变换后的点
            
            elevated_cloud = pc2.create_cloud(data.header, data.fields, transformed_points)  # 创建新的点云

        # 构建新的 PointCloud2 消息
        elevated_cloud.header.stamp = Time(nanoseconds=Time.from_msg(elevated_cloud.header.stamp).nanoseconds + self.time_stamp_offset).to_msg()  # 设置点云的时间戳
        elevated_cloud.header.frame_id = "body"  # 设置点云的坐标系
        elevated_cloud.is_dense = data.is_dense  # 设置点云的稠密性

        self.cloud_pub.publish(elevated_cloud)

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        w, x, y, z = q.unbind(-1)
        zeros = torch.zeros_like(w)
        ones = torch.ones_like(w)
        return torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
            2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
            2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y
        ], dim=-1).view(-1, 3, 3)

    def grid_downsample(self, points, grid_size=0.05):
        grid_dict = {}
        filtered_points = []

        for point in points:
            x, y, z, *rest = point
            if not (self.x_filter_min < x < self.x_filter_max and
                    self.y_filter_min < y < self.y_filter_max and
                    self.z_filter_min < z < self.z_filter_max):
                continue

            # 计算所属栅格
            ix = int(x / grid_size)
            iy = int(y / grid_size)
            iz = int(z / grid_size)
            key = (ix, iy, iz)

            if key not in grid_dict:
                grid_dict[key] = point
                filtered_points.append(point)

        return filtered_points
    


    def imu_callback(self, data):  # IMU回调函数   
        
        # 把宇树时间戳的TimeSpec格式转化成ros2的Time格式
        ros_time = Time(
            seconds=data.stamp.sec,
            nanoseconds=data.stamp.nanosec
        )
        
        transformed_orientation = np.zeros(4)  # 创建旋转四元数
        transformed_orientation[0] = float(data.imu_state.quaternion[1])  # 设置X分量
        transformed_orientation[1] = float(data.imu_state.quaternion[2])  # 设置Y分量
        transformed_orientation[2] = float(data.imu_state.quaternion[3])  # 设置Z分量
        transformed_orientation[3] = float(data.imu_state.quaternion[0])  # 设置W分量

        
        x = float(data.imu_state.gyroscope[0])  # 获取角速度X分量  
        y = float(data.imu_state.gyroscope[1])  # 获取角速度Y分量
        z = float(data.imu_state.gyroscope[2])  # 获取角速度Z分量

        x2 = x - self.ang_bias_x  # 应用X轴偏差
        y2 = y - self.ang_bias_y  # 应用Y轴偏差
        z2 = z - self.ang_bias_z  # 应用Z轴偏差
        
        x_comp_rate = self.ang_z2x_proj  # 获取Z到X投影率
        y_comp_rate = self.ang_z2y_proj  # 获取Z到Y投影率
        
        x2 += x_comp_rate * z2  # 应用Z到X投影
        y2 += y_comp_rate * z2  # 应用Z到Y投影
        
        transformed_angular_velocity = Vector3()  # 创建角速度向量
        transformed_angular_velocity.x = x2  # 设置X角速度
        transformed_angular_velocity.y = y2  # 设置Y角速度
        transformed_angular_velocity.z = z2  # 设置Z角速度
        
        acc_x = float(data.imu_state.accelerometer[0])  # 获取线加速度X分量   
        acc_y = float(data.imu_state.accelerometer[1])  # 获取线加速度Y分量
        acc_z = float(data.imu_state.accelerometer[2])  # 获取线加速度Z分量
        
        transformed_linear_acceleration = Vector3()  # 创建线加速度向量
        transformed_linear_acceleration.x = acc_x - self.acc_bias_x  # 设置X加速度
        transformed_linear_acceleration.y = acc_y - self.acc_bias_y  # 设置Y加速度
        transformed_linear_acceleration.z = acc_z - self.acc_bias_z  # 设置Z加速度
        
        transformed_imu = Imu()  # 创建IMU消息
        transformed_imu.header.stamp = ros_time.to_msg()
        transformed_imu.header.frame_id = 'body'  # 设置坐标系
        transformed_imu.orientation.x = transformed_orientation[0]  # 设置姿态X分量
        transformed_imu.orientation.y = transformed_orientation[1]  # 设置姿态Y分量
        transformed_imu.orientation.z = transformed_orientation[2]  # 设置姿态Z分量
        transformed_imu.orientation.w = transformed_orientation[3]  # 设置姿态W分量
        transformed_imu.angular_velocity = transformed_angular_velocity  # 设置角速度
        transformed_imu.linear_acceleration = transformed_linear_acceleration  # 设置线加速度
        
        # 应用时间偏移到转换后的IMU消息
        # 新时间戳 = 原始时间戳 + 时间偏移量
        
        transformed_imu.header.stamp = Time(nanoseconds=Time.from_msg(transformed_imu.header.stamp).nanoseconds + self.time_stamp_offset).to_msg()
        

        self.imu_raw_pub.publish(transformed_imu)  # 发布原始IMU数据
        
        transformed_imu.orientation.x = 0.0  # 重置姿态X分量
        transformed_imu.orientation.y = 0.0  # 重置姿态Y分量
        transformed_imu.orientation.z = 0.0  # 重置姿态Z分量
        transformed_imu.orientation.w = 1.0  # 重置姿态W分量
        
        transformed_imu.linear_acceleration.x = 0.0  # 重置加速度X分量
        transformed_imu.linear_acceleration.y = 0.0  # 重置加速度Y分量
        transformed_imu.linear_acceleration.z = 0.0  # 重置加速度Z分量
        
        self.imu_pub.publish(transformed_imu)  # 发布转换后的IMU数据

        
@torch.jit.script
def quat_rotate_inverse_torch(q_original: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    支持二维及多维输入的四元数逆旋转
    参数:
        q_original: 四元数张量 (...,4)
        v: 向量张量 (...,3)
    返回:
        旋转后的向量 (...,3)
    """
    # 分离四元数实部与虚部
    q_w = q_original[..., 0]
    q_vec = -q_original[..., 1:4]  # 共轭处理
    
    # 扩展维度以支持广播
    q_w_expanded = q_w.unsqueeze(-1)  # (...,1)
    q_vec_cross = q_vec.unsqueeze(-2)  # (...,1,3) 用于cross计算
    
    # 计算旋转分量
    a = v * (2.0 * q_w_expanded.pow(2) - 1.0)  # 标量系数乘法
    
    # 交叉项计算
    cross_term = torch.cross(q_vec_cross, v, dim=-1)
    b = cross_term.squeeze(-2) * (2.0 * q_w_expanded)
    
    # 点积项计算
    dot_term = (q_vec * v).sum(dim=-1, keepdim=True)  # (...,1)
    c = q_vec * (2.0 * dot_term)
    
    # 组合各分量
    return a - b + c



def farthest_point_sampling(point_cloud, sample_size):
    """
    Sample points using the farthest point sampling algorithm.
    Args:
        point_cloud: Tensor of shape (num_envs, 1, num_points,1, 3) or with additional features (e.g., [1325, 6])
        sample_size: Number of points to sample
    Returns:
        Downsampled point cloud of shape (num_envs, 1, sample_size, d), where d is the feature dimensionality
    """
    # 判断是否是批量模式
    if point_cloud.dim() == 5:
        # 如果输入是5维张量（批量点云），获取环境数和点数
        num_envs, _, num_points, _, _ = point_cloud.shape
        points = point_cloud[:, 0, :, 0]  # 提取每个环境下的所有点（去掉多余的维度）
    elif point_cloud.dim() == 2 and point_cloud.size(1) >= 3:
        # 如果输入是二维张量（如 [1325, 6]），直接处理
        # 处理二维输入，例如 torch.Size([1325, 6])
        num_envs = 1  # 只有一个环境
        num_points = point_cloud.size(0)  # 点的数量
        points = point_cloud  # 保留全部维度
    else:
        # 输入维度不支持，抛出异常
        raise ValueError("Unsupported input dimensions")

    device = point_cloud.device  # 获取输入张量所在的设备（CPU或GPU）
    result = []  # 用于存储每个环境采样后的点

    for env_idx in range(num_envs):
        # 针对每个环境分别采样
        current_points = points[env_idx] if point_cloud.dim() == 5 else points  # 取出当前环境的点

        # 初始化采样索引，首先随机选一个点作为第一个采样点
        sampled_indices = torch.zeros(sample_size, dtype=torch.long, device=device)  # 存储采样点的索引
        sampled_indices[0] = torch.randint(0, num_points, (1,), device=device)  # 随机选一个点作为起始点

        # 计算所有点到第一个采样点的距离（只用前三个坐标）
        distances = torch.norm(current_points[:, :3] - current_points[sampled_indices[0], :3], dim=1)

        # 迭代选择最远的点
        for i in range(1, sample_size):
            # 选择距离当前已采样点中最远的点
            sampled_indices[i] = torch.argmax(distances)

            # 更新每个点到已采样点集合的最小距离
            if i < sample_size - 1:
                new_distances = torch.norm(current_points[:, :3] - current_points[sampled_indices[i], :3], dim=1)
                distances = torch.min(distances, new_distances)

        # 根据采样索引获取采样后的点（保留所有特征维度）
        sampled_points = current_points[sampled_indices]
        result.append(sampled_points)  # 将采样结果加入结果列表

    return torch.stack(result)  # 返回所有环境的采样结果，堆叠成一个张量



def main(args=None):
    rclpy.init(args=args)

    transform_node = Repuber()

    rclpy.spin(transform_node)

    Repuber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
