#!/usr/bin/env python
import rclpy  # 导入ROS2 Python客户端库
from rclpy.node import Node  # 导入ROS2节点类
from rclpy.time import Time  # 导入ROS2时间类
from sensor_msgs.msg import Imu  # 导入IMU消息类型
from sensor_msgs.msg import PointCloud2, PointField  # 导入点云消息类型
from geometry_msgs.msg import TransformStamped, Vector3  # 导入变换和向量消息类型
import sensor_msgs_py.point_cloud2 as pc2  # 导入点云处理工具

# 修复 NumPy 兼容性问题：在导入 transforms3d 之前添加 np.float 别名
import numpy as np  # 导入numpy库
if not hasattr(np, 'float'):
    np.float = np.float64
    np.int = np.int_
    np.complex = np.complex_
    np.bool = np.bool_

import tf_transformations  # 导入TF变换工具

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from transforms3d.quaternions import quat2mat  # 导入四元数到矩阵转换工具

from unitree_go.msg import SportModeState

from copy import deepcopy  # 导入深拷贝工具
import yaml  # 导入YAML解析库

import os  # 导入操作系统接口

class Repuber(Node):  # 定义传感器转换节点类
    def __init__(self):  # 初始化方法
        super().__init__('transform_hesai_on_pc2')  # 调用父类初始化方法
        # 创建匹配的QoS配置
        matching_imu_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,  # 修改为RELIABLE以匹配发布者
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
)
        self.imu_sub = self.create_subscription(SportModeState, '/sportmodestate', self.imu_callback, 50)  # 创建IMU订阅者
        self.cloud_sub = self.create_subscription(PointCloud2, '/lidar_points', self.cloud_callback, 50)  # 创建点云订阅者
        # self.cloud_sub = self.create_subscription(PointCloud2, '/cloud_result', self.cloud_callback, 50)  # 创建点云订阅者(降采样)
       
        self.imu_raw_pub = self.create_publisher(Imu, '/hesai_go2/transformed_raw_imu', 50)  # 创建原始IMU发布者
        self.imu_pub = self.create_publisher(Imu, '/hesai_go2/transformed_imu', 50)  # 创建转换后IMU发布者
        self.cloud_pub = self.create_publisher(PointCloud2, '/hesai_go2/transformed_cloud', 50)  # 创建转换后点云发布者

        self.imu_stationary_list = []  # 初始化IMU静止列表
        
        self.hesai_time_stamp_offset = 0  # 初始化hesai雷达时间戳偏移
        self.hesai_time_stamp_offset_set = False  # 初始化hesai雷达时间戳偏移设置标志
        self.go2imu_time_stamp_offset = 0  # 初始化go2自带IMU时间戳偏移
        self.go2imu_time_stamp_offset_set = False  # 初始化go2自带IMU时间戳偏移设置标志
        
        self.cam_offset = 0.2908  # 设置相机偏移量

        # 加载标定数据
        calib_data = calib_data = {  # 设置默认标定数据
                'acc_bias_x': 0.0,  # 加速度X轴偏差
                'acc_bias_y': 0.0,  # 加速度Y轴偏差
                'acc_bias_z': 0.0,  # 加速度Z轴偏差
                'ang_bias_x': 0.0,  # 角速度X轴偏差
                'ang_bias_y': 0.0,  # 角速度Y轴偏差
                'ang_bias_z': 0.0,  # 角速度Z轴偏差
                'ang_z2x_proj': 0.15,  # Z轴到X轴投影
                'ang_z2y_proj': -0.28  # Z轴到Y轴投影
            }
        try:  # 尝试加载标定文件
            home_path = os.path.expanduser('~')  # 获取用户主目录
            calib_file_path = os.path.join(home_path, 'Desktop/go2_imu_calib_data.yaml')  # 构建标定文件路径
            calib_file = open(calib_file_path, 'r')  # 打开标定文件
            calib_data = yaml.load(calib_file, Loader=yaml.FullLoader)  # 加载标定数据
            print("go2_imu_calib.yaml loaded")  # 打印加载成功信息
            calib_file.close()  # 关闭文件
        except:  # 加载失败时使用默认值
            print("go2_imu_calib.yaml not found, using defualt values")  # 打印使用默认值信息
            
        self.acc_bias_x = calib_data['acc_bias_x']  # 设置加速度X轴偏差
        self.acc_bias_y = calib_data['acc_bias_y']  # 设置加速度Y轴偏差
        self.acc_bias_z = calib_data['acc_bias_z']  # 设置加速度Z轴偏差
        self.ang_bias_x = calib_data['ang_bias_x']  # 设置角速度X轴偏差
        self.ang_bias_y = calib_data['ang_bias_y']  # 设置角速度Y轴偏差
        self.ang_bias_z = calib_data['ang_bias_z']  # 设置角速度Z轴偏差
        self.ang_z2x_proj = calib_data['ang_z2x_proj']  # 设置Z轴到X轴投影
        self.ang_z2y_proj = calib_data['ang_z2y_proj']  # 设置Z轴到Y轴投影
                
        self.body2cloud_trans = TransformStamped()  # 创建机体到点云的变换
        self.body2cloud_trans.header.stamp = self.get_clock().now().to_msg()  # 设置时间戳
        self.body2cloud_trans.header.frame_id = "body"  # 设置父坐标系
        self.body2cloud_trans.child_frame_id = "hesai_lidar_1"  # 设置子坐标系
        self.body2cloud_trans.transform.translation.x = 0.0  # 设置X轴平移
        self.body2cloud_trans.transform.translation.y = 0.0  # 设置Y轴平移
        self.body2cloud_trans.transform.translation.z = 0.0  # 设置Z轴平移
        # quat = tf_transformations.quaternion_from_euler(0, 0, 1.5708)  # 计算欧拉角到四元数的转换
        quat = tf_transformations.quaternion_from_euler(0, 0, 0)  # 计算欧拉角到四元数的转换
        self.body2cloud_trans.transform.rotation.x = quat[0]  # 设置四元数X分量
        self.body2cloud_trans.transform.rotation.y = quat[1]  # 设置四元数Y分量
        self.body2cloud_trans.transform.rotation.z = quat[2]  # 设置四元数Z分量
        self.body2cloud_trans.transform.rotation.w = quat[3]  # 设置四元数W分量
        
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
        
        # 实现了点云的空间过滤，定义了一个过滤框，在这个范围内的点会被过滤掉。
        self.x_filter_min = -0.55  # 设置X轴过滤最小值
        self.x_filter_max = 0.2  # 设置X轴过滤最大值
        self.y_filter_min = -0.15  # 设置Y轴过滤最小值
        self.y_filter_max = 0.15  # 设置Y轴过滤最大值
        self.z_filter_min = -0.5 
        self.z_filter_max = 0 

        rclpy.spin(self)  # 运行节点
                
    def is_in_filter_box(self, point):  # 检查点是否在过滤框内
        # 检查点是否在过滤框内
        is_in_box = point[0] > self.x_filter_min and \
                    point[0] < self.x_filter_max and \
                    point[1] > self.y_filter_min and \
                    point[1] < self.y_filter_max and \
                    point[2] > self.z_filter_min and \
                    point[2] < self.z_filter_max  # 检查所有维度是否在范围内
        return is_in_box  # 返回检查结果

    def cloud_callback(self, data):  # 点云回调函数

        # 第一次接收到点云数据时计算时间偏移
        # 计算当前ROS时间和传感器时间戳之间的差值：
        # offset = 当前ROS时间 - 传感器时间戳
        # (这个偏移量只计算一次，后续所有数据都使用这个偏移量)
        if not self.hesai_time_stamp_offset_set:  # 如果时间戳偏移未设置
            self.hesai_time_stamp_offset = self.get_clock().now().nanoseconds - Time.from_msg(data.header.stamp).nanoseconds  # 计算时间戳偏移
            self.hesai_time_stamp_offset_set = True  # 标记时间戳偏移已设置
                
        # # 点云做切割处理        
        # cloud_arr = pc2.read_points_list(data)  # 读取点云数据
        # points = np.array(cloud_arr)  # 转换为numpy数组

        # transform = self.body2cloud_trans.transform  # 获取变换信息
        # mat = quat2mat(np.array([transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z]))  # 四元数转旋转矩阵
        # translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])  # 获取平移向量
        
        # transformed_points = points  # 初始化变换后的点
        # transformed_points[:, 0:3] = points[:, 0:3] @ mat.T + translation  # 应用旋转和平移
        # transformed_points[:, 2] += self.cam_offset  # 应用相机偏移
        # i = 0  # 初始化计数器
        # remove_list = []  # 初始化移除列表
        # transformed_points = transformed_points.tolist()  # 转换为列表

        # for i in range(len(transformed_points)):  # 遍历所有点
        #     transformed_points[i][4] = int(transformed_points[i][4])  # 转换点云线数值为整数
        #     if self.is_in_filter_box(transformed_points[i]):  # 检查点是否在过滤框内
        #         remove_list.append(i)  # 添加到移除列表

        # remove_list.sort(reverse=True)  # 反向排序移除列表

        # for id_to_remove in remove_list:  # 遍历移除列表
        #     del transformed_points[id_to_remove]  # 移除点
        
        # # 应用时间偏移到转换后的点云消息
        # # 新时间戳 = 原始时间戳 + 时间偏移量
        # elevated_cloud = pc2.create_cloud(data.header, data.fields, transformed_points)  # 创建新的点云消息
        # elevated_cloud.header.stamp = Time(nanoseconds=Time.from_msg(elevated_cloud.header.stamp).nanoseconds + self.hesai_time_stamp_offset).to_msg()  # 更新时间戳
        # elevated_cloud.header.frame_id = "body"  # 设置坐标系
        # elevated_cloud.is_dense = data.is_dense  # 设置密度标志

        # self.cloud_pub.publish(elevated_cloud)  # 发布转换后的点云



        # 仅做时间同步，不做其他处理
        raw_cloud = data  # 创建点云消息
        raw_cloud.header.stamp = Time(nanoseconds=Time.from_msg(raw_cloud.header.stamp).nanoseconds + self.hesai_time_stamp_offset).to_msg()  # 更新时间戳
        raw_cloud.header.frame_id = "body"  # 设置坐标系
        raw_cloud.is_dense = data.is_dense  # 设置密度标志

        self.cloud_pub.publish(raw_cloud)  # 发布转换后的点云
            

    def imu_callback(self, data):  # IMU回调函数   
        start_time = self.get_clock().now()
        

        # 把宇树时间戳的TimeSpec格式转化成ros2的Time格式
        ros_time = Time(
            seconds=data.stamp.sec,
            nanoseconds=data.stamp.nanosec
        )

        if not self.go2imu_time_stamp_offset_set:  # 如果时间戳偏移未设置
            self.go2imu_time_stamp_offset = self.get_clock().now().nanoseconds - ros_time.nanoseconds  # 计算时间戳偏移
            self.go2imu_time_stamp_offset_set = True  # 标记时间戳偏移已设置


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
        transformed_imu.header.stamp = ros_time.to_msg()  # 设置时间戳
        transformed_imu.header.frame_id = 'body'  # 设置坐标系
        transformed_imu.orientation.x = transformed_orientation[0]  # 设置姿态X分量
        transformed_imu.orientation.y = transformed_orientation[1]  # 设置姿态Y分量
        transformed_imu.orientation.z = transformed_orientation[2]  # 设置姿态Z分量
        transformed_imu.orientation.w = transformed_orientation[3]  # 设置姿态W分量
        transformed_imu.angular_velocity = transformed_angular_velocity  # 设置角速度
        transformed_imu.linear_acceleration = transformed_linear_acceleration  # 设置线加速度
        
        # 应用时间偏移到转换后的IMU消息
        # 新时间戳 = 原始时间戳 + 时间偏移量
        transformed_imu.header.stamp = Time(nanoseconds=Time.from_msg(transformed_imu.header.stamp).nanoseconds + self.go2imu_time_stamp_offset).to_msg()  # 更新时间戳
        
        self.imu_raw_pub.publish(transformed_imu)  # 发布原始IMU数据
        
        transformed_imu.orientation.x = 0.0  # 重置姿态X分量
        transformed_imu.orientation.y = 0.0  # 重置姿态Y分量
        transformed_imu.orientation.z = 0.0  # 重置姿态Z分量
        transformed_imu.orientation.w = 1.0  # 重置姿态W分量
        
        transformed_imu.linear_acceleration.x = 0.0  # 重置加速度X分量
        transformed_imu.linear_acceleration.y = 0.0  # 重置加速度Y分量
        transformed_imu.linear_acceleration.z = 0.0  # 重置加速度Z分量
        
        self.imu_pub.publish(transformed_imu)  # 发布转换后的IMU数据

        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e9
        # print(f"处理时间: {processing_time} 秒")

def main(args=None):  # 主函数
    rclpy.init(args=args)  # 初始化ROS2

    transform_node = Repuber()  # 创建节点实例

    rclpy.spin(transform_node)  # 运行节点

    Repuber.destroy_node()  # 销毁节点
    rclpy.shutdown()  # 关闭ROS2

if __name__ == '__main__':  # 主程序入口
    main()  # 运行主函数
