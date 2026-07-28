#!/usr/bin/env python
import rclpy  # 导入ROS2 Python客户端库
from rclpy.node import Node  # 导入ROS2节点类
from rclpy.time import Time  # 导入ROS2时间类
from sensor_msgs.msg import Imu  # 导入IMU消息类型
from geometry_msgs.msg import Vector3  # 导入向量消息类型

# 修复 NumPy 兼容性问题：在导入前添加 np.float 别名
import numpy as np  # 导入numpy库
if not hasattr(np, 'float'):
    np.float = np.float64
    np.int = np.int_
    np.complex = np.complex_
    np.bool = np.bool_

from unitree_go.msg import SportModeState
from livox_ros_driver2.msg import CustomMsg, CustomPoint  # Livox CustomMsg（xfer_format=1）

import yaml  # 导入YAML解析库
import os  # 导入操作系统接口


class Repuber(Node):  # 定义传感器转换节点类
    def __init__(self):  # 初始化方法
        super().__init__('transform_mid360_onboard')  # 调用父类初始化方法

        # 与 msg_MID360_launch.py 当前 remapping 对齐；若去掉 remap 可改成 /livox/lidar
        self.cloud_sub = self.create_subscription(CustomMsg, '/livox/lidar', self.cloud_callback, 50)  # 创建点云订阅者
        self.imu_sub = self.create_subscription(SportModeState, '/sportmodestate', self.imu_callback, 50)  # 创建IMU订阅者

        self.imu_raw_pub = self.create_publisher(Imu, '/mid360/transformed_raw_imu', 50)  # 创建原始IMU发布者
        self.imu_pub = self.create_publisher(Imu, '/mid360/transformed_imu', 50)  # 创建转换后IMU发布者
        self.cloud_pub = self.create_publisher(CustomMsg, '/mid360/transformed_cloud', 50)  # 创建转换后点云发布者（CustomMsg）

        self.mid360_time_stamp_offset = 0  # 初始化mid360雷达时间戳偏移
        self.mid360_time_stamp_offset_set = False  # 初始化mid360雷达时间戳偏移设置标志
        self.go2imu_time_stamp_offset = 0  # 初始化go2自带IMU时间戳偏移
        self.go2imu_time_stamp_offset_set = False  # 初始化go2自带IMU时间戳偏移设置标志

        # 加载标定数据
        calib_data = {  # 设置默认标定数据
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

        # 方案A：前倾角写入 Point-LIO extrinsic_R，此处不对点云做外参旋转
        # 实现了点云的空间过滤，定义了一个过滤框，在这个范围内的点会被过滤掉。
        # 过滤框在雷达坐标系下，需按安装再调
        self.x_filter_min = -0.55  # 设置X轴过滤最小值
        self.x_filter_max = 0.2  # 设置X轴过滤最大值
        self.y_filter_min = -0.15  # 设置Y轴过滤最小值
        self.y_filter_max = 0.15  # 设置Y轴过滤最大值
        self.z_filter_min = -0.5
        self.z_filter_max = 0

    def is_in_filter_box(self, point):  # 检查点是否在过滤框内
        # 检查点是否在过滤框内（point 为 [x, y, z, ...]）
        is_in_box = point[0] > self.x_filter_min and \
                    point[0] < self.x_filter_max and \
                    point[1] > self.y_filter_min and \
                    point[1] < self.y_filter_max and \
                    point[2] > self.z_filter_min and \
                    point[2] < self.z_filter_max  # 检查所有维度是否在范围内
        return is_in_box  # 返回检查结果

    def cloud_callback(self, data):  # 点云回调函数
        # 第一次接收到点云数据时计算时间偏移
        # offset = 当前ROS时间 - 传感器时间戳
        if not self.mid360_time_stamp_offset_set:  # 如果时间戳偏移未设置
            self.mid360_time_stamp_offset = self.get_clock().now().nanoseconds - Time.from_msg(data.header.stamp).nanoseconds  # 计算时间戳偏移
            self.mid360_time_stamp_offset_set = True  # 标记时间戳偏移已设置

        # 方案A：不做外参旋转，仅时间同步 + 近处过滤，保留逐点时间等字段
        out = CustomMsg()  # 创建输出 CustomMsg
        out.header = data.header  # 复制消息头
        out.header.stamp = Time(nanoseconds=Time.from_msg(data.header.stamp).nanoseconds + self.mid360_time_stamp_offset).to_msg()  # 更新时间戳
        out.header.frame_id = data.header.frame_id if data.header.frame_id else "livox_frame"  # 保持雷达坐标系
        out.timebase = data.timebase  # 保留时间基
        out.lidar_id = data.lidar_id  # 保留雷达ID
        out.rsvd = data.rsvd  # 保留保留字段

        kept = []  # 过滤后保留的点
        for pt in data.points:  # 遍历所有点
            if self.is_in_filter_box([pt.x, pt.y, pt.z]):  # 检查点是否在过滤框内
                continue  # 在框内则丢弃

            new_pt = CustomPoint()  # 创建新点
            new_pt.x = pt.x  # X坐标（雷达系，不做旋转）
            new_pt.y = pt.y  # Y坐标
            new_pt.z = pt.z  # Z坐标
            new_pt.reflectivity = pt.reflectivity  # 反射率
            new_pt.tag = pt.tag  # 标签
            new_pt.line = pt.line  # 线号
            new_pt.offset_time = pt.offset_time  # 逐点时间，Point-LIO 必需
            kept.append(new_pt)  # 加入保留列表

        out.points = kept  # 设置点列表
        out.point_num = len(kept)  # 设置点数
        self.cloud_pub.publish(out)  # 发布转换后的点云

    def imu_callback(self, data):  # IMU回调函数
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


def main(args=None):  # 主函数
    rclpy.init(args=args)  # 初始化ROS2

    transform_node = Repuber()  # 创建节点实例

    rclpy.spin(transform_node)  # 运行节点

    transform_node.destroy_node()  # 销毁节点
    rclpy.shutdown()  # 关闭ROS2


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主函数
