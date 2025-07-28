// 包含所需的头文件
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <queue>

// ROS2相关头文件
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "builtin_interfaces/msg/time.hpp"

// ROS2消息类型头文件
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

// TF2相关头文件
#include "tf2/transform_datatypes.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// PCL点云处理相关头文件
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 消息过滤器相关头文件
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "rmw/types.h"
#include "rmw/qos_profiles.h"

using namespace std;

// 定义常量
const double PI = 3.1415926;

// 点云处理参数
double scanVoxelSize = 0.1;        // 体素大小
double decayTime = 10.0;           // 点云衰减时间
double noDecayDis = 0;             // 不衰减的距离阈值
double clearingDis = 30.0;         // 清除点云的距离阈值
bool clearingCloud = false;        // 是否清除点云标志
bool useSorting = false;           // 是否使用排序
double quantileZ = 0.25;           // Z轴分位数
double vehicleHeight = 1.5;        // 车辆高度
int voxelPointUpdateThre = 100;    // 体素点更新阈值
double voxelTimeUpdateThre = 2.0;  // 体素时间更新阈值
double lowerBoundZ = -1.5;         // Z轴下界
double upperBoundZ = 1.0;          // Z轴上界
double disRatioZ = 0.1;           // 距离比例系数
bool checkTerrainConn = true;      // 是否检查地形连通性
double terrainUnderVehicle = -0.75;// 车辆下方地形高度
double terrainConnThre = 0.5;      // 地形连通性阈值
double ceilingFilteringThre = 2.0; // 天花板过滤阈值
double localTerrainMapRadius = 4.0;// 局部地形图半径，这个距离内的点，使用局部地图terrainmap的点，并与远距离地形点云缝合

// 地形体素参数
float terrainVoxelSize = 2.0;      // 地形体素大小
int terrainVoxelShiftX = 0;        // X方向偏移
int terrainVoxelShiftY = 0;        // Y方向偏移
const int terrainVoxelWidth = 41;  // 地形体素宽度
int terrainVoxelHalfWidth = (terrainVoxelWidth - 1) / 2;  // 远距离，扩展地形体素的半径，近处不扩展
const int terrainVoxelNum = terrainVoxelWidth * terrainVoxelWidth;  // 地形体素总数

// 平面体素参数
float planarVoxelSize = 0.4;       // 平面体素大小，0.4米 x 0.4米，用于所有距离的地形分析，不仅仅是远距离
const int planarVoxelWidth = 101;  // 平面体素网格宽度，101 x 101的网格
int planarVoxelHalfWidth = (planarVoxelWidth - 1) / 2;  // 半宽
const int planarVoxelNum = planarVoxelWidth * planarVoxelWidth;  // 总网格数

// 点云数据结构
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());  // 原始激光点云
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());  // 裁剪后的点云
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());   // 降采样后的点云
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloud(new pcl::PointCloud<pcl::PointXYZI>());    // 地形点云
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudElev(new pcl::PointCloud<pcl::PointXYZI>());// 带高程的地形点云
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudLocal(new pcl::PointCloud<pcl::PointXYZI>());// 局部地形点云
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloud[terrainVoxelNum];  // 地形体素点云数组

// 地形体素状态数组
int terrainVoxelUpdateNum[terrainVoxelNum] = { 0 };      // 更新计数
float terrainVoxelUpdateTime[terrainVoxelNum] = { 0 };   // 更新时间
float planarVoxelElev[planarVoxelNum] = { 0 };          // 平面体素高程
int planarVoxelConn[planarVoxelNum] = { 0 };            // 平面体素连通性
vector<float> planarPointElev[planarVoxelNum];          // 平面点高程
queue<int> planarVoxelQueue;                            // 平面体素队列

// 时间和状态变量
double laserCloudTime = 0;         // 激光点云时间戳
bool newlaserCloud = false;        // 新点云标志
double systemInitTime = 0;         // 系统初始化时间
bool systemInited = false;         // 系统初始化标志

// 车辆姿态和位置
float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;  // 横滚、俯仰、偏航角
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;           // 位置坐标

// PCL滤波器
pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;  // 降采样滤波器
pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;        // K-D树

// 状态估计回调函数 - 处理车辆位姿信息
void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odom->pose.pose.orientation;
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  vehicleX = odom->pose.pose.position.x;
  vehicleY = odom->pose.pose.position.y;
  vehicleZ = odom->pose.pose.position.z;
}

// 激光点云回调函数 - 处理注册后的激光扫描数据
void laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloud2)
{
  laserCloudTime = rclcpp::Time(laserCloud2->header.stamp).seconds();

  // 系统初始化
  if (!systemInited)
  {
    systemInitTime = laserCloudTime;
    systemInited = true;
  }

  // 转换点云格式并进行初步过滤
  laserCloud->clear();
  pcl::fromROSMsg(*laserCloud2, *laserCloud);

  pcl::PointXYZI point;
  laserCloudCrop->clear();
  int laserCloudSize = laserCloud->points.size();
  for (int i = 0; i < laserCloudSize; i++)
  {
    point = laserCloud->points[i];

    float pointX = point.x;
    float pointY = point.y;
    float pointZ = point.z;

    // 根据距离和高度过滤点云
    float dis = sqrt((pointX - vehicleX) * (pointX - vehicleX) + (pointY - vehicleY) * (pointY - vehicleY));
    if (pointZ - vehicleZ > lowerBoundZ - disRatioZ * dis && pointZ - vehicleZ < upperBoundZ + disRatioZ * dis &&
        dis < terrainVoxelSize * (terrainVoxelHalfWidth + 1))
    {
      point.x = pointX;
      point.y = pointY;
      point.z = pointZ;
      point.intensity = laserCloudTime - systemInitTime;
      laserCloudCrop->push_back(point);
    }
  }

  newlaserCloud = true;
}

// 局部地形点云回调函数
void terrainCloudLocalHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrainCloudLocal2)
{
  terrainCloudLocal->clear();
  pcl::fromROSMsg(*terrainCloudLocal2, *terrainCloudLocal);
}

// 手柄回调函数 - 处理手动清除点云的指令
void joystickHandler(const sensor_msgs::msg::Joy::ConstSharedPtr joy)
{
  if (joy->buttons[5] > 0.5)
  {
    clearingCloud = true;
  }
}

// 点云清除回调函数
void clearingHandler(const std_msgs::msg::Float32::ConstSharedPtr dis)
{
  clearingDis = dis->data;
  clearingCloud = true;
}

// 主函数
int main(int argc, char** argv)
{
  // 初始化ROS2节点
  rclcpp::init(argc, argv);
  auto nh = rclcpp::Node::make_shared("terrainAnalysisExt");

  // 声明和获取参数
  nh->declare_parameter<double>("scanVoxelSize", scanVoxelSize);
  nh->declare_parameter<double>("decayTime", decayTime);
  nh->declare_parameter<double>("noDecayDis", noDecayDis);
  nh->declare_parameter<double>("clearingDis", clearingDis);
  nh->declare_parameter<bool>("useSorting", useSorting);
  nh->declare_parameter<double>("quantileZ", quantileZ);
  nh->declare_parameter<double>("vehicleHeight", vehicleHeight);
  nh->declare_parameter<int>("voxelPointUpdateThre", voxelPointUpdateThre);
  nh->declare_parameter<double>("voxelTimeUpdateThre", voxelTimeUpdateThre);
  nh->declare_parameter<double>("lowerBoundZ", lowerBoundZ);
  nh->declare_parameter<double>("upperBoundZ", upperBoundZ);
  nh->declare_parameter<double>("disRatioZ", disRatioZ);
  nh->declare_parameter<bool>("checkTerrainConn", checkTerrainConn);
  nh->declare_parameter<double>("terrainUnderVehicle", terrainUnderVehicle);
  nh->declare_parameter<double>("terrainConnThre", terrainConnThre);
  nh->declare_parameter<double>("ceilingFilteringThre", ceilingFilteringThre);
  nh->declare_parameter<double>("localTerrainMapRadius", localTerrainMapRadius);
  nh->declare_parameter<float>("planarVoxelSize", planarVoxelSize);     // 默认值0.4


  nh->get_parameter("scanVoxelSize", scanVoxelSize);
  nh->get_parameter("decayTime", decayTime);
  nh->get_parameter("noDecayDis", noDecayDis);
  nh->get_parameter("clearingDis", clearingDis);
  nh->get_parameter("useSorting", useSorting);
  nh->get_parameter("quantileZ", quantileZ);
  nh->get_parameter("vehicleHeight", vehicleHeight);
  nh->get_parameter("voxelPointUpdateThre", voxelPointUpdateThre);
  nh->get_parameter("voxelTimeUpdateThre", voxelTimeUpdateThre);
  nh->get_parameter("lowerBoundZ", lowerBoundZ);
  nh->get_parameter("upperBoundZ", upperBoundZ);
  nh->get_parameter("disRatioZ", disRatioZ);
  nh->get_parameter("checkTerrainConn", checkTerrainConn);
  nh->get_parameter("terrainUnderVehicle", terrainUnderVehicle);
  nh->get_parameter("terrainConnThre", terrainConnThre);
  nh->get_parameter("ceilingFilteringThre", ceilingFilteringThre);
  nh->get_parameter("localTerrainMapRadius", localTerrainMapRadius);
  nh->get_parameter("planarVoxelSize", planarVoxelSize);


  // 创建订阅者
  auto subOdometry = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odometryHandler);
  auto subLaserCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan", 5, laserCloudHandler);
  auto subJoystick = nh->create_subscription<sensor_msgs::msg::Joy>("/joy", 5, joystickHandler);
  auto subClearing = nh->create_subscription<std_msgs::msg::Float32>("/cloud_clearing", 5, clearingHandler);
  auto subTerrainCloudLocal = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/terrain_map", 2, terrainCloudLocalHandler);

  // 创建发布者
  auto pubTerrainCloud = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/terrain_map_ext", 2);

  // 初始化地形体素点云数组
  for (int i = 0; i < terrainVoxelNum; i++)
  {
    terrainVoxelCloud[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }

  // 设置降采样滤波器参数
  downSizeFilter.setLeafSize(scanVoxelSize, scanVoxelSize, scanVoxelSize);

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;

  // 主循环
  rclcpp::Rate rate(100);
  bool status = rclcpp::ok();
  while (status)
  {
    rclcpp::spin_some(nh);

    if (newlaserCloud)
    {
      newlaserCloud = false;

      // 地形体素滚动更新
      float terrainVoxelCenX = terrainVoxelSize * terrainVoxelShiftX;
      float terrainVoxelCenY = terrainVoxelSize * terrainVoxelShiftY;

      // 处理X方向的体素滚动
      while (vehicleX - terrainVoxelCenX < -terrainVoxelSize)
      {
        for (int indY = 0; indY < terrainVoxelWidth; indY++)
        {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[terrainVoxelWidth * (terrainVoxelWidth - 1) + indY];
          for (int indX = terrainVoxelWidth - 1; indX >= 1; indX--)
          {
            terrainVoxelCloud[terrainVoxelWidth * indX + indY] =
                terrainVoxelCloud[terrainVoxelWidth * (indX - 1) + indY];
          }
          terrainVoxelCloud[indY] = terrainVoxelCloudPtr;
          terrainVoxelCloud[indY]->clear();
        }
        terrainVoxelShiftX--;
        terrainVoxelCenX = terrainVoxelSize * terrainVoxelShiftX;
      }

      while (vehicleX - terrainVoxelCenX > terrainVoxelSize)
      {
        for (int indY = 0; indY < terrainVoxelWidth; indY++)
        {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr = terrainVoxelCloud[indY];
          for (int indX = 0; indX < terrainVoxelWidth - 1; indX++)
          {
            terrainVoxelCloud[terrainVoxelWidth * indX + indY] =
                terrainVoxelCloud[terrainVoxelWidth * (indX + 1) + indY];
          }
          terrainVoxelCloud[terrainVoxelWidth * (terrainVoxelWidth - 1) + indY] = terrainVoxelCloudPtr;
          terrainVoxelCloud[terrainVoxelWidth * (terrainVoxelWidth - 1) + indY]->clear();
        }
        terrainVoxelShiftX++;
        terrainVoxelCenX = terrainVoxelSize * terrainVoxelShiftX;
      }

      // 处理Y方向的体素滚动
      while (vehicleY - terrainVoxelCenY < -terrainVoxelSize)
      {
        for (int indX = 0; indX < terrainVoxelWidth; indX++)
        {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[terrainVoxelWidth * indX + (terrainVoxelWidth - 1)];
          for (int indY = terrainVoxelWidth - 1; indY >= 1; indY--)
          {
            terrainVoxelCloud[terrainVoxelWidth * indX + indY] =
                terrainVoxelCloud[terrainVoxelWidth * indX + (indY - 1)];
          }
          terrainVoxelCloud[terrainVoxelWidth * indX] = terrainVoxelCloudPtr;
          terrainVoxelCloud[terrainVoxelWidth * indX]->clear();
        }
        terrainVoxelShiftY--;
        terrainVoxelCenY = terrainVoxelSize * terrainVoxelShiftY;
      }

      while (vehicleY - terrainVoxelCenY > terrainVoxelSize)
      {
        for (int indX = 0; indX < terrainVoxelWidth; indX++)
        {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr = terrainVoxelCloud[terrainVoxelWidth * indX];
          for (int indY = 0; indY < terrainVoxelWidth - 1; indY++)
          {
            terrainVoxelCloud[terrainVoxelWidth * indX + indY] =
                terrainVoxelCloud[terrainVoxelWidth * indX + (indY + 1)];
          }
          terrainVoxelCloud[terrainVoxelWidth * indX + (terrainVoxelWidth - 1)] = terrainVoxelCloudPtr;
          terrainVoxelCloud[terrainVoxelWidth * indX + (terrainVoxelWidth - 1)]->clear();
        }
        terrainVoxelShiftY++;
        terrainVoxelCenY = terrainVoxelSize * terrainVoxelShiftY;
      }

      // 将注册的激光扫描数据堆叠到地形体素中
      pcl::PointXYZI point;
      int laserCloudCropSize = laserCloudCrop->points.size();
      for (int i = 0; i < laserCloudCropSize; i++)
      {
        point = laserCloudCrop->points[i];

        // 计算点云在体素网格中的索引
        int indX = int((point.x - vehicleX + terrainVoxelSize / 2) / terrainVoxelSize) + terrainVoxelHalfWidth;
        int indY = int((point.y - vehicleY + terrainVoxelSize / 2) / terrainVoxelSize) + terrainVoxelHalfWidth;

        if (point.x - vehicleX + terrainVoxelSize / 2 < 0)
          indX--;
        if (point.y - vehicleY + terrainVoxelSize / 2 < 0)
          indY--;

        // 将点云添加到对应的体素中
        if (indX >= 0 && indX < terrainVoxelWidth && indY >= 0 && indY < terrainVoxelWidth)
        {
          terrainVoxelCloud[terrainVoxelWidth * indX + indY]->push_back(point);
          terrainVoxelUpdateNum[terrainVoxelWidth * indX + indY]++;
        }
      }

      // 更新地形体素
      for (int ind = 0; ind < terrainVoxelNum; ind++)
      {
        if (terrainVoxelUpdateNum[ind] >= voxelPointUpdateThre ||
            laserCloudTime - systemInitTime - terrainVoxelUpdateTime[ind] >= voxelTimeUpdateThre || clearingCloud)
        {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr = terrainVoxelCloud[ind];

          // 对体素中的点云进行降采样
          laserCloudDwz->clear();
          downSizeFilter.setInputCloud(terrainVoxelCloudPtr);
          downSizeFilter.filter(*laserCloudDwz);

          // 根据时间和距离过滤点云
          terrainVoxelCloudPtr->clear();
          int laserCloudDwzSize = laserCloudDwz->points.size();
          for (int i = 0; i < laserCloudDwzSize; i++)
          {
            point = laserCloudDwz->points[i];
            float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) + (point.y - vehicleY) * (point.y - vehicleY));
            if (point.z - vehicleZ > lowerBoundZ - disRatioZ * dis &&
                point.z - vehicleZ < upperBoundZ + disRatioZ * dis &&
                (laserCloudTime - systemInitTime - point.intensity < decayTime || dis < noDecayDis) &&
                !(dis < clearingDis && clearingCloud))
            {
              terrainVoxelCloudPtr->push_back(point);
            }
          }

          terrainVoxelUpdateNum[ind] = 0;
          terrainVoxelUpdateTime[ind] = laserCloudTime - systemInitTime;
        }
      }

      // 合并地形点云
      terrainCloud->clear();
      for (int indX = terrainVoxelHalfWidth - 10; indX <= terrainVoxelHalfWidth + 10; indX++)
      {
        for (int indY = terrainVoxelHalfWidth - 10; indY <= terrainVoxelHalfWidth + 10; indY++)
        {
          *terrainCloud += *terrainVoxelCloud[terrainVoxelWidth * indX + indY];
        }
      }

      // 估计地面并计算每个点的高程
      for (int i = 0; i < planarVoxelNum; i++)
      {
        planarVoxelElev[i] = 0;
        planarVoxelConn[i] = 0;
        planarPointElev[i].clear();
      }

      // 计算每个点的高程
      int terrainCloudSize = terrainCloud->points.size();
      for (int i = 0; i < terrainCloudSize; i++)
      {
        point = terrainCloud->points[i];
        float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) + (point.y - vehicleY) * (point.y - vehicleY));
        if (point.z - vehicleZ > lowerBoundZ - disRatioZ * dis && point.z - vehicleZ < upperBoundZ + disRatioZ * dis)
        {
          int indX = int((point.x - vehicleX + planarVoxelSize / 2) / planarVoxelSize) + planarVoxelHalfWidth;
          int indY = int((point.y - vehicleY + planarVoxelSize / 2) / planarVoxelSize) + planarVoxelHalfWidth;

          if (point.x - vehicleX + planarVoxelSize / 2 < 0)
            indX--;
          if (point.y - vehicleY + planarVoxelSize / 2 < 0)
            indY--;

          // 将点的高程添加到相邻体素中
          for (int dX = -1; dX <= 1; dX++)
          {
            for (int dY = -1; dY <= 1; dY++)
            {
              if (indX + dX >= 0 && indX + dX < planarVoxelWidth && indY + dY >= 0 && indY + dY < planarVoxelWidth)
              {
                planarPointElev[planarVoxelWidth * (indX + dX) + indY + dY].push_back(point.z);
              }
            }
          }
        }
      }

      // 使用排序或最小值方法计算体素高程
      if (useSorting)
      {
        for (int i = 0; i < planarVoxelNum; i++)
        {
          int planarPointElevSize = planarPointElev[i].size();
          if (planarPointElevSize > 0)
          {
            sort(planarPointElev[i].begin(), planarPointElev[i].end());

            int quantileID = int(quantileZ * planarPointElevSize);
            if (quantileID < 0)
              quantileID = 0;
            else if (quantileID >= planarPointElevSize)
              quantileID = planarPointElevSize - 1;

            planarVoxelElev[i] = planarPointElev[i][quantileID];
          }
        }
      }
      else
      {
        for (int i = 0; i < planarVoxelNum; i++)
        {
          int planarPointElevSize = planarPointElev[i].size();
          if (planarPointElevSize > 0)
          {
            float minZ = 1000.0;
            int minID = -1;
            for (int j = 0; j < planarPointElevSize; j++)
            {
              if (planarPointElev[i][j] < minZ)
              {
                minZ = planarPointElev[i][j];
                minID = j;
              }
            }

            if (minID != -1)
            {
              planarVoxelElev[i] = planarPointElev[i][minID];
            }
          }
        }
      }
  
      // 检查地形连通性以去除天花板
      if (checkTerrainConn)
      {
        int ind = planarVoxelWidth * planarVoxelHalfWidth + planarVoxelHalfWidth;
        if (planarPointElev[ind].size() == 0)
          planarVoxelElev[ind] = vehicleZ + terrainUnderVehicle;

        // 使用广度优先搜索检查连通性
        planarVoxelQueue.push(ind);
        planarVoxelConn[ind] = 1;
        while (!planarVoxelQueue.empty())
        {
          int front = planarVoxelQueue.front();
          planarVoxelConn[front] = 2;
          planarVoxelQueue.pop();

          int indX = int(front / planarVoxelWidth);
          int indY = front % planarVoxelWidth;
          for (int dX = -10; dX <= 10; dX++)
          {
            for (int dY = -10; dY <= 10; dY++)
            {
              if (indX + dX >= 0 && indX + dX < planarVoxelWidth && indY + dY >= 0 && indY + dY < planarVoxelWidth)
              {
                ind = planarVoxelWidth * (indX + dX) + indY + dY;
                if (planarVoxelConn[ind] == 0 && planarPointElev[ind].size() > 0)
                {
                  if (fabs(planarVoxelElev[front] - planarVoxelElev[ind]) < terrainConnThre)
                  {
                    planarVoxelQueue.push(ind);
                    planarVoxelConn[ind] = 1;
                  } else if (fabs(planarVoxelElev[front] - planarVoxelElev[ind]) > ceilingFilteringThre)
                  {
                    planarVoxelConn[ind] = -1;
                  }
                }
              }
            }
          }
        } 
      }

      // compute terrain map beyond localTerrainMapRadius
      terrainCloudElev->clear();
      int terrainCloudElevSize = 0;
      for (int i = 0; i < terrainCloudSize; i++)
      {
        point = terrainCloud->points[i];
        float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) + (point.y - vehicleY) * (point.y - vehicleY));
        if (point.z - vehicleZ > lowerBoundZ - disRatioZ * dis && point.z - vehicleZ < upperBoundZ + disRatioZ * dis && dis > localTerrainMapRadius)
        {
          int indX = int((point.x - vehicleX + planarVoxelSize / 2) / planarVoxelSize) + planarVoxelHalfWidth;
          int indY = int((point.y - vehicleY + planarVoxelSize / 2) / planarVoxelSize) + planarVoxelHalfWidth;

          if (point.x - vehicleX + planarVoxelSize / 2 < 0)
            indX--;
          if (point.y - vehicleY + planarVoxelSize / 2 < 0)
            indY--;

          if (indX >= 0 && indX < planarVoxelWidth && indY >= 0 && indY < planarVoxelWidth)
          {
            int ind = planarVoxelWidth * indX + indY;
            float disZ = fabs(point.z - planarVoxelElev[ind]);
            if (disZ < vehicleHeight && (planarVoxelConn[ind] == 2 || !checkTerrainConn))
            {
              terrainCloudElev->push_back(point);
              terrainCloudElev->points[terrainCloudElevSize].x = point.x;
              terrainCloudElev->points[terrainCloudElevSize].y = point.y;
              terrainCloudElev->points[terrainCloudElevSize].z = point.z;
              terrainCloudElev->points[terrainCloudElevSize].intensity = disZ;

              terrainCloudElevSize++;
            }
          }
        }
      }

      // merge in local terrain map within localTerrainMapRadius
      int terrainCloudLocalSize = terrainCloudLocal->points.size();
      for (int i = 0; i < terrainCloudLocalSize; i++) {
        point = terrainCloudLocal->points[i];
        float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) + (point.y - vehicleY) * (point.y - vehicleY));
        if (dis <= localTerrainMapRadius)
        {
          terrainCloudElev->push_back(point);
        }
      }

      clearingCloud = false;

      // publish points with elevation
      sensor_msgs::msg::PointCloud2 terrainCloud2;
      pcl::toROSMsg(*terrainCloudElev, terrainCloud2);
      terrainCloud2.header.stamp = rclcpp::Time(static_cast<uint64_t>(laserCloudTime * 1e9));
      terrainCloud2.header.frame_id = "map";
      pubTerrainCloud->publish(terrainCloud2);
    }

    status = rclcpp::ok();
    rate.sleep();
  }
  
  return 0;
}
