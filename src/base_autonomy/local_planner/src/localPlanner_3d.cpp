#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/clock.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/path.hpp>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <sensor_msgs/msg/imu.h>

#include "tf2/transform_datatypes.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "rmw/types.h"
#include "rmw/qos_profiles.h"

using namespace std;

const double PI = 3.1415926;

#define PLOTPATHSET 1

string pathFolder;
double vehicleLength = 0.6;
double vehicleWidth = 0.6;
double sensorOffsetX = 0;
double sensorOffsetY = 0;
bool twoWayDrive = true;
double laserVoxelSize = 0.05;
double terrainVoxelSize = 0.2;
bool useTerrainAnalysis = false;
bool checkObstacle = true;
bool checkRotObstacle = false;
double adjacentRange = 3.5;
double obstacleHeightThre = 0.2;
double groundHeightThre = 0.1;
double costHeightThre = 0.1;
double costScore = 0.02;
bool useCost = false;
const int laserCloudStackNum = 1;
int laserCloudCount = 0;
int pointPerPathThre = 2;
double minRelZ = -0.5;
double maxRelZ = 0.25;
double maxSpeed = 1.0;
double dirWeight = 0.02;
double dirThre = 90.0;
bool dirToVehicle = false;
double pathScale = 1.0;
double minPathScale = 0.75;
double pathScaleStep = 0.25;
bool pathScaleBySpeed = true;
double minPathRange = 1.0;
double pathRangeStep = 0.5;
bool pathRangeBySpeed = true;
bool pathCropByGoal = true;
bool use3DMode = false;  // 是否启用完整3D模式（包含3D距离计算和完整坐标变换，适用于爬楼梯等复杂地形）
bool autonomyMode = false;
double autonomySpeed = 1.0;
double joyToSpeedDelay = 2.0;
double joyToCheckObstacleDelay = 5.0;
double goalCloseDis_XY = 1.0;
double goalClearRange = 0.5;
double goalCloseDis_Z = 0.05;  // Z轴接近距离阈值，用于判断垂直方向是否接近目标
double goalX = 0;
double goalY = 0;
double goalZ = 0;   //增加z轴目标点参数

float joySpeed = 0;
float joySpeedRaw = 0;
float joyDir = 0;

const int pathNum = 343;
const int groupNum = 7;
float gridVoxelSize = 0.02;
float searchRadius = 0.55;
float gridVoxelOffsetX = 3.2;
float gridVoxelOffsetY = 4.5;
const int gridVoxelNumX = 161;
const int gridVoxelNumY = 451;
const int gridVoxelNum = gridVoxelNumX * gridVoxelNumY;

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudStack[laserCloudStackNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr plannerCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr plannerCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr addedObstacles(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZ>::Ptr startPaths[groupNum];
#if PLOTPATHSET == 1
pcl::PointCloud<pcl::PointXYZI>::Ptr paths[pathNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr freePaths(new pcl::PointCloud<pcl::PointXYZI>());
#endif

int pathList[pathNum] = {0};
float endDirPathList[pathNum] = {0};
int clearPathList[36 * pathNum] = {0};
float pathPenaltyList[36 * pathNum] = {0};
float clearPathPerGroupScore[36 * groupNum] = {0};
std::vector<int> correspondences[gridVoxelNum];

bool newLaserCloud = false;
bool newTerrainCloud = false;

double odomTime = 0;
double joyTime = 0;

float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;

pcl::VoxelGrid<pcl::PointXYZI> laserDwzFilter, terrainDwzFilter;
rclcpp::Node::SharedPtr nh;

void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  odomTime = rclcpp::Time(odom->header.stamp).seconds();
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odom->pose.pose.orientation;
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  
  if (use3DMode) {
    // 3D模式：考虑完整的3D传感器偏移变换
    float cosRoll = cos(roll);
    float sinRoll = sin(roll);
    float cosPitch = cos(pitch);
    float sinPitch = sin(pitch);
    float cosYaw = cos(yaw);
    float sinYaw = sin(yaw);
    
    // 应用完整的3D旋转变换到传感器偏移向量
    // 第一步：Roll旋转（绕X轴）
    float temp1X = sensorOffsetX;
    float temp1Y = sensorOffsetY * cosRoll;  // 简化：假设传感器主要在XY平面
    float temp1Z = sensorOffsetY * sinRoll;
    
    // 第二步：Pitch旋转（绕Y轴）
    float temp2X = temp1X * cosPitch + temp1Z * sinPitch;
    float temp2Y = temp1Y;
    float temp2Z = -temp1X * sinPitch + temp1Z * cosPitch;
    
    // 第三步：Yaw旋转（绕Z轴）
    float sensorOffsetWorldX = temp2X * cosYaw - temp2Y * sinYaw;
    float sensorOffsetWorldY = temp2X * sinYaw + temp2Y * cosYaw;
    float sensorOffsetWorldZ = temp2Z;
    
    vehicleX = odom->pose.pose.position.x - sensorOffsetWorldX;
    vehicleY = odom->pose.pose.position.y - sensorOffsetWorldY;
    vehicleZ = odom->pose.pose.position.z - sensorOffsetWorldZ;
  } else {
    // 标准模式：仅考虑Yaw旋转的传感器偏移（保持原始逻辑）
    vehicleX = odom->pose.pose.position.x - cos(yaw) * sensorOffsetX + sin(yaw) * sensorOffsetY;
    vehicleY = odom->pose.pose.position.y - sin(yaw) * sensorOffsetX - cos(yaw) * sensorOffsetY;
    vehicleZ = odom->pose.pose.position.z;
  }
}

void laserCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloud2)
{
  if (!useTerrainAnalysis) {
    laserCloud->clear();
    pcl::fromROSMsg(*laserCloud2, *laserCloud);

    pcl::PointXYZI point;
    laserCloudCrop->clear();
    int laserCloudSize = laserCloud->points.size();
    for (int i = 0; i < laserCloudSize; i++) {
      point = laserCloud->points[i];

      float pointX = point.x;
      float pointY = point.y;
      float pointZ = point.z;

      float dis = sqrt((pointX - vehicleX) * (pointX - vehicleX) + (pointY - vehicleY) * (pointY - vehicleY));
      if (dis < adjacentRange) {
        point.x = pointX;
        point.y = pointY;
        point.z = pointZ;
        laserCloudCrop->push_back(point);
      }
    }

    laserCloudDwz->clear();
    laserDwzFilter.setInputCloud(laserCloudCrop);
    laserDwzFilter.filter(*laserCloudDwz);

    newLaserCloud = true;
  }
}

void terrainCloudHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrainCloud2)
{
  if (useTerrainAnalysis) {
    terrainCloud->clear();
    pcl::fromROSMsg(*terrainCloud2, *terrainCloud);

    pcl::PointXYZI point;
    terrainCloudCrop->clear();
    int terrainCloudSize = terrainCloud->points.size();
    for (int i = 0; i < terrainCloudSize; i++) {
      point = terrainCloud->points[i];

      float pointX = point.x;
      float pointY = point.y;
      float pointZ = point.z;

      float dis = sqrt((pointX - vehicleX) * (pointX - vehicleX) + (pointY - vehicleY) * (pointY - vehicleY));
      if (dis < adjacentRange && (point.intensity > obstacleHeightThre || useCost)) {
        point.x = pointX;
        point.y = pointY;
        point.z = pointZ;
        terrainCloudCrop->push_back(point);
      }
    }

    terrainCloudDwz->clear();
    terrainDwzFilter.setInputCloud(terrainCloudCrop);
    terrainDwzFilter.filter(*terrainCloudDwz);

    newTerrainCloud = true;
  }
}

void joystickHandler(const sensor_msgs::msg::Joy::ConstSharedPtr joy)
{
  joyTime = nh->now().seconds();
  joySpeedRaw = sqrt(joy->axes[3] * joy->axes[3] + joy->axes[4] * joy->axes[4]);
  joySpeed = joySpeedRaw;
  if (joySpeed > 1.0) joySpeed = 1.0;
  if (joy->axes[4] == 0) joySpeed = 0;

  if (joySpeed > 0) {
    joyDir = atan2(joy->axes[3], joy->axes[4]) * 180 / PI;
    if (joy->axes[4] < 0) joyDir *= -1;
  }

  if (joy->axes[4] < 0 && !twoWayDrive) joySpeed = 0;

  if (joy->axes[2] > -0.1) {
    autonomyMode = false;
  } else {
    autonomyMode = true;
  }

  if (joy->axes[5] > -0.1) {
    checkObstacle = true;
  } else {
    checkObstacle = false;
  }
}

void goalHandler(const geometry_msgs::msg::PointStamped::ConstSharedPtr goal)
{
  goalX = goal->point.x;
  goalY = goal->point.y;
  goalZ = goal->point.z;   // 处理Z坐标
}

void speedHandler(const std_msgs::msg::Float32::ConstSharedPtr speed)
{
  double speedTime = nh->now().seconds();
  if (autonomyMode && speedTime - joyTime > joyToSpeedDelay && joySpeedRaw == 0) {
    joySpeed = speed->data / maxSpeed;

    if (joySpeed < 0) joySpeed = 0;
    else if (joySpeed > 1.0) joySpeed = 1.0;
  }
}

void boundaryHandler(const geometry_msgs::msg::PolygonStamped::ConstSharedPtr boundary)
{
  boundaryCloud->clear();
  pcl::PointXYZI point, point1, point2;
  int boundarySize = boundary->polygon.points.size();

  if (boundarySize >= 1) {
    point2.x = boundary->polygon.points[0].x;
    point2.y = boundary->polygon.points[0].y;
    point2.z = boundary->polygon.points[0].z;
  }

  for (int i = 0; i < boundarySize; i++) {
    point1 = point2;

    point2.x = boundary->polygon.points[i].x;
    point2.y = boundary->polygon.points[i].y;
    point2.z = boundary->polygon.points[i].z;

    if (point1.z == point2.z) {
      float disX = point1.x - point2.x;
      float disY = point1.y - point2.y;
      float dis = sqrt(disX * disX + disY * disY);

      int pointNum = int(dis / terrainVoxelSize) + 1;
      for (int pointID = 0; pointID < pointNum; pointID++) {
        point.x = float(pointID) / float(pointNum) * point1.x + (1.0 - float(pointID) / float(pointNum)) * point2.x;
        point.y = float(pointID) / float(pointNum) * point1.y + (1.0 - float(pointID) / float(pointNum)) * point2.y;
        point.z = 0;
        point.intensity = 100.0;

        for (int j = 0; j < pointPerPathThre; j++) {
          boundaryCloud->push_back(point);
        }
      }
    }
  }
}

void addedObstaclesHandler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr addedObstacles2)
{
  addedObstacles->clear();
  pcl::fromROSMsg(*addedObstacles2, *addedObstacles);

  int addedObstaclesSize = addedObstacles->points.size();
  for (int i = 0; i < addedObstaclesSize; i++) {
    addedObstacles->points[i].intensity = 200.0;
  }
}

void checkObstacleHandler(const std_msgs::msg::Bool::ConstSharedPtr checkObs)
{
  double checkObsTime = nh->now().seconds();
  if (autonomyMode && checkObsTime - joyTime > joyToCheckObstacleDelay) {
    checkObstacle = checkObs->data;
  }
}

int readPlyHeader(FILE *filePtr)
{
  char str[50];
  int val, pointNum;
  string strCur, strLast;
  while (strCur != "end_header") {
    val = fscanf(filePtr, "%s", str);
    if (val != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
      exit(1);
    }

    strLast = strCur;
    strCur = string(str);

    if (strCur == "vertex" && strLast == "element") {
      val = fscanf(filePtr, "%d", &pointNum);
      if (val != 1) {
        RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
      }
    }
  }

  return pointNum;
}

void readStartPaths()
{
  string fileName = pathFolder + "/startPaths.ply";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  int pointNum = readPlyHeader(filePtr);

  pcl::PointXYZ point;
  int val1, val2, val3, val4, groupID;
  for (int i = 0; i < pointNum; i++) {
    val1 = fscanf(filePtr, "%f", &point.x);
    val2 = fscanf(filePtr, "%f", &point.y);
    val3 = fscanf(filePtr, "%f", &point.z);
    val4 = fscanf(filePtr, "%d", &groupID);

    if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    if (groupID >= 0 && groupID < groupNum) {
      startPaths[groupID]->push_back(point);
    }
  }

  fclose(filePtr);
}

#if PLOTPATHSET == 1
void readPaths()
{
  string fileName = pathFolder + "/paths.ply";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  int pointNum = readPlyHeader(filePtr);

  pcl::PointXYZI point;
  int pointSkipNum = 30;
  int pointSkipCount = 0;
  int val1, val2, val3, val4, val5, pathID;
  for (int i = 0; i < pointNum; i++) {
    val1 = fscanf(filePtr, "%f", &point.x);
    val2 = fscanf(filePtr, "%f", &point.y);
    val3 = fscanf(filePtr, "%f", &point.z);
    val4 = fscanf(filePtr, "%d", &pathID);
    val5 = fscanf(filePtr, "%f", &point.intensity);

    if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1 || val5 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    if (pathID >= 0 && pathID < pathNum) {
      pointSkipCount++;
      if (pointSkipCount > pointSkipNum) {
        paths[pathID]->push_back(point);
        pointSkipCount = 0;
      }
    }
  }

  fclose(filePtr);
}
#endif

void readPathList()
{
  string fileName = pathFolder + "/pathList.ply";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  if (pathNum != readPlyHeader(filePtr)) {
    RCLCPP_INFO(nh->get_logger(), "Incorrect path number, exit.");
    exit(1);
  }

  int val1, val2, val3, val4, val5, pathID, groupID;
  float endX, endY, endZ;
  for (int i = 0; i < pathNum; i++) {
    val1 = fscanf(filePtr, "%f", &endX);
    val2 = fscanf(filePtr, "%f", &endY);
    val3 = fscanf(filePtr, "%f", &endZ);
    val4 = fscanf(filePtr, "%d", &pathID);
    val5 = fscanf(filePtr, "%d", &groupID);

    if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1 || val5 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    if (pathID >= 0 && pathID < pathNum && groupID >= 0 && groupID < groupNum) {
      pathList[pathID] = groupID;
      endDirPathList[pathID] = 2.0 * atan2(endY, endX) * 180 / PI;
    }
  }

  fclose(filePtr);
}

void readCorrespondences()
{
  string fileName = pathFolder + "/correspondences.txt";

  FILE *filePtr = fopen(fileName.c_str(), "r");
  if (filePtr == NULL) {
    RCLCPP_INFO(nh->get_logger(), "Cannot read input files, exit.");
    exit(1);
  }

  int val1, gridVoxelID, pathID;
  for (int i = 0; i < gridVoxelNum; i++) {
    val1 = fscanf(filePtr, "%d", &gridVoxelID);
    if (val1 != 1) {
      RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
        exit(1);
    }

    while (1) {
      val1 = fscanf(filePtr, "%d", &pathID);
      if (val1 != 1) {
        RCLCPP_INFO(nh->get_logger(), "Error reading input files, exit.");
          exit(1);
      }

      if (pathID != -1) {
        if (gridVoxelID >= 0 && gridVoxelID < gridVoxelNum && pathID >= 0 && pathID < pathNum) {
          correspondences[gridVoxelID].push_back(pathID);
        }
      } else {
        break;
      }
    }
  }

  fclose(filePtr);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("localPlanner");

  nh->declare_parameter<std::string>("pathFolder", pathFolder);
  nh->declare_parameter<double>("vehicleLength", vehicleLength);
  nh->declare_parameter<double>("vehicleWidth", vehicleWidth);
  nh->declare_parameter<double>("sensorOffsetX", sensorOffsetX);
  nh->declare_parameter<double>("sensorOffsetY", sensorOffsetY);
  nh->declare_parameter<bool>("twoWayDrive", twoWayDrive);
  nh->declare_parameter<double>("laserVoxelSize", laserVoxelSize);
  nh->declare_parameter<double>("terrainVoxelSize", terrainVoxelSize);
  nh->declare_parameter<bool>("useTerrainAnalysis", useTerrainAnalysis);
  nh->declare_parameter<bool>("checkObstacle", checkObstacle);
  nh->declare_parameter<bool>("checkRotObstacle", checkRotObstacle);
  nh->declare_parameter<double>("adjacentRange", adjacentRange);
  nh->declare_parameter<double>("obstacleHeightThre", obstacleHeightThre);
  nh->declare_parameter<double>("groundHeightThre", groundHeightThre);
  nh->declare_parameter<double>("costHeightThre", costHeightThre);
  nh->declare_parameter<double>("costScore", costScore);
  nh->declare_parameter<bool>("useCost", useCost);
  nh->declare_parameter<int>("pointPerPathThre", pointPerPathThre);
  nh->declare_parameter<double>("minRelZ", minRelZ);
  nh->declare_parameter<double>("maxRelZ", maxRelZ);
  nh->declare_parameter<double>("maxSpeed", maxSpeed);
  nh->declare_parameter<double>("dirWeight", dirWeight);
  nh->declare_parameter<double>("dirThre", dirThre);
  nh->declare_parameter<bool>("dirToVehicle", dirToVehicle);
  nh->declare_parameter<double>("pathScale", pathScale);
  nh->declare_parameter<double>("minPathScale", minPathScale);
  nh->declare_parameter<double>("pathScaleStep", pathScaleStep);
  nh->declare_parameter<bool>("pathScaleBySpeed", pathScaleBySpeed);
  nh->declare_parameter<double>("minPathRange", minPathRange);
  nh->declare_parameter<double>("pathRangeStep", pathRangeStep);
  nh->declare_parameter<bool>("pathRangeBySpeed", pathRangeBySpeed);
  nh->declare_parameter<bool>("pathCropByGoal", pathCropByGoal);
  nh->declare_parameter<bool>("use3DMode", use3DMode);                  //是否启用3d模式
  nh->declare_parameter<bool>("autonomyMode", autonomyMode);
  nh->declare_parameter<double>("autonomySpeed", autonomySpeed);
  nh->declare_parameter<double>("joyToSpeedDelay", joyToSpeedDelay);
  nh->declare_parameter<double>("joyToCheckObstacleDelay", joyToCheckObstacleDelay);
  nh->declare_parameter<double>("goalClearRange", goalClearRange);
  nh->declare_parameter<double>("goalCloseDis_XY", goalCloseDis_XY);
  nh->declare_parameter<double>("goalCloseDis_Z", goalCloseDis_Z);  // Z轴接近距离阈值，用于判断垂直方向是否接近目标
  nh->declare_parameter<double>("goalX", goalX);
  nh->declare_parameter<double>("goalY", goalY);
  nh->declare_parameter<double>("goalZ", goalZ);            // 目标点Z坐标

  nh->get_parameter("pathFolder", pathFolder);
  nh->get_parameter("vehicleLength", vehicleLength);
  nh->get_parameter("vehicleWidth", vehicleWidth);
  nh->get_parameter("sensorOffsetX", sensorOffsetX);
  nh->get_parameter("sensorOffsetY", sensorOffsetY);
  nh->get_parameter("twoWayDrive", twoWayDrive);
  nh->get_parameter("laserVoxelSize", laserVoxelSize);
  nh->get_parameter("terrainVoxelSize", terrainVoxelSize);
  nh->get_parameter("useTerrainAnalysis", useTerrainAnalysis);
  nh->get_parameter("checkObstacle", checkObstacle);
  nh->get_parameter("checkRotObstacle", checkRotObstacle);
  nh->get_parameter("adjacentRange", adjacentRange);
  nh->get_parameter("obstacleHeightThre", obstacleHeightThre);
  nh->get_parameter("groundHeightThre", groundHeightThre);
  nh->get_parameter("costHeightThre", costHeightThre);
  nh->get_parameter("costScore", costScore);
  nh->get_parameter("useCost", useCost);
  nh->get_parameter("pointPerPathThre", pointPerPathThre);
  nh->get_parameter("minRelZ", minRelZ);
  nh->get_parameter("maxRelZ", maxRelZ);
  nh->get_parameter("maxSpeed", maxSpeed);
  nh->get_parameter("dirWeight", dirWeight);
  nh->get_parameter("dirThre", dirThre);
  nh->get_parameter("dirToVehicle", dirToVehicle);
  nh->get_parameter("pathScale", pathScale);
  nh->get_parameter("minPathScale", minPathScale);
  nh->get_parameter("pathScaleStep", pathScaleStep);
  nh->get_parameter("pathScaleBySpeed", pathScaleBySpeed);
  nh->get_parameter("minPathRange", minPathRange);
  nh->get_parameter("pathRangeStep", pathRangeStep);
  nh->get_parameter("pathRangeBySpeed", pathRangeBySpeed);
  nh->get_parameter("pathCropByGoal", pathCropByGoal);
  nh->get_parameter("use3DMode", use3DMode);
  nh->get_parameter("autonomyMode", autonomyMode);
  nh->get_parameter("autonomySpeed", autonomySpeed);
  nh->get_parameter("joyToSpeedDelay", joyToSpeedDelay);
  nh->get_parameter("joyToCheckObstacleDelay", joyToCheckObstacleDelay);
  nh->get_parameter("goalClearRange", goalClearRange);
  nh->get_parameter("goalCloseDis_XY", goalCloseDis_XY);
  nh->get_parameter("goalCloseDis_Z", goalCloseDis_Z);  // Z轴接近距离阈值，用于判断垂直方向是否接近目标
  nh->get_parameter("goalX", goalX);
  nh->get_parameter("goalY", goalY);
  nh->get_parameter("goalZ", goalZ);

  auto subOdometry = nh->create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 5, odometryHandler);

  auto subLaserCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/registered_scan", 5, laserCloudHandler);

  auto subTerrainCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/terrain_map", 5, terrainCloudHandler);

  auto subJoystick = nh->create_subscription<sensor_msgs::msg::Joy>("/joy", 5, joystickHandler);

  auto subGoal = nh->create_subscription<geometry_msgs::msg::PointStamped> ("/way_point", 5, goalHandler);

  auto subSpeed = nh->create_subscription<std_msgs::msg::Float32>("/speed", 5, speedHandler);

  auto subBoundary = nh->create_subscription<geometry_msgs::msg::PolygonStamped>("/navigation_boundary", 5, boundaryHandler);

  auto subAddedObstacles = nh->create_subscription<sensor_msgs::msg::PointCloud2>("/added_obstacles", 5, addedObstaclesHandler);

  auto subCheckObstacle = nh->create_subscription<std_msgs::msg::Bool>("/check_obstacle", 5, checkObstacleHandler);

  auto pubPath = nh->create_publisher<nav_msgs::msg::Path>("/path", 5);
  nav_msgs::msg::Path path;

  #if PLOTPATHSET == 1
  auto pubFreePaths = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/free_paths", 2);
  #endif

  //auto pubLaserCloud = nh->create_publisher<sensor_msgs::msg::PointCloud2> ("/stacked_scans", 2);

  RCLCPP_INFO(nh->get_logger(), "Reading path files.");

  if (autonomyMode) {
    joySpeed = autonomySpeed / maxSpeed;

    if (joySpeed < 0) joySpeed = 0;
    else if (joySpeed > 1.0) joySpeed = 1.0;
  }

  for (int i = 0; i < laserCloudStackNum; i++) {
    laserCloudStack[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  for (int i = 0; i < groupNum; i++) {
    startPaths[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
  }
  #if PLOTPATHSET == 1
  for (int i = 0; i < pathNum; i++) {
    paths[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  #endif
  for (int i = 0; i < gridVoxelNum; i++) {
    correspondences[i].resize(0);
  }

  laserDwzFilter.setLeafSize(laserVoxelSize, laserVoxelSize, laserVoxelSize);
  terrainDwzFilter.setLeafSize(terrainVoxelSize, terrainVoxelSize, terrainVoxelSize);

  readStartPaths();
  #if PLOTPATHSET == 1
  readPaths();
  #endif
  readPathList();
  readCorrespondences();

  RCLCPP_INFO(nh->get_logger(), "Initialization complete.");

  rclcpp::Rate rate(100);
  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);

    if (newLaserCloud || newTerrainCloud) {
      if (newLaserCloud) {
        newLaserCloud = false;

        laserCloudStack[laserCloudCount]->clear();
        *laserCloudStack[laserCloudCount] = *laserCloudDwz;
        laserCloudCount = (laserCloudCount + 1) % laserCloudStackNum;

        plannerCloud->clear();
        for (int i = 0; i < laserCloudStackNum; i++) {
          *plannerCloud += *laserCloudStack[i];
        }
      }

      if (newTerrainCloud) {
        newTerrainCloud = false;

        plannerCloud->clear();
        *plannerCloud = *terrainCloudDwz;
      }

      float sinVehicleRoll = sin(vehicleRoll);
      float cosVehicleRoll = cos(vehicleRoll);
      float sinVehiclePitch = sin(vehiclePitch);
      float cosVehiclePitch = cos(vehiclePitch);
      float sinVehicleYaw = sin(vehicleYaw);
      float cosVehicleYaw = cos(vehicleYaw);

      pcl::PointXYZI point;
      plannerCloudCrop->clear();
      int plannerCloudSize = plannerCloud->points.size();
      for (int i = 0; i < plannerCloudSize; i++) {
        float pointX1 = plannerCloud->points[i].x - vehicleX;
        float pointY1 = plannerCloud->points[i].y - vehicleY;
        float pointZ1 = plannerCloud->points[i].z - vehicleZ;

        if (use3DMode) {
          // 完整的3D坐标变换（与目标点变换保持一致）
          // 第一步：Yaw旋转（绕Z轴）
          float temp1X = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
          float temp1Y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
          float temp1Z = pointZ1;
          
          // 第二步：Pitch旋转（绕Y轴）
          float temp2X = temp1X * cosVehiclePitch - temp1Z * sinVehiclePitch;
          float temp2Y = temp1Y;
          float temp2Z = temp1X * sinVehiclePitch + temp1Z * cosVehiclePitch;
          
          // 第三步：Roll旋转（绕X轴）
          point.x = temp2X;
          point.y = temp2Y * cosVehicleRoll + temp2Z * sinVehicleRoll;
          point.z = -temp2Y * sinVehicleRoll + temp2Z * cosVehicleRoll;
        } else {
          // 简化变换：仅Yaw旋转
          point.x = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
          point.y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
          point.z = pointZ1;
        }
        point.intensity = plannerCloud->points[i].intensity;

        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < adjacentRange && ((point.z > minRelZ && point.z < maxRelZ) || useTerrainAnalysis)) {
          plannerCloudCrop->push_back(point);
        }
      }

      int boundaryCloudSize = boundaryCloud->points.size();
      for (int i = 0; i < boundaryCloudSize; i++) {
        float pointX1 = boundaryCloud->points[i].x - vehicleX;
        float pointY1 = boundaryCloud->points[i].y - vehicleY;
        float pointZ1 = boundaryCloud->points[i].z - vehicleZ;

        if (use3DMode) {
          // 完整的3D坐标变换（与点云处理保持一致）
          // 第一步：Yaw旋转（绕Z轴）
          float temp1X = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
          float temp1Y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
          float temp1Z = pointZ1;
          
          // 第二步：Pitch旋转（绕Y轴）
          float temp2X = temp1X * cosVehiclePitch - temp1Z * sinVehiclePitch;
          float temp2Y = temp1Y;
          float temp2Z = temp1X * sinVehiclePitch + temp1Z * cosVehiclePitch;
          
          // 第三步：Roll旋转（绕X轴）
          point.x = temp2X;
          point.y = temp2Y * cosVehicleRoll + temp2Z * sinVehicleRoll;
          point.z = -temp2Y * sinVehicleRoll + temp2Z * cosVehicleRoll;
        } else {
          // 简化变换：仅Yaw旋转
          point.x = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
          point.y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
          point.z = pointZ1;
        }
        point.intensity = boundaryCloud->points[i].intensity;

        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < adjacentRange) {
          plannerCloudCrop->push_back(point);
        }
      }

      int addedObstaclesSize = addedObstacles->points.size();
      for (int i = 0; i < addedObstaclesSize; i++) {
        float pointX1 = addedObstacles->points[i].x - vehicleX;
        float pointY1 = addedObstacles->points[i].y - vehicleY;
        float pointZ1 = addedObstacles->points[i].z - vehicleZ;

        if (use3DMode) {
          // 完整的3D坐标变换（与点云处理保持一致）
          // 第一步：Yaw旋转（绕Z轴）
          float temp1X = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
          float temp1Y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
          float temp1Z = pointZ1;
          
          // 第二步：Pitch旋转（绕Y轴）
          float temp2X = temp1X * cosVehiclePitch - temp1Z * sinVehiclePitch;
          float temp2Y = temp1Y;
          float temp2Z = temp1X * sinVehiclePitch + temp1Z * cosVehiclePitch;
          
          // 第三步：Roll旋转（绕X轴）
          point.x = temp2X;
          point.y = temp2Y * cosVehicleRoll + temp2Z * sinVehicleRoll;
          point.z = -temp2Y * sinVehicleRoll + temp2Z * cosVehicleRoll;
        } else {
          // 简化变换：仅Yaw旋转
          point.x = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
          point.y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
          point.z = pointZ1;
        }
        point.intensity = addedObstacles->points[i].intensity;

        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < adjacentRange) {
          plannerCloudCrop->push_back(point);
        }
      }

      float pathRange = adjacentRange;
      if (pathRangeBySpeed) pathRange = adjacentRange * joySpeed;
      if (pathRange < minPathRange) pathRange = minPathRange;
      float relativeGoalDis = adjacentRange;
      float relativeGoalDisZ = 0.1;  // 初始化Z轴距离，用于3D导航

      if (autonomyMode) {
        // 计算目标点与车辆的差值（在条件判断前统一计算）
        float deltaX = goalX - vehicleX;
        float deltaY = goalY - vehicleY;
        float deltaZ = goalZ - vehicleZ;
        
        float relativeGoalX, relativeGoalY, relativeGoalZ;
        
        if (use3DMode) {
          // 对于爬楼梯等复杂地形，使用完整的3D坐标变换
          float sinVehicleRoll = sin(vehicleRoll);
          float cosVehicleRoll = cos(vehicleRoll);  
          float sinVehiclePitch = sin(vehiclePitch);
          float cosVehiclePitch = cos(vehiclePitch);
          
          // 完整的3D旋转变换（Yaw -> Pitch -> Roll）
          // 第一步：Yaw旋转（绕Z轴）
          float temp1X = deltaX * cosVehicleYaw + deltaY * sinVehicleYaw;
          float temp1Y = -deltaX * sinVehicleYaw + deltaY * cosVehicleYaw;
          float temp1Z = deltaZ;
          
          // 第二步：Pitch旋转（绕Y轴）
          float temp2X = temp1X * cosVehiclePitch - temp1Z * sinVehiclePitch;
          float temp2Y = temp1Y;
          float temp2Z = temp1X * sinVehiclePitch + temp1Z * cosVehiclePitch;
          
          // 第三步：Roll旋转（绕X轴）
          relativeGoalX = temp2X;
          relativeGoalY = temp2Y * cosVehicleRoll + temp2Z * sinVehicleRoll;
          relativeGoalZ = -temp2Y * sinVehicleRoll + temp2Z * cosVehicleRoll;
        } else {
          // 简化版本：仅进行Yaw旋转，适用于平地导航
          relativeGoalX = (deltaX * cosVehicleYaw + deltaY * sinVehicleYaw);
          relativeGoalY = (-deltaX * sinVehicleYaw + deltaY * cosVehicleYaw);
          relativeGoalZ = deltaZ;  // 简单高度差
        }

        // 分别计算XY平面距离和Z轴距离（适合地面机器人的运动特性）
        float relativeGoalDisXY = sqrt(relativeGoalX * relativeGoalX + relativeGoalY * relativeGoalY);
        relativeGoalDisZ = fabs(relativeGoalZ);
        
        // 更新relativeGoalDis：支持可配置的距离计算方式
        if (use3DMode) {
          // 复杂地形模式：使用3D欧几里得距离（适用于爬楼梯等场景）
          relativeGoalDis = sqrt(relativeGoalDisXY * relativeGoalDisXY + relativeGoalDisZ * relativeGoalDisZ);
        } else {
          // 标准模式：使用XY距离（适用于平地和一般导航）
          relativeGoalDis = relativeGoalDisXY;
        }
        
        joyDir = atan2(relativeGoalY, relativeGoalX) * 180 / PI;

        if (!twoWayDrive) {
          if (joyDir > 90.0) joyDir = 90.0;
          else if (joyDir < -90.0) joyDir = -90.0;
        }
      }

      bool pathFound = false;
      float defPathScale = pathScale;
      if (pathScaleBySpeed) pathScale = defPathScale * joySpeed;
      if (pathScale < minPathScale) pathScale = minPathScale;

      while (pathScale >= minPathScale && pathRange >= minPathRange) {
        for (int i = 0; i < 36 * pathNum; i++) {
          clearPathList[i] = 0;
          pathPenaltyList[i] = 0;
        }
        for (int i = 0; i < 36 * groupNum; i++) {
          clearPathPerGroupScore[i] = 0;
        }

        float minObsAngCW = -180.0;
        float minObsAngCCW = 180.0;
        float diameter = sqrt(vehicleLength / 2.0 * vehicleLength / 2.0 + vehicleWidth / 2.0 * vehicleWidth / 2.0);
        float angOffset = atan2(vehicleWidth, vehicleLength) * 180.0 / PI;
        int plannerCloudCropSize = plannerCloudCrop->points.size();
        for (int i = 0; i < plannerCloudCropSize; i++) {
          float x = plannerCloudCrop->points[i].x / pathScale;
          float y = plannerCloudCrop->points[i].y / pathScale;
          float z = plannerCloudCrop->points[i].z / pathScale;  // 添加Z坐标
          float h = plannerCloudCrop->points[i].intensity;
          
          // 根据3D模式选择距离计算方式（与其他部分保持一致）
          float dis;
          if (use3DMode) {
            dis = sqrt(x * x + y * y + z * z);
          } else {
            dis = sqrt(x * x + y * y);
          }

          if (dis < pathRange / pathScale && (dis <= (relativeGoalDis + goalClearRange) / pathScale || !pathCropByGoal) && checkObstacle) {
            for (int rotDir = 0; rotDir < 36; rotDir++) {
              float rotAng = (10.0 * rotDir - 180.0) * PI / 180;
              float angDiff = fabs(joyDir - (10.0 * rotDir - 180.0));
              if (angDiff > 180.0) {
                angDiff = 360.0 - angDiff;
              }
              if ((angDiff > dirThre && !dirToVehicle) || (fabs(10.0 * rotDir - 180.0) > dirThre && fabs(joyDir) <= 90.0 && dirToVehicle) ||
                  ((10.0 * rotDir > dirThre && 360.0 - 10.0 * rotDir > dirThre) && fabs(joyDir) > 90.0 && dirToVehicle)) {
                continue;
              }

              float x2 = cos(rotAng) * x + sin(rotAng) * y;
              float y2 = -sin(rotAng) * x + cos(rotAng) * y;

              float scaleY = x2 / gridVoxelOffsetX + searchRadius / gridVoxelOffsetY 
                             * (gridVoxelOffsetX - x2) / gridVoxelOffsetX;

              int indX = int((gridVoxelOffsetX + gridVoxelSize / 2 - x2) / gridVoxelSize);
              int indY = int((gridVoxelOffsetY + gridVoxelSize / 2 - y2 / scaleY) / gridVoxelSize);
              if (indX >= 0 && indX < gridVoxelNumX && indY >= 0 && indY < gridVoxelNumY) {
                int ind = gridVoxelNumY * indX + indY;
                int blockedPathByVoxelNum = correspondences[ind].size();
                for (int j = 0; j < blockedPathByVoxelNum; j++) {
                  if (h > obstacleHeightThre || !useTerrainAnalysis) {
                    clearPathList[pathNum * rotDir + correspondences[ind][j]]++;
                  } else {
                    if (pathPenaltyList[pathNum * rotDir + correspondences[ind][j]] < h && h > groundHeightThre) {
                      pathPenaltyList[pathNum * rotDir + correspondences[ind][j]] = h;
                    }
                  }
                }
              }
            }
          }

          if (dis < diameter / pathScale && (fabs(x) > vehicleLength / pathScale / 2.0 || fabs(y) > vehicleWidth / pathScale / 2.0) && 
              (h > obstacleHeightThre || !useTerrainAnalysis) && checkRotObstacle) {
            float angObs = atan2(y, x) * 180.0 / PI;
            if (angObs > 0) {
              if (minObsAngCCW > angObs - angOffset) minObsAngCCW = angObs - angOffset;
              if (minObsAngCW < angObs + angOffset - 180.0) minObsAngCW = angObs + angOffset - 180.0;
            } else {
              if (minObsAngCW < angObs + angOffset) minObsAngCW = angObs + angOffset;
              if (minObsAngCCW > 180.0 + angObs - angOffset) minObsAngCCW = 180.0 + angObs - angOffset;
            }
          }
        }

        if (minObsAngCW > 0) minObsAngCW = 0;
        if (minObsAngCCW < 0) minObsAngCCW = 0;

        for (int i = 0; i < 36 * pathNum; i++) {
          int rotDir = int(i / pathNum);
          float angDiff = fabs(joyDir - (10.0 * rotDir - 180.0));
          if (angDiff > 180.0) {
            angDiff = 360.0 - angDiff;
          }
          if ((angDiff > dirThre && !dirToVehicle) || (fabs(10.0 * rotDir - 180.0) > dirThre && fabs(joyDir) <= 90.0 && dirToVehicle) ||
              ((10.0 * rotDir > dirThre && 360.0 - 10.0 * rotDir > dirThre) && fabs(joyDir) > 90.0 && dirToVehicle)) {
            continue;
          }

          if (clearPathList[i] < pointPerPathThre) {
            float penaltyScore = 1.0 - pathPenaltyList[i] / costHeightThre;
            if (penaltyScore < costScore) penaltyScore = costScore;

            float dirDiff = fabs(joyDir - endDirPathList[i % pathNum] - (10.0 * rotDir - 180.0));
            if (dirDiff > 360.0) {
              dirDiff -= 360.0;
            }
            if (dirDiff > 180.0) {
              dirDiff = 360.0 - dirDiff;
            }

            float rotDirW;
            if (rotDir < 18) rotDirW = fabs(fabs(rotDir - 9) + 1);
            else rotDirW = fabs(fabs(rotDir - 27) + 1);
            float groupDirW = 4  - fabs(pathList[i % pathNum] - 3);
            
            // 3D导航：分别判断XY和Z距离条件
            bool goalCloseXY = (relativeGoalDis < goalCloseDis_XY);      // XY平面接近
            bool goalCloseZ = (relativeGoalDisZ < goalCloseDis_Z);    // Z轴接近
            
            float score;
            if (goalCloseXY && goalCloseZ) {
              // 方法1：XY和Z都接近目标，使用groupDirW权重（更精细的方向控制）
              score = (1 - sqrt(sqrt(dirWeight * dirDiff))) * groupDirW * groupDirW * penaltyScore;
            } else {
              // 方法2：距离目标较远，使用rotDirW权重（更灵活的旋转控制）
              score = (1 - sqrt(sqrt(dirWeight * dirDiff))) * rotDirW * rotDirW * rotDirW * rotDirW * penaltyScore;
            }
            
            if (score > 0) {
              clearPathPerGroupScore[groupNum * rotDir + pathList[i % pathNum]] += score;
            }
          }
        }

        float maxScore = 0;
        int selectedGroupID = -1;
        for (int i = 0; i < 36 * groupNum; i++) {
          int rotDir = int(i / groupNum);
          float rotAng = (10.0 * rotDir - 180.0) * PI / 180;
          float rotDeg = 10.0 * rotDir;
          if (rotDeg > 180.0) rotDeg -= 360.0;
          if (maxScore < clearPathPerGroupScore[i] && ((rotAng * 180.0 / PI > minObsAngCW && rotAng * 180.0 / PI < minObsAngCCW) || 
              (rotDeg > minObsAngCW && rotDeg < minObsAngCCW && twoWayDrive) || !checkRotObstacle)) {
            maxScore = clearPathPerGroupScore[i];
            selectedGroupID = i;
          }
        }

        if (selectedGroupID >= 0) {
          int rotDir = int(selectedGroupID / groupNum);
          float rotAng = (10.0 * rotDir - 180.0) * PI / 180;

          selectedGroupID = selectedGroupID % groupNum;
          int selectedPathLength = startPaths[selectedGroupID]->points.size();
          path.poses.resize(selectedPathLength);
          for (int i = 0; i < selectedPathLength; i++) {
            float x = startPaths[selectedGroupID]->points[i].x;
            float y = startPaths[selectedGroupID]->points[i].y;
            float z = startPaths[selectedGroupID]->points[i].z;
            
            // 根据3D模式选择距离计算方式（与障碍物检测保持一致）
            float dis;
            if (use3DMode) {
              dis = sqrt(x * x + y * y + z * z);
            } else {
              dis = sqrt(x * x + y * y);
            }

            if (dis <= pathRange / pathScale && dis <= relativeGoalDis / pathScale) {
              if (use3DMode) {
                // 3D模式：进行旋转变换，但保持相对于vehicle frame的坐标
                // 第一步：应用路径旋转（rotAng）
                float rotX = cos(rotAng) * x - sin(rotAng) * y;
                float rotY = sin(rotAng) * x + cos(rotAng) * y;
                float rotZ = z;
                
                // 保持相对于vehicle frame，不进行全局坐标变换
                point.x = pathScale * rotX;
                point.y = pathScale * rotY;
                point.z = pathScale * rotZ;
              } else {
                // 标准模式：仅Yaw变换（原始逻辑）
                point.x = pathScale * (cos(rotAng) * x - sin(rotAng) * y);
                point.y = pathScale * (sin(rotAng) * x + cos(rotAng) * y);
                point.z = pathScale * z;
              }
              point.intensity = 1.0;

              // 统一使用vehicle frame，不加vehicle位置
              path.poses[i].pose.position.x = point.x;
              path.poses[i].pose.position.y = point.y;
              path.poses[i].pose.position.z = point.z;
            } else {
              path.poses.resize(i);
              break;
            }
          }

          path.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
          path.header.frame_id = "vehicle";
          pubPath->publish(path);

          #if PLOTPATHSET == 1
          freePaths->clear();
          for (int i = 0; i < 36 * pathNum; i++) {
            int rotDir = int(i / pathNum);
            float rotAng = (10.0 * rotDir - 180.0) * PI / 180;
            float rotDeg = 10.0 * rotDir;
            if (rotDeg > 180.0) rotDeg -= 360.0;
            float angDiff = fabs(joyDir - (10.0 * rotDir - 180.0));
            if (angDiff > 180.0) {
              angDiff = 360.0 - angDiff;
            }
            if ((angDiff > dirThre && !dirToVehicle) || (fabs(10.0 * rotDir - 180.0) > dirThre && fabs(joyDir) <= 90.0 && dirToVehicle) ||
                ((10.0 * rotDir > dirThre && 360.0 - 10.0 * rotDir > dirThre) && fabs(joyDir) > 90.0 && dirToVehicle) || 
                !((rotAng * 180.0 / PI > minObsAngCW && rotAng * 180.0 / PI < minObsAngCCW) || 
                (rotDeg > minObsAngCW && rotDeg < minObsAngCCW && twoWayDrive) || !checkRotObstacle)) {
              continue;
            }

            if (clearPathList[i] < pointPerPathThre) {
              int freePathLength = paths[i % pathNum]->points.size();
              for (int j = 0; j < freePathLength; j++) {
                point = paths[i % pathNum]->points[j];

                float x = point.x;
                float y = point.y;
                float z = point.z;

                // 根据3D模式选择距离计算方式（与其他部分保持一致）
                float dis;
                if (use3DMode) {
                  dis = sqrt(x * x + y * y + z * z);
                } else {
                  dis = sqrt(x * x + y * y);
                }

                if (dis <= pathRange / pathScale && (dis <= (relativeGoalDis + goalClearRange) / pathScale || !pathCropByGoal)) {
                  if (use3DMode) {
                    // 3D模式：进行旋转变换，但保持相对于vehicle frame的坐标
                    float rotX = cos(rotAng) * x - sin(rotAng) * y;
                    float rotY = sin(rotAng) * x + cos(rotAng) * y;
                    float rotZ = z;
                    
                    point.x = pathScale * rotX;
                    point.y = pathScale * rotY;
                    point.z = pathScale * rotZ;
                  } else {
                    // 标准模式：仅Yaw变换（原始逻辑）
                    point.x = pathScale * (cos(rotAng) * x - sin(rotAng) * y);
                    point.y = pathScale * (sin(rotAng) * x + cos(rotAng) * y);
                    point.z = pathScale * z;
                  }
                  point.intensity = 1.0;

                  freePaths->push_back(point);
                }
              }
            }
          }

          sensor_msgs::msg::PointCloud2 freePaths2;
          pcl::toROSMsg(*freePaths, freePaths2);
          freePaths2.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
          freePaths2.header.frame_id = "vehicle";
          pubFreePaths->publish(freePaths2);
          #endif
        }

        if (selectedGroupID < 0) {
          if (pathScale >= minPathScale + pathScaleStep) {
            pathScale -= pathScaleStep;
            pathRange = adjacentRange * pathScale / defPathScale;
          } else {
            pathRange -= pathRangeStep;
          }
        } else {
          pathFound = true;
          break;
        }
      }
      pathScale = defPathScale;

      if (!pathFound) {
        path.poses.resize(1);
        path.poses[0].pose.position.x = 0;
        path.poses[0].pose.position.y = 0;
        path.poses[0].pose.position.z = 0;

        path.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
        path.header.frame_id = "vehicle";
        pubPath->publish(path);

        #if PLOTPATHSET == 1
        freePaths->clear();
        sensor_msgs::msg::PointCloud2 freePaths2;
        pcl::toROSMsg(*freePaths, freePaths2);
        freePaths2.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
        freePaths2.header.frame_id = "vehicle";
        pubFreePaths->publish(freePaths2);
        #endif
      }

      /*sensor_msgs::msg::PointCloud2 plannerCloud2;
      pcl::toROSMsg(*plannerCloudCrop, plannerCloud2);
      plannerCloud2.header.stamp = rclcpp::Time(static_cast<uint64_t>(odomTime * 1e9));
      plannerCloud2.header.frame_id = "vehicle";
      pubLaserCloud->publish(plannerCloud2);*/
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  return 0;
}