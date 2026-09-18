#include <waypoint_tool.hpp>

#include <string>

#include <rviz_common/display_context.hpp>
#include <rviz_common/logging.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_common/properties/qos_profile_property.hpp>

namespace waypoint_rviz_plugin
{
WaypointTool::WaypointTool()
: rviz_default_plugins::tools::PoseTool(), qos_profile_(5),
  localization_z_(0), state_z_(0), has_localization_(false)
{
  shortcut_key_ = 'w';

  topic_property_ = new rviz_common::properties::StringProperty("Topic", "waypoint", "The topic on which to publish navigation waypionts.",
                                       getPropertyContainer(), SLOT(updateTopic()), this);
  
  qos_profile_property_ = new rviz_common::properties::QosProfileProperty(
    topic_property_, qos_profile_);
}

WaypointTool::~WaypointTool() = default;

void WaypointTool::onInitialize()
{
  rviz_default_plugins::tools::PoseTool::onInitialize();
  qos_profile_property_->initialize(
    [this](rclcpp::QoS profile) {this->qos_profile_ = profile;});
  setName("Waypoint");
  updateTopic();
}

void WaypointTool::updateTopic()
{
  rclcpp::Node::SharedPtr raw_node =
    context_->getRosNodeAbstraction().lock()->get_raw_node();
  sub_localization_ = raw_node->template create_subscription<nav_msgs::msg::Odometry>(
    "/localization", 5, std::bind(&WaypointTool::localizationHandler, this, std::placeholders::_1));
  sub_state_ = raw_node->template create_subscription<nav_msgs::msg::Odometry>(
    "/state_estimation", 5, std::bind(&WaypointTool::odomHandler, this, std::placeholders::_1));

  pub_ = raw_node->template create_publisher<geometry_msgs::msg::PointStamped>("/way_point", qos_profile_);
  pub_joy_ = raw_node->template create_publisher<sensor_msgs::msg::Joy>("/joy", qos_profile_);
  clock_ = raw_node->get_clock();
}

void WaypointTool::localizationHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  localization_z_ = odom->pose.pose.position.z;
  last_localization_time_ = clock_ ? clock_->now() : rclcpp::Clock().now();
  has_localization_ = true;
}

void WaypointTool::odomHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom)
{
  state_z_ = odom->pose.pose.position.z;
}

float WaypointTool::selectVehicleZ(const char ** source) const
{
  *source = "/state_estimation";
  if (!has_localization_ || !clock_) {
    return state_z_;
  }
  const double age = (clock_->now() - last_localization_time_).seconds();
  if (age < 1.0) {
    *source = "/localization";
    return localization_z_;
  }
  return state_z_;
}

void WaypointTool::onPoseSet(double x, double y, double theta)
{
  sensor_msgs::msg::Joy joy;

  // 添加彩色日志输出
  const char * z_source = "/state_estimation";
  const float vehicle_z = selectVehicleZ(&z_source);
  RCLCPP_INFO_STREAM(
    context_->getRosNodeAbstraction().lock()->get_raw_node()->get_logger(),
    "\033[1;32m[WaypointTool]\033[0m 设置2D导航点 - 坐标: (" << x << ", " << y
    << ", " << vehicle_z << ") z来源: " << z_source);

  joy.axes.push_back(0);
  joy.axes.push_back(0);
  joy.axes.push_back(-1.0);
  joy.axes.push_back(0);
  joy.axes.push_back(1.0);
  joy.axes.push_back(1.0);
  joy.axes.push_back(0);
  joy.axes.push_back(0);

  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(1);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);
  joy.buttons.push_back(0);

  joy.header.stamp = clock_->now();
  joy.header.frame_id = "waypoint_tool";
  pub_joy_->publish(joy);

  geometry_msgs::msg::PointStamped waypoint;
  waypoint.header.frame_id = "map";
  waypoint.header.stamp = joy.header.stamp;
  waypoint.point.x = x;
  waypoint.point.y = y;
  waypoint.point.z = vehicle_z;

  pub_->publish(waypoint);
  usleep(10000);
  pub_->publish(waypoint);
}
}

#include <pluginlib/class_list_macros.hpp> 
PLUGINLIB_EXPORT_CLASS(waypoint_rviz_plugin::WaypointTool, rviz_common::Tool)
