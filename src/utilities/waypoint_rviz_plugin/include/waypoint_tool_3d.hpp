#ifndef WAYPOINT_TOOL_3D_H
#define WAYPOINT_TOOL_3D_H

#include <QObject>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <sensor_msgs/msg/joy.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <dddmr_rviz_default_plugins/tools/pose/pose_tool.hpp>

#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_common/tool.hpp>

namespace rviz_common
{
class DisplayContext;
namespace properties
{
class StringProperty;
class QosProfileProperty;
}  // namespace properties
}  // namespace rviz_common

namespace waypoint_rviz_plugin
{
class WaypointTool3D : public dddmr_rviz_default_plugins::tools::PoseTool
{
  Q_OBJECT
public:
  WaypointTool3D();
  ~WaypointTool3D() override;
  virtual void onInitialize() override;

protected:
  void localizationHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom);
  void onPoseSet(double x, double y, double z, double theta) override;

private Q_SLOTS:
  void updateTopic();

private:
  float z_offset_;
  bool has_localization_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_localization_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr pub_joy_;
  rclcpp::Clock::SharedPtr clock_;
  
  rviz_common::properties::StringProperty* topic_property_;
  rviz_common::properties::QosProfileProperty* qos_profile_property_;
  rclcpp::QoS qos_profile_;
};
}

#endif  // WAYPOINT_TOOL_3D_H
