#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64.hpp"

#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

namespace
{
constexpr double kPi = 3.14159265358979323846;

double yawFromQuat(double x, double y, double z, double w)
{
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double wrapPi(double angle)
{
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}
}  // namespace

/**
 * 话题驱动的原地转向（风格对齐 pathFollower）：
 *   订阅 /rotate_goal (Float64)  → 设定目标 yaw 并开始转向
 *   订阅 /rotate_cancel (Empty)  → 停止并清零指令
 *   订阅 /localization          → 当前 yaw
 *   发布 /rotate_done (Empty)  → 对齐完成
 *
 * 默认通过 /joy 切入 pathFollower 的 manualMode，由 pathFollower 统一发 Sport Move，
 * 避免与 pathFollower 抢 /api/sport/request。
 * 若 publish_joy_guard:=false，则本节点直接发 Sport Move。
 */
class RotateToYawNode : public rclcpp::Node
{
public:
  RotateToYawNode()
  : Node("rotateToYaw")
  {
    this->declare_parameter<std::string>("localization_topic", "/localization");
    this->declare_parameter<std::string>("rotate_goal_topic", "/rotate_goal");
    this->declare_parameter<std::string>("rotate_cancel_topic", "/rotate_cancel");
    this->declare_parameter<std::string>("rotate_done_topic", "/rotate_done");
    this->declare_parameter<std::string>("sport_request_topic", "/api/sport/request");
    this->declare_parameter<std::string>("joy_topic", "/joy");
    this->declare_parameter<double>("control_rate", 50.0);
    this->declare_parameter<double>("max_yaw_rate", 0.6);
    // 须与 pathFollower 的 maxYawRate（度）一致：manualMode 下
    // angular.z = maxYawRate_deg * pi/180 * axes[0]
    this->declare_parameter<double>("path_follower_max_yaw_rate_deg", 80.0);
    this->declare_parameter<double>("yaw_gain", 1.5);
    this->declare_parameter<double>("yaw_tolerance", 0.08);
    this->declare_parameter<double>("hold_sec", 0.3);
    this->declare_parameter<bool>("publish_joy_guard", true);
    this->declare_parameter<bool>("is_real_robot", true);

    localization_topic_ = this->get_parameter("localization_topic").as_string();
    rotate_goal_topic_ = this->get_parameter("rotate_goal_topic").as_string();
    rotate_cancel_topic_ = this->get_parameter("rotate_cancel_topic").as_string();
    rotate_done_topic_ = this->get_parameter("rotate_done_topic").as_string();
    sport_request_topic_ = this->get_parameter("sport_request_topic").as_string();
    joy_topic_ = this->get_parameter("joy_topic").as_string();
    control_rate_ = this->get_parameter("control_rate").as_double();
    max_yaw_rate_ = this->get_parameter("max_yaw_rate").as_double();
    path_follower_max_yaw_rate_deg_ =
      this->get_parameter("path_follower_max_yaw_rate_deg").as_double();
    yaw_gain_ = this->get_parameter("yaw_gain").as_double();
    yaw_tolerance_ = this->get_parameter("yaw_tolerance").as_double();
    hold_sec_ = this->get_parameter("hold_sec").as_double();
    publish_joy_guard_ = this->get_parameter("publish_joy_guard").as_bool();
    is_real_robot_ = this->get_parameter("is_real_robot").as_bool();

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      localization_topic_, 10,
      std::bind(&RotateToYawNode::odomHandler, this, std::placeholders::_1));
    goal_sub_ = this->create_subscription<std_msgs::msg::Float64>(
      rotate_goal_topic_, 5,
      std::bind(&RotateToYawNode::goalHandler, this, std::placeholders::_1));
    cancel_sub_ = this->create_subscription<std_msgs::msg::Empty>(
      rotate_cancel_topic_, 5,
      std::bind(&RotateToYawNode::cancelHandler, this, std::placeholders::_1));

    sport_pub_ = this->create_publisher<unitree_api::msg::Request>(sport_request_topic_, 10);
    joy_pub_ = this->create_publisher<sensor_msgs::msg::Joy>(joy_topic_, 10);
    done_pub_ = this->create_publisher<std_msgs::msg::Empty>(rotate_done_topic_, 5);

    const double period = 1.0 / std::max(control_rate_, 1.0);
    hold_ticks_needed_ = static_cast<int>(std::ceil(hold_sec_ * std::max(control_rate_, 1.0)));
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(period),
      std::bind(&RotateToYawNode::controlLoop, this));

    RCLCPP_INFO(
      this->get_logger(),
      "rotateToYaw ready: goal=%s, cancel=%s, done=%s, loc=%s, joy_guard=%s",
      rotate_goal_topic_.c_str(), rotate_cancel_topic_.c_str(),
      rotate_done_topic_.c_str(), localization_topic_.c_str(),
      publish_joy_guard_ ? "true" : "false");
  }

private:
  void odomHandler(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const auto & q = msg->pose.pose.orientation;
    std::lock_guard<std::mutex> lock(mutex_);
    current_yaw_ = yawFromQuat(q.x, q.y, q.z, q.w);
    has_yaw_ = true;
  }

  void goalHandler(const std_msgs::msg::Float64::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    target_yaw_ = msg->data;
    active_ = true;
    hold_ticks_ = 0;
    done_published_ = false;
    RCLCPP_INFO(this->get_logger(), "Rotate goal: %.3f rad", target_yaw_);
  }

  void cancelHandler(const std_msgs::msg::Empty::SharedPtr)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = false;
    hold_ticks_ = 0;
    RCLCPP_INFO(this->get_logger(), "Rotate cancelled");
    stopMotion();
  }

  void controlLoop()
  {
    double target = 0.0;
    double yaw = 0.0;
    bool active = false;
    bool has_yaw = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active = active_;
      has_yaw = has_yaw_;
      target = target_yaw_;
      yaw = current_yaw_;
    }

    if (!active || !has_yaw) {
      return;
    }

    const double err = wrapPi(target - yaw);
    if (std::fabs(err) <= yaw_tolerance_) {
      int hold = 0;
      bool should_finish = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        hold_ticks_ += 1;
        hold = hold_ticks_;
        if (hold_ticks_ >= hold_ticks_needed_ && !done_published_) {
          done_published_ = true;
          active_ = false;
          should_finish = true;
        }
      }
      stopMotion();
      if (should_finish) {
        std_msgs::msg::Empty done;
        done_pub_->publish(done);
        RCLCPP_INFO(
          this->get_logger(),
          "Rotate done: yaw=%.3f target=%.3f err=%.3f (hold=%d)",
          yaw, target, err, hold);
      }
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      hold_ticks_ = 0;
    }

    double vyaw = yaw_gain_ * err;
    vyaw = std::max(-max_yaw_rate_, std::min(max_yaw_rate_, vyaw));

    // joy_guard 开启时只发 /joy，让 pathFollower(manualMode) 统一下发 Move
    if (publish_joy_guard_) {
      const double pf_max =
        std::max(1e-6, path_follower_max_yaw_rate_deg_ * kPi / 180.0);
      const float yaw_norm = static_cast<float>(vyaw / pf_max);
      publishJoyGuard(yaw_norm);
    } else {
      publishMove(0.0f, 0.0f, static_cast<float>(vyaw));
    }
  }

  void stopMotion()
  {
    if (publish_joy_guard_) {
      publishJoyGuard(0.0f);
    } else {
      publishMove(0.0f, 0.0f, 0.0f);
    }
  }

  void publishMove(float vx, float vy, float vyaw)
  {
    if (!is_real_robot_) {
      return;
    }
    unitree_api::msg::Request req;
    sport_client_.Move(req, vx, vy, vyaw);
    sport_pub_->publish(req);
  }

  void publishJoyGuard(float yaw_cmd_norm)
  {
    // axes[0]=manual yaw, axes[2]=0 关闭 autonomy, axes[5]=-1 打开 manualMode
    sensor_msgs::msg::Joy joy;
    joy.header.stamp = this->now();
    joy.header.frame_id = "rotateToYaw";
    joy.axes = {
      std::max(-1.0f, std::min(1.0f, yaw_cmd_norm)),
      0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f};
    joy.buttons.assign(11, 0);
    joy_pub_->publish(joy);
  }

  std::string localization_topic_;
  std::string rotate_goal_topic_;
  std::string rotate_cancel_topic_;
  std::string rotate_done_topic_;
  std::string sport_request_topic_;
  std::string joy_topic_;
  double control_rate_{50.0};
  double max_yaw_rate_{0.6};
  double path_follower_max_yaw_rate_deg_{80.0};
  double yaw_gain_{1.5};
  double yaw_tolerance_{0.08};
  double hold_sec_{0.3};
  int hold_ticks_needed_{15};
  bool publish_joy_guard_{true};
  bool is_real_robot_{true};

  std::mutex mutex_;
  double current_yaw_{0.0};
  double target_yaw_{0.0};
  bool has_yaw_{false};
  bool active_{false};
  int hold_ticks_{0};
  bool done_published_{false};

  SportClient sport_client_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr goal_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr cancel_sub_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr done_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RotateToYawNode>());
  rclcpp::shutdown();
  return 0;
}
