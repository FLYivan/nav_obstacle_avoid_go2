cmake_minimum_required(VERSION 3.8)
project(waypoint_rviz_plugin)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt5 REQUIRED COMPONENTS Core)
find_package(Qt5 REQUIRED COMPONENTS Quick)
find_package(rclcpp REQUIRED)
find_package(rviz2 REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(rviz_ogre_vendor REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(dddmr_rviz_default_plugins REQUIRED)

set(CMAKE_AUTOMOC ON)

# 添加2D/3D工具的头文件
set(HDR_FILES
    include/waypoint_tool.hpp
    include/waypoint_tool_3d.hpp
)

foreach(header "${HDR_FILES}")
  qt5_wrap_cpp(rviz_plugins_moc_files "${header}")
endforeach()

# set(HDR_FILES
#     include/waypoint_tool.hpp
# )

# 添加2D/3D工具的源文件
set(SRC_FILES
  src/waypoint_tool.cpp
  src/waypoint_tool_3d.cpp
)

include_directories(include)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES} ${HDR_FILES} ${rviz_plugins_moc_files})

ament_target_dependencies(${PROJECT_NAME}
  PUBLIC
  "geometry_msgs"
  "rclcpp"
  "rviz_common"
  "rviz_rendering"
  "sensor_msgs"
  "rviz_default_plugins"
  "rviz2"
  "std_msgs"
  "dddmr_rviz_default_plugins" 
  )

target_include_directories(${PROJECT_NAME} PUBLIC
  ${Qt5Widgets_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${rviz_common}
  ${rviz_rendering}
  ${rviz_default_plugins}
  ${pluginlib}
  ${dddmr_rviz_default_plugins_LIBRARIES}
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_export_dependencies(
  geometry_msgs
  rclcpp
  rviz_common
  rviz_rendering
  sensor_msgs
  rviz_default_plugins
  rviz2
  std_msgs
  dddmr_rviz_default_plugins
  )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME} 
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
