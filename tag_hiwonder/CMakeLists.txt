cmake_minimum_required(VERSION 3.5)
project(tag_hiwonder)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python executables (Hiwonder servo driver + helpers).
# The old C++ DYNAMIXEL nodes (dynamixel_sdk) were removed; the robot uses the
# Python hiwonder_compat_node.py over HID.
install(PROGRAMS
  scripts/pantilt.py
  scripts/hiwonder_compat_node.py
  scripts/feetech_vel_node.py
  scripts/feetech_config.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
