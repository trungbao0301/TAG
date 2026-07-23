from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    device = LaunchConfiguration("device")
    gpu_backend = LaunchConfiguration("gpu_backend")
    gpu_device_id = LaunchConfiguration("gpu_device_id")
    require_gpu = LaunchConfiguration("require_gpu")
    estimator_executable = LaunchConfiguration("estimator_executable")

    return LaunchDescription(
        [
            DeclareLaunchArgument("device", default_value="/dev/video2"),
            DeclareLaunchArgument("gpu_backend", default_value="auto"),
            DeclareLaunchArgument("gpu_device_id", default_value="0"),
            DeclareLaunchArgument("require_gpu", default_value="false"),
            DeclareLaunchArgument("camera_fps", default_value="60.0"),
            DeclareLaunchArgument("camera_width", default_value="1280"),
            DeclareLaunchArgument("camera_height", default_value="720"),
            DeclareLaunchArgument("output_width", default_value="640"),
            DeclareLaunchArgument("output_height", default_value="360"),
            DeclareLaunchArgument("border_y", default_value="20"),
            DeclareLaunchArgument("pipeline_fps", default_value="55.0"),
            DeclareLaunchArgument("process_every_n", default_value="1"),
            DeclareLaunchArgument("estimator_executable", default_value="estimator_sub"),
            Node(
                package="tag_camera",
                executable="cam_publisher.py",
                name="tag_camera",
                arguments=[device],
                output="screen",
                parameters=[
                    {
                        "use_gpu": True,
                        "gpu_backend": gpu_backend,
                        "gpu_device_id": ParameterValue(gpu_device_id, value_type=int),
                        "require_gpu": ParameterValue(require_gpu, value_type=bool),
                        "fps": ParameterValue(
                            LaunchConfiguration("camera_fps"), value_type=float
                        ),
                        "width": ParameterValue(
                            LaunchConfiguration("camera_width"), value_type=int
                        ),
                        "height": ParameterValue(
                            LaunchConfiguration("camera_height"), value_type=int
                        ),
                        "output_width": ParameterValue(
                            LaunchConfiguration("output_width"), value_type=int
                        ),
                        "output_height": ParameterValue(
                            LaunchConfiguration("output_height"), value_type=int
                        ),
                        "border_y": ParameterValue(
                            LaunchConfiguration("border_y"), value_type=int
                        ),
                    }
                ],
            ),
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package="tag_state_estimation",
                        executable=estimator_executable,
                        name="tag_state_estimation",
                        output="screen",
                        parameters=[
                            {
                                "use_gpu": True,
                                "gpu_backend": gpu_backend,
                                "gpu_device_id": ParameterValue(
                                    gpu_device_id, value_type=int
                                ),
                                "require_gpu": ParameterValue(
                                    require_gpu, value_type=bool
                                ),
                                "pipeline_fps": ParameterValue(
                                    LaunchConfiguration("pipeline_fps"),
                                    value_type=float,
                                ),
                                "process_every_n": ParameterValue(
                                    LaunchConfiguration("process_every_n"),
                                    value_type=int,
                                ),
                            }
                        ],
                    )
                ],
            ),
        ]
    )
