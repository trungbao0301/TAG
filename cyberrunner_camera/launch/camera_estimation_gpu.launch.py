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
            # The AI-map estimator is the supported one; estimator_sub is the retired
            # HSV/ocam pipeline kept only for reference.
            DeclareLaunchArgument(
                "estimator_executable", default_value="estimator_ai_map"
            ),
            DeclareLaunchArgument("ai_mode", default_value="off"),
            DeclareLaunchArgument("ai_model_path", default_value=""),
            DeclareLaunchArgument("ai_confidence_threshold", default_value="0.90"),
            DeclareLaunchArgument("ai_check_every_n_frames", default_value="5"),
            DeclareLaunchArgument("ai_occlusion_grace_frames", default_value="90"),
            Node(
                package="cyberrunner_camera",
                executable="cam_publisher.py",
                name="cyberrunner_camera",
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
                        package="cyberrunner_state_estimation",
                        executable=estimator_executable,
                        name="cyberrunner_state_estimation",
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
                                "ai_mode": LaunchConfiguration("ai_mode"),
                                "ai_model_path": LaunchConfiguration(
                                    "ai_model_path"
                                ),
                                "ai_confidence_threshold": ParameterValue(
                                    LaunchConfiguration(
                                        "ai_confidence_threshold"
                                    ),
                                    value_type=float,
                                ),
                                "ai_check_every_n_frames": ParameterValue(
                                    LaunchConfiguration(
                                        "ai_check_every_n_frames"
                                    ),
                                    value_type=int,
                                ),
                                "ai_occlusion_grace_frames": ParameterValue(
                                    LaunchConfiguration(
                                        "ai_occlusion_grace_frames"
                                    ),
                                    value_type=int,
                                ),
                            }
                        ],
                    )
                ],
            ),
        ]
    )
