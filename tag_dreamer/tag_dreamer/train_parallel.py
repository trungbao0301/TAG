import os
import rclpy
from dreamerv3.train import main as train


def main(args=None):
    rclpy.init(args=args)
    run_name = os.environ.get("TAG_RUN_NAME", "robust_2")
    logdir = os.environ.get(
        "TAG_LOGDIR", os.path.join("~/tag_logs", run_name)
    )
    argv = [
        "--configs",
        "tag",
        "large",  # TODO add config file here!
        "--task",
        "gym_tag_dreamer:tag-ros-v0",
        "--logdir",
        logdir,
        "--replay_size",
        "1e6",
        "--run.script",
        "parallel",
        "--run.train_ratio",
        "-1",
        "--run.save_every",
        "20",
        "--run.log_every",
        "10",
        "--jax.policy_devices",
        "1",
        "--jax.train_devices",
        "0",
    ]

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(argv)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
