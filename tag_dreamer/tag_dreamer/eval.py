import sys
from datetime import datetime
import rclpy
from dreamerv3.train import main as train
from datetime import datetime


def main(args=None):
    rclpy.init(args=args)
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y, %H:%M:%S")
    date_str = now.strftime("%Y%m%d-%H%M%S")
    logdir = "robust_2"  #'meetyourlab' # 'tag_clean_1100k' #'wef'#'tag_clean_10k' #  'meetyourlab'
    checkpoint = "~/tag_logs/{}/checkpoint.ckpt".format(logdir)
    # date_str = '2023-08-23:04-17-46'
    argv = [
        "--configs",
        "tag",
        "large",  # TODO add config file here!
        "--task",
        "gym_tag_dreamer:tag-ros-v0",
        "--logdir",
        "~/tag_logs/eval_" + date_str,
        "--run.from_checkpoint",
        checkpoint,
        "--run.steps",
        "10000",
        "--run.script",
        "eval_only",
        "--jax.policy_devices",
        "0",
        "--jax.train_devices",
        "0",
    ]
    train(argv)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
