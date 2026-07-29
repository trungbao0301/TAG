#!/usr/bin/env python3

import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node

from tag_interfaces.msg import HiwonderVel


HELP = """\
WASD servo control

  w/s : servo 1 velocity +/-
  d/a : servo 2 velocity +/-
  x   : reverse both axes
  space : zero command
  q   : quit
"""


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def read_key(timeout):
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return ""
    return sys.stdin.read(1).lower()


def publish(pub, vel_1, vel_2):
    msg = HiwonderVel()
    msg.vel_1 = float(vel_1)
    msg.vel_2 = float(vel_2)
    pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="Manual WASD servo controller.")
    parser.add_argument("--topic", default="/tag_hiwonder/cmd")
    parser.add_argument("--step", type=float, default=20.0)
    parser.add_argument("--max", type=float, default=400.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--decay", type=float, default=0.80)
    args = parser.parse_args()

    rclpy.init()
    node = Node("wasd_servo_control")
    pub = node.create_publisher(HiwonderVel, args.topic, 10)

    vel_1 = 0.0
    vel_2 = 0.0
    reverse = False
    dt = 1.0 / max(args.rate, 1.0)

    print(HELP)
    print(f"Publishing to {args.topic}. step={args.step}, max={args.max}")

    try:
        with RawTerminal():
            while rclpy.ok():
                key = read_key(dt)
                if key == "q":
                    break
                if key == " ":
                    vel_1 = 0.0
                    vel_2 = 0.0
                elif key == "x":
                    reverse = not reverse
                    print(f"\nreverse={reverse}")
                elif key == "d":
                    vel_1 += args.step
                elif key == "a":
                    vel_1 -= args.step
                elif key == "w":
                    vel_2 += args.step
                elif key == "s":
                    vel_2 -= args.step
                elif key == "":
                    vel_1 *= args.decay
                    vel_2 *= args.decay

                vel_1 = max(-args.max, min(args.max, vel_1))
                vel_2 = max(-args.max, min(args.max, vel_2))

                out_1 = -vel_1 if reverse else vel_1
                out_2 = -vel_2 if reverse else vel_2
                publish(pub, out_1, out_2)
                rclpy.spin_once(node, timeout_sec=0.0)
                print(
                    f"\rvel_1={out_1:7.1f} vel_2={out_2:7.1f} "
                    f"raw=({vel_1:6.1f},{vel_2:6.1f})",
                    end="",
                    flush=True,
                )
    finally:
        publish(pub, 0.0, 0.0)
        time.sleep(0.05)
        print("\nSent zero command.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
