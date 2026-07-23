#!/usr/bin/env python3

import argparse
import time

try:
    from vassar_feetech_servo_sdk import ServoController
except ImportError as exc:
    raise SystemExit(
        "Missing vassar_feetech_servo_sdk. Install with: "
        "python3 -m pip install --user vassar-feetech-servo-sdk"
    ) from exc


MODE_POSITION = 0
MODE_SPEED = 1


def write_speed(controller, servo_type, servo_id, speed, acc, torque_limit):
    packet = controller.packet_handler
    if servo_type == "hls":
        result = packet.WriteSpec(servo_id, int(speed), int(acc), int(torque_limit))
    else:
        result = packet.WriteSpec(servo_id, int(speed), int(acc))
    if isinstance(result, tuple):
        return result[0] == 0
    return result == 0


def probe_servo_sign(controller, servo_type, servo_id, speed, sec, acc, torque_limit):
    before = controller.read_position(servo_id)
    write_speed(controller, servo_type, servo_id, speed, acc, torque_limit)
    time.sleep(sec)
    write_speed(controller, servo_type, servo_id, 0, acc, torque_limit)
    time.sleep(0.1)
    after = controller.read_position(servo_id)
    delta = after - before
    direction = "increases" if delta > 0 else "decreases" if delta < 0 else "unchanged"
    print(
        f"servo {servo_id}: +{speed} moved position {before} -> {after} "
        f"(delta {delta}, positive speed {direction} position)"
    )
    return delta


def main():
    parser = argparse.ArgumentParser(
        description="Configure/test Feetech servos for Tag."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1000000)
    parser.add_argument("--servo-type", choices=["sts", "hls"], default="sts")
    parser.add_argument("--servo1-id", type=int, default=1)
    parser.add_argument("--servo2-id", type=int, default=2)
    parser.add_argument("--reset-position-1", type=int, default=2048)
    parser.add_argument("--reset-position-2", type=int, default=2048)
    parser.add_argument("--reset-speed", type=int, default=600)
    parser.add_argument("--reset-acc", type=int, default=50)
    parser.add_argument("--mode", choices=["speed", "position"], default="speed")
    parser.add_argument("--set-middle", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--test-speed-1", type=int, default=0)
    parser.add_argument("--test-speed-2", type=int, default=0)
    parser.add_argument("--test-sec", type=float, default=0.0)
    parser.add_argument("--probe-signs", action="store_true")
    parser.add_argument("--probe-speed", type=int, default=80)
    parser.add_argument("--probe-sec", type=float, default=0.2)
    parser.add_argument("--torque-limit", type=int, default=1000)
    args = parser.parse_args()

    servo_ids = [args.servo1_id, args.servo2_id]
    controller = ServoController(
        servo_ids=servo_ids,
        servo_type=args.servo_type,
        port=args.port,
        baudrate=args.baudrate,
    )

    print(f"Connecting to Feetech {args.servo_type.upper()} on {args.port}...")
    controller.connect()
    try:
        positions = controller.read_positions()
        print(f"Current positions: {positions}")
        try:
            voltages = controller.read_voltages()
            print(f"Voltages: {voltages}")
        except Exception as exc:
            print(f"Voltage read skipped/failed: {exc}")

        if args.read_only:
            return

        if args.probe_signs:
            print("Setting operating mode to speed (1) for sign probe...")
            for servo_id in servo_ids:
                ok = controller.set_operating_mode(servo_id, MODE_SPEED)
                print(f"servo {servo_id}: mode set {ok}")
            probe_servo_sign(
                controller,
                args.servo_type,
                args.servo1_id,
                args.probe_speed,
                args.probe_sec,
                args.reset_acc,
                args.torque_limit,
            )
            probe_servo_sign(
                controller,
                args.servo_type,
                args.servo2_id,
                args.probe_speed,
                args.probe_sec,
                args.reset_acc,
                args.torque_limit,
            )
            return

        if args.set_middle:
            print("Calibrating selected servos to middle position...")
            ok = controller.set_middle_position(servo_ids)
            print(f"set_middle_position: {ok}")

        reset_positions = {
            args.servo1_id: args.reset_position_1,
            args.servo2_id: args.reset_position_2,
        }
        print(
            "Moving to reset/balance positions "
            f"{reset_positions} speed={args.reset_speed} acc={args.reset_acc}"
        )
        results = controller.write_position(
            reset_positions,
            speed=args.reset_speed,
            acceleration=args.reset_acc,
        )
        print(f"Position write results: {results}")
        time.sleep(1.0)

        target_mode = MODE_SPEED if args.mode == "speed" else MODE_POSITION
        print(f"Setting operating mode to {args.mode} ({target_mode})...")
        for servo_id in servo_ids:
            ok = controller.set_operating_mode(servo_id, target_mode)
            print(f"servo {servo_id}: mode set {ok}")

        if args.test_sec > 0:
            print(
                f"Testing speed for {args.test_sec:.2f}s: "
                f"({args.test_speed_1}, {args.test_speed_2}) acc={args.reset_acc}"
            )
            write_speed(
                controller,
                args.servo_type,
                args.servo1_id,
                args.test_speed_1,
                args.reset_acc,
                args.torque_limit,
            )
            write_speed(
                controller,
                args.servo_type,
                args.servo2_id,
                args.test_speed_2,
                args.reset_acc,
                args.torque_limit,
            )
            time.sleep(args.test_sec)
            write_speed(
                controller,
                args.servo_type,
                args.servo1_id,
                0,
                args.reset_acc,
                args.torque_limit,
            )
            write_speed(
                controller,
                args.servo_type,
                args.servo2_id,
                0,
                args.reset_acc,
                args.torque_limit,
            )
            print("Speed test stopped.")

        print("\nUse these matching runtime params:")
        print(
            "ros2 run tag_hiwonder feetech_vel_node.py --ros-args "
            f"-p port:={args.port} "
            f"-p baudrate:={args.baudrate} "
            f"-p servo_type:={args.servo_type} "
            f"-p servo1_id:={args.servo1_id} "
            f"-p servo2_id:={args.servo2_id} "
            f"-p reset_position_1:={args.reset_position_1} "
            f"-p reset_position_2:={args.reset_position_2} "
            f"-p reset_speed:={args.reset_speed} "
            f"-p reset_acc:={args.reset_acc}"
        )
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
