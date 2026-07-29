#!/usr/bin/env bash
set -euo pipefail

VID="${HIWONDER_VID:-0483}"
PID="${HIWONDER_PID:-5750}"
ROOT="${TAG_ROOT:-/home/trungbao/CYBER/tag}"
DEEP_RESET=0

if [ "${1:-}" = "--deep-reset" ]; then
  DEEP_RESET=1
fi

echo "Checking Hiwonder USB ${VID}:${PID}..."
if ! lsusb | grep -qi "${VID}:${PID}"; then
  echo "Hiwonder USB ${VID}:${PID} is not visible."
  if [ "$DEEP_RESET" = "0" ]; then
    echo "Run this script with --deep-reset to try a remote USB hub reset:"
    echo "  $0 --deep-reset"
    echo "If that still fails, the device needs a physical replug or power cycle."
    exit 1
  fi

  echo "Trying remote USB reset. This can briefly reset camera/Arduino devices."
  for dev in /dev/bus/usb/*/*; do
    desc="$(lsusb -D "$dev" 2>/dev/null | grep -m1 -E 'idVendor|idProduct' || true)"
    if lsusb -D "$dev" 2>/dev/null | grep -Eq 'idVendor[[:space:]]+0x05e3|idProduct[[:space:]]+0x0610|idProduct[[:space:]]+0x0625'; then
      echo "Resetting possible USB hub $dev"
      sudo usbreset "$dev" || true
      sleep 2
    fi
  done

  sudo udevadm trigger || true
  sleep 2

  if ! lsusb | grep -qi "${VID}:${PID}"; then
    echo "Hiwonder still not visible after hub reset."
    echo "At this point Linux cannot see the controller, so software cannot reconnect it."
    echo "Need physical USB replug, hub power cycle, or a controllable USB power hub."
    exit 1
  fi
fi

echo "Hiwonder USB is visible."

found_hidraw=0
for h in /sys/class/hidraw/hidraw*; do
  [ -e "$h" ] || continue
  if cat "$h/device/uevent" 2>/dev/null | grep -qi "HID_ID=.*${VID}.*${PID}"; then
    dev="/dev/$(basename "$h")"
    echo "Hiwonder hidraw: $dev"
    sudo chmod a+rw "$dev"
    found_hidraw=1
  fi
done

if [ "$found_hidraw" = "0" ]; then
  echo "USB is visible, but no hidraw node was found. Trying USB rebind..."
  usbdev="$(
    for d in /sys/bus/usb/devices/*; do
      [ -f "$d/idVendor" ] && [ -f "$d/idProduct" ] || continue
      if [ "$(cat "$d/idVendor")" = "$VID" ] && [ "$(cat "$d/idProduct")" = "$PID" ]; then
        basename "$d"
        break
      fi
    done
  )"
  if [ -z "$usbdev" ]; then
    echo "Could not find USB device directory for ${VID}:${PID}."
    exit 1
  fi
  echo "Rebinding USB device $usbdev"
  echo -n "$usbdev" | sudo tee /sys/bus/usb/drivers/usb/unbind >/dev/null
  sleep 1
  echo -n "$usbdev" | sudo tee /sys/bus/usb/drivers/usb/bind >/dev/null
  sleep 1
  for h in /sys/class/hidraw/hidraw*; do
    [ -e "$h" ] || continue
    if cat "$h/device/uevent" 2>/dev/null | grep -qi "HID_ID=.*${VID}.*${PID}"; then
      dev="/dev/$(basename "$h")"
      echo "Hiwonder hidraw after rebind: $dev"
      sudo chmod a+rw "$dev"
      found_hidraw=1
    fi
  done
fi

if [ "$found_hidraw" = "0" ]; then
  echo "Still no Hiwonder hidraw node. Try a different USB port or cable."
  exit 1
fi

echo "Restarting Hiwonder ROS node..."
pkill -f hiwonder_compat_node.py 2>/dev/null || true
cd "$ROOT"
source install/setup.bash
setsid ros2 run tag_hiwonder hiwonder_compat_node.py \
  > /tmp/hiwonder_compat_node.log 2>&1 < /dev/null &
sleep 2
pgrep -af hiwonder_compat_node.py || true
tail -30 /tmp/hiwonder_compat_node.log || true
