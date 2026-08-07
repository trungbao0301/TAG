"""A minimal stand-in for cv_bridge, for environments where conda cannot install it.

The real cv_bridge is a C++ package. On the training server the ROS environment
pins `ros2-distro-mutex 0.7.0`, while every `ros-humble-cv-bridge` build in the
channel wants `=0.6` or `>=0.9`, and the matching `libopencv` drags in a newer
ffmpeg and openvino. Installing it would mean upgrading a large part of the ROS
stack under a working training environment.

TAG only ever calls two methods, and both are a few lines of numpy:

    bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    bridge.cv2_to_imgmsg(array, encoding="bgr8")

The repo already open-codes the same conversion in tcp_ros_bridge.imgmsg_to_gray64,
so this adds no new assumption about how images arrive -- it just puts the
conversion behind the name the nodes import.
"""
import numpy as np

__all__ = ["CvBridge", "CvBridgeError"]

# Channel count per ROS image encoding.
_CHANNELS = {
    "mono8": 1, "8UC1": 1, "mono16": 1, "16UC1": 1,
    "bgr8": 3, "rgb8": 3, "8UC3": 3,
    "bgra8": 4, "rgba8": 4, "8UC4": 4,
}

_DTYPES = {
    "mono8": np.uint8, "8UC1": np.uint8, "8UC3": np.uint8, "8UC4": np.uint8,
    "bgr8": np.uint8, "rgb8": np.uint8, "bgra8": np.uint8, "rgba8": np.uint8,
    "mono16": np.uint16, "16UC1": np.uint16,
}


class CvBridgeError(TypeError):
    pass


def _as_array(msg):
    encoding = (msg.encoding or "").lower()
    if encoding not in _CHANNELS:
        raise CvBridgeError("unsupported encoding %r" % msg.encoding)
    dtype = _DTYPES[encoding]
    channels = _CHANNELS[encoding]
    data = np.frombuffer(bytes(msg.data), dtype=dtype)
    itemsize = np.dtype(dtype).itemsize
    # step is in BYTES per row and may include padding, so index by row first.
    row_items = int(msg.step) // itemsize
    if row_items <= 0 or msg.height == 0 or msg.width == 0:
        raise CvBridgeError("image message has no pixels")
    array = data[: msg.height * row_items].reshape(msg.height, row_items)
    array = array[:, : msg.width * channels]
    if channels == 1:
        return array.reshape(msg.height, msg.width), encoding
    return array.reshape(msg.height, msg.width, channels), encoding


def _convert(array, source, target):
    if target in ("", "passthrough", source):
        return array
    import cv2

    pairs = {
        ("rgb8", "bgr8"): cv2.COLOR_RGB2BGR,
        ("bgr8", "rgb8"): cv2.COLOR_BGR2RGB,
        ("bgr8", "mono8"): cv2.COLOR_BGR2GRAY,
        ("rgb8", "mono8"): cv2.COLOR_RGB2GRAY,
        ("mono8", "bgr8"): cv2.COLOR_GRAY2BGR,
        ("mono8", "rgb8"): cv2.COLOR_GRAY2RGB,
        ("bgra8", "bgr8"): cv2.COLOR_BGRA2BGR,
        ("rgba8", "bgr8"): cv2.COLOR_RGBA2BGR,
        ("bgra8", "rgb8"): cv2.COLOR_BGRA2RGB,
        ("rgba8", "rgb8"): cv2.COLOR_RGBA2RGB,
    }
    key = (source, target)
    if key not in pairs:
        raise CvBridgeError("no conversion from %s to %s" % (source, target))
    return cv2.cvtColor(array, pairs[key])


class CvBridge:
    """Same surface as the real one, for the two calls TAG makes."""

    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        array, encoding = _as_array(img_msg)
        return _convert(array, encoding, (desired_encoding or "").lower())

    def cv2_to_imgmsg(self, cvim, encoding="passthrough", header=None):
        from sensor_msgs.msg import Image

        array = np.asarray(cvim)
        if array.ndim == 2:
            height, width = array.shape
            channels = 1
        elif array.ndim == 3:
            height, width, channels = array.shape
        else:
            raise CvBridgeError("expected a 2D or 3D array, got %dD" % array.ndim)

        if encoding in ("", "passthrough"):
            encoding = {1: "mono8", 3: "bgr8", 4: "bgra8"}.get(channels)
            if encoding is None:
                raise CvBridgeError("cannot guess an encoding for %d channels" % channels)

        message = Image()
        message.height = int(height)
        message.width = int(width)
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = int(width * channels * array.dtype.itemsize)
        message.data = np.ascontiguousarray(array).tobytes()
        if header is not None:
            message.header = header
        return message
