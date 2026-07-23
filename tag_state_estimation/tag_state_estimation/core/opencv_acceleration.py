import cv2


def configure_opencv_acceleration(use_gpu, backend, device_id=0, require_gpu=False):
    backend = str(backend).strip().lower()
    if backend not in ("auto", "cuda", "opencl", "cpu"):
        raise ValueError(
            f"Unsupported gpu_backend '{backend}'. Use auto, cuda, opencl, or cpu."
        )

    if not use_gpu or backend == "cpu":
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
        return "cpu", "OpenCV GPU acceleration disabled; using CPU."

    reasons = []

    if backend in ("auto", "cuda"):
        cuda_count = 0
        if hasattr(cv2, "cuda"):
            try:
                cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            except cv2.error as exc:
                reasons.append(f"CUDA query failed: {exc}")

        cuda_ops_available = (
            hasattr(cv2, "cuda")
            and hasattr(cv2, "cuda_GpuMat")
            and all(hasattr(cv2.cuda, name) for name in ("cvtColor", "inRange"))
        )

        if cuda_count > 0 and cuda_ops_available:
            cv2.cuda.setDevice(int(device_id))
            return "cuda", f"Using OpenCV CUDA device {device_id} for HSV masking."

        reasons.append(
            "CUDA unavailable"
            if cuda_count <= 0
            else "OpenCV CUDA cvtColor/inRange bindings unavailable"
        )

    if backend in ("auto", "opencl"):
        if hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                return "opencl", "Using OpenCV OpenCL/UMat for estimation masking."
            reasons.append("OpenCL present but OpenCV did not enable it")
        else:
            reasons.append("OpenCL unavailable")

    message = f"GPU requested but unavailable ({'; '.join(reasons)}); using CPU."
    if require_gpu:
        raise RuntimeError(message)
    return "cpu", message
