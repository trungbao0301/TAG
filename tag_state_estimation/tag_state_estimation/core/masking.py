import numpy as np
import cv2 as cv

from typing import Tuple


def mask_hsv(img, color_params=None, acceleration_backend="cpu"):
    min_h, max_h = color_params[0]
    min_s, max_s = color_params[1]
    min_v, max_v = color_params[2]

    # Red wraps around in HSV (e.g. 170-10): split into two ranges and OR.
    wraps = min_h > max_h

    if (
        acceleration_backend == "cuda"
        and hasattr(cv, "cuda")
        and hasattr(cv, "cuda_GpuMat")
        and all(hasattr(cv.cuda, name) for name in ("cvtColor", "inRange"))
    ):
        img_gpu = cv.cuda_GpuMat()
        img_gpu.upload(img)
        imageHsv = cv.cuda.cvtColor(img_gpu, cv.COLOR_BGR2HSV)
        if wraps:
            lo = np.array([min_h, min_s, min_v])
            hi = np.array([179,   max_s, max_v])
            lo2 = np.array([0,    min_s, min_v])
            hi2 = np.array([max_h, max_s, max_v])
            mask = cv.bitwise_or(
                cv.cuda.inRange(imageHsv, lo, hi).download(),
                cv.cuda.inRange(imageHsv, lo2, hi2).download(),
            )
        else:
            mask = cv.cuda.inRange(
                imageHsv,
                np.array([min_h, min_s, min_v]),
                np.array([max_h, max_s, max_v]),
            ).download()
        return None, mask

    if acceleration_backend == "opencl":
        imageHsv = cv.cvtColor(cv.UMat(img), cv.COLOR_BGR2HSV)
        if wraps:
            lo = np.array([min_h, min_s, min_v])
            hi = np.array([179,   max_s, max_v])
            lo2 = np.array([0,    min_s, min_v])
            hi2 = np.array([max_h, max_s, max_v])
            mask = cv.bitwise_or(
                cv.inRange(imageHsv, lo, hi).get(),
                cv.inRange(imageHsv, lo2, hi2).get(),
            )
        else:
            mask = cv.inRange(
                imageHsv,
                np.array([min_h, min_s, min_v]),
                np.array([max_h, max_s, max_v]),
            ).get()
        return None, mask

    imageHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    if wraps:
        lo = np.array([min_h, min_s, min_v])
        hi = np.array([179,   max_s, max_v])
        lo2 = np.array([0,    min_s, min_v])
        hi2 = np.array([max_h, max_s, max_v])
        mask = cv.bitwise_or(
            cv.inRange(imageHsv, lo, hi),
            cv.inRange(imageHsv, lo2, hi2),
        )
    else:
        mask = cv.inRange(
            imageHsv,
            np.array([min_h, min_s, min_v]),
            np.array([max_h, max_s, max_v]),
        )
    return None, mask


# def masking_hsv():

#     SET_MASK_WINDOW = "Set Mask"
#     cv.namedWindow(SET_MASK_WINDOW)

#     minHue = 0
#     maxHue = 34
#     minSat = 120
#     maxSat = 255
#     minVal = 161
#     maxVal = 255
#     cv.createTrackbar("Min Hue", SET_MASK_WINDOW, minHue, 179, noop)
#     cv.createTrackbar("Max Hue", SET_MASK_WINDOW, maxHue, 179, noop)
#     cv.createTrackbar("Min Sat", SET_MASK_WINDOW, minSat, 255, noop)
#     cv.createTrackbar("Max Sat", SET_MASK_WINDOW, maxSat, 255, noop)
#     cv.createTrackbar("Min Val", SET_MASK_WINDOW, minVal, 255, noop)
#     cv.createTrackbar("Max Val", SET_MASK_WINDOW, maxVal, 255, noop)

#      ## 2. Read and convert image to HSV color space
#     image = cv.imread("imgs/sub_edge_case0.png")
#     #image = undistort_img(raw)
#     imageHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     width  = image.shape[1]
#     height = image.shape[0]
#     size_ratio = 0.5

#     winIn = cv.namedWindow("in", cv.WINDOW_NORMAL)
#     cv.resizeWindow("in", int(width * size_ratio), int(height * size_ratio))

#     winOut = cv.namedWindow("out", cv.WINDOW_NORMAL)
#     cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))

#     while True:


#         ## 3. Get min and max HSV values from Set Mask window
#         minHue = cv.getTrackbarPos("Min Hue", SET_MASK_WINDOW)
#         maxHue = cv.getTrackbarPos("Max Hue", SET_MASK_WINDOW)
#         minSat = cv.getTrackbarPos("Min Sat", SET_MASK_WINDOW)
#         maxSat = cv.getTrackbarPos("Max Sat", SET_MASK_WINDOW)
#         minVal = cv.getTrackbarPos("Min Val", SET_MASK_WINDOW)
#         maxVal = cv.getTrackbarPos("Max Val", SET_MASK_WINDOW)
#         minHsv = np.array([minHue, minSat, minVal])
#         maxHsv = np.array([maxHue, maxSat, maxVal])

#         ## 4. Create mask and result (masked) image
#         # params: input array, lower boundary array, upper boundary array
#         mask = cv.inRange(imageHsv, minHsv, maxHsv)
#         cv.imwrite("imgs/mask.jpg", mask)

#         # params: src1 array, src2 array, mask
#         resultImage = cv.bitwise_and(image, image, mask=mask)
#         cropped = resultImage[:, :]
#         cv.imwrite("imgs/feature.jpg", cropped)

#         ## 5. Show images
#         win = cv.namedWindow("out", cv.WINDOW_NORMAL)
#         # size_ratio = 5
#         # height, width,_ = resultImage.shape
#         # cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))
#         #cv.resizeWindow("out", 1000,1000)
#         cv.imshow("in", image)
#         # cv.imshow("Mask", mask)   # optional
#         cv.imshow("out", resultImage)
#         resultImage = resultImage.astype(np.uint8)
#         cv.imwrite("imgs/masked.jpg", resultImage)
#         if cv.waitKey(1) == 27: break   # Wait Esc key to end program


# def save_hsv():
#     image = cv.imread("img_proc_python/imgs/sub.jpg")
#     img_masked  = mask_hsv(image)
#     cv.imwrite("imgs/disk_masked.jpg", img_masked)
#     #cv.imwrite("imgs/disk.jpg", img_processed)

# def mask_hsb(img):
#     brightness = np.sum(img, axis = 2) / 3
#     mask = (brightness > 90).astype(float) * 255
#     img_masked = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
#     return img_masked, img_masked

# def save_hsb():
#     image = cv.imread("imgs/sub_hsv.jpg")
#     img_masked, img_processed = mask_hsb(image)
#     cv.imwrite("imgs/sub_masked_hsb.jpg", img_masked)
#     cv.imwrite("imgs/sub_hsb.jpg", img_processed)

# def noop(x):
#     pass

# def masking_hsv_by():
#     SET_MASK_WINDOW = "Set Mask"
#     cv.namedWindow(SET_MASK_WINDOW)

#     minHue = 0
#     maxHue = 34
#     minSat = 120
#     maxSat = 255
#     minVal = 161
#     maxVal = 255
#     cv.createTrackbar("Min Hue", SET_MASK_WINDOW, minHue, 179, noop)
#     cv.createTrackbar("Max Hue", SET_MASK_WINDOW, maxHue, 179, noop)
#     cv.createTrackbar("Min Sat", SET_MASK_WINDOW, minSat, 255, noop)
#     cv.createTrackbar("Max Sat", SET_MASK_WINDOW, maxSat, 255, noop)
#     cv.createTrackbar("Min Val", SET_MASK_WINDOW, minVal, 255, noop)
#     cv.createTrackbar("Max Val", SET_MASK_WINDOW, maxVal, 255, noop)

#      ## 2. Read and convert image to HSV color space
#     image = cv.imread("imgs/edge_case0.png")
#     #image = undistort_img(raw)
#     imageHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     width  = image.shape[1]
#     height = image.shape[0]
#     size_ratio = 0.5

#     winIn = cv.namedWindow("in", cv.WINDOW_NORMAL)
#     cv.resizeWindow("in", int(width * size_ratio), int(height * size_ratio))

#     winOut = cv.namedWindow("out", cv.WINDOW_NORMAL)
#     cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))

#     while True:


#         ## 3. Get min and max HSV values from Set Mask window
#         minHue = cv.getTrackbarPos("Min Hue", SET_MASK_WINDOW)
#         maxHue = cv.getTrackbarPos("Max Hue", SET_MASK_WINDOW)
#         minSat = cv.getTrackbarPos("Min Sat", SET_MASK_WINDOW)
#         maxSat = cv.getTrackbarPos("Max Sat", SET_MASK_WINDOW)
#         minVal = cv.getTrackbarPos("Min Val", SET_MASK_WINDOW)
#         maxVal = cv.getTrackbarPos("Max Val", SET_MASK_WINDOW)
#         minHsv = np.array([minHue, minSat, minVal])
#         maxHsv = np.array([maxHue, maxSat, maxVal])

#         ## 4. Create mask and result (masked) image
#         # params: input array, lower boundary array, upper boundary array
#         mask = cv.inRange(imageHsv, minHsv, maxHsv)
#         cv.imwrite("imgs/mask.jpg", mask)

#         # params: src1 array, src2 array, mask
#         resultImage = cv.bitwise_and(image, image, mask=mask)
#         cropped = resultImage[:, :]
#         cv.imwrite("imgs/feature.jpg", cropped)

#         ## 5. Show images
#         win = cv.namedWindow("out", cv.WINDOW_NORMAL)
#         # size_ratio = 5
#         # height, width,_ = resultImage.shape
#         # cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))
#         #cv.resizeWindow("out", 1000,1000)
#         cv.imshow("in", image)
#         # cv.imshow("Mask", mask)   # optional
#         cv.imshow("out", resultImage)
#         resultImage = resultImage.astype(np.uint8)
#         cv.imwrite("imgs/masked.jpg", resultImage)
#         if cv.waitKey(1) == 27: break   # Wait Esc key to end program
