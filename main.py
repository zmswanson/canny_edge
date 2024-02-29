from canny_edge import CannyEdgeDetector, GradientKernels, PixelConnectivity
import numpy as np
import cv2
import time

# Load image
image = cv2.imread("data/input/cameraman.tif", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("data/input/fishingboat.tif", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("data/input/lena.bmp", cv2.IMREAD_GRAYSCALE)

start_time = time.time_ns() / 1e6

# Use openCV canny edge detector for comparison
cv2_edges = cv2.Canny(image, 100, 200)

print("OpenCV Canny: ", time.time_ns() / 1e6 - start_time, "ms.")

# Create CannyEdgeDetector object
canny = CannyEdgeDetector(image, sigma=30, kernel_size=5, kernel_type=GradientKernels.SCHARR, low_thresh=0.09, high_thresh=0.27)

smoothed = canny.smooth_image()
grad_x, grad_y = canny.calculate_gradient()

# Use openCV gradient calculation for comparisoncv2_edges
grad_x_cv = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
grad_y_cv = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

magnitude = canny.gradient_magnitude()
magnitude_cv = cv2.magnitude(grad_x_cv, grad_y_cv)

direction = canny.gradient_direction()

nms = canny.non_max_suppression(PixelConnectivity.EIGHT_CONNECTED)
weak, strong = canny.double_threshold()
edges = canny.threshold_hysteresis()

print(max(smoothed.flatten()),
        max(grad_x.flatten()),
        max(grad_y.flatten()),
        max(magnitude.flatten()),
        max(direction.flatten()),
        max(nms.flatten()),
        max(edges.flatten()))

print(max(grad_x_cv.flatten()),
        max(grad_y_cv.flatten()),
        max(magnitude_cv.flatten()),
        max(cv2_edges.flatten()))

# scale everything to 0-1.0
image = (image - image.min()) / (image.max() - image.min())
grad_x = (grad_x - grad_x.min()) / (grad_x.max() - grad_x.min())
grad_y = (grad_y - grad_y.min()) / (grad_y.max() - grad_y.min())
magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
nms = (nms - nms.min()) / (nms.max() - nms.min())
edges = (edges - edges.min()) / (edges.max() - edges.min())

grad_x_cv = (grad_x_cv - grad_x_cv.min()) / (grad_x_cv.max() - grad_x_cv.min())
grad_y_cv = (grad_y_cv - grad_y_cv.min()) / (grad_y_cv.max() - grad_y_cv.min())
magnitude_cv = (magnitude_cv - magnitude_cv.min()) / (magnitude_cv.max() - magnitude_cv.min())
cv2_edges = (cv2_edges - cv2_edges.min()) / (cv2_edges.max() - cv2_edges.min())



cv2.imshow("Original", np.vstack([
    np.hstack([grad_x_cv, grad_y_cv, magnitude_cv, cv2_edges]), 
    np.hstack([grad_x, grad_y, magnitude, edges]),
    np.hstack([nms, image, weak, strong])
]))
cv2.waitKey(0)
cv2.destroyAllWindows()
