__author__ = "Zachary M. Swanson"
__status__ = "Development"
__version__ = "1.0.0"
__description__ = "Canny Edge Detector implementation in Python using Numpy."

import numpy as np
import enum
import time

class GradientKernels(enum.Enum):
    """
    Enum class for the different gradient kernels that can be used in the Canny Edge Detector.
    """
    SOBEL = 1
    PREWITT = 2
    SCHARR = 3

class PixelConnectivity(enum.Enum):
    """
    Enum class for the different pixel connectivities that can be used in the Canny Edge Detector.
    """
    FOUR_CONNECTED = 1
    EIGHT_CONNECTED = 2

class CannyEdgeDetector:
    """
    Canny Edge Detector implementation in Python using Numpy. This class provides a simple
    interface for detecting edges in an image using the Canny Edge Detector algorithm. The
    algorithm consists of the following steps:
    1. Apply Gaussian blur to the image.
    2. Calculate the gradient of the smoothed image using a specified kernel type.
    3. Calculate the gradient magnitude and direction.
    4. Perform non-maximum suppression on the gradient magnitude image.
    5. Perform double thresholding on the non-maximum suppressed image.
    6. Perform threshold hysteresis on the weak and strong edges.
    7. Return the final Canny-detected edges.
    
    The user can specify the following parameters:
    - sigma: The standard deviation of the Gaussian filter.
    - kernel_size: The size of the kernel for the Gaussian filter.
    - kernel_type: The type of kernel to use for gradient calculation (Sobel, Prewitt, Scharr).
    - neighbor_depth: The depth of neighbors to consider for non-maximum suppression.
    - pixel_connectivity: The pixel connectivity to use for non-maximum suppression (4-connected, 8-connected).
    - low_thresh: The low threshold for the double thresholding.
    - high_thresh: The high threshold for the double thresholding.
    """
    def __init__(
                self, image: np.array, sigma=1, kernel_size=5, kernel_type=GradientKernels.SOBEL,
                neighbor_depth=1, pixel_connectivity=PixelConnectivity.EIGHT_CONNECTED,
                low_thresh=0.05, high_thresh=0.15
        ):
        if len(image.shape) != 2:
            raise ValueError("Image must be a 2D array (grayscale)")
        
        self.image = image.astype(np.float32)

        # print(f"Image shape: {self.image.shape} ... Image dtype: {self.image.dtype}")
        # print(f"Canny {sigma}, {kernel_size}, {kernel_type.name}, {neighbor_depth}, {pixel_connectivity.name}, {low_thresh}, {high_thresh}")

        self.sigma = sigma
        self.kernel_size = kernel_size

        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        elif 5 < self.kernel_size < 3:
            raise ValueError("Kernel size must be 3 or 5") 

        self.kernel_type = kernel_type

        if self.kernel_type == GradientKernels.SOBEL:
            self.kern_x, self.kern_y = self.sobel_filter()
            # print("SOBEL")
        elif self.kernel_type == GradientKernels.PREWITT:
            self.kern_x, self.kern_y = self.prewitt_filter()
            # print("PREWITT")
        elif self.kernel_type == GradientKernels.SCHARR:
            self.kern_x, self.kern_y = self.scharr_filter()
            # print("SCHARR")
        else:
            raise ValueError("Invalid kernel type")
        
        self.neighbor_depth = neighbor_depth
        self.pixel_connectivity = pixel_connectivity
        
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

        self.smoothed_image = None
        self.grad_x = None
        self.grad_y = None
        self.magnitude = None
        self.direction = None
        self.nms = None
        self.strong_edges = None
        self.weak_edges = None
        self.canny_edges = None


    def gaussian_kernel(self, size, sigma=1):
        """
        Generate a nxn Gaussian kernel given a size (n) and sigma value.
        """
        half_size = int(size) // 2
        x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        kern_gauss =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        kern_gauss = kern_gauss / kern_gauss.sum()
        
        return kern_gauss
    

    def sobel_filter(self):
        """
        Generate the 3x3 Sobel filter kernels for X and Y gradients.
        """
        kern_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kern_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

        kern_x = kern_x / np.sum(np.abs(kern_x))
        kern_y = kern_y / np.sum(np.abs(kern_y))

        return (kern_x, kern_y)
    
    def prewitt_filter(self):
        """
        Generate the 3x3 Prewitt filter kernels for X and Y gradients.
        """
        kern_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        kern_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

        kern_x = kern_x / np.sum(np.abs(kern_x))
        kern_y = kern_y / np.sum(np.abs(kern_y))

        return (kern_x, kern_y)

    def scharr_filter(self):
        """
        Generate the 3x3 Scharr filter kernels for X and Y gradients.
        """
        kern_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32)
        kern_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], np.float32)

        kern_x = kern_x / np.sum(np.abs(kern_x))
        kern_y = kern_y / np.sum(np.abs(kern_y))

        return (kern_x, kern_y)


    def smooth_image(self):
        """
        Apply a Gaussian blur to the image using the specified sigma and kernel size. Accomplished
        by convolving the image with the Gaussian kernel.
        """
        if self.image is None:
            raise ValueError("Image not found")

        kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        padded_image = np.pad(self.image, pad_width=self.kernel_size//2, mode='constant', constant_values=0)
        smoothed_image = np.zeros(self.image.shape)

        for i in range(0, padded_image.shape[0]-self.kernel_size+1):
            for j in range(0, padded_image.shape[1]-self.kernel_size+1):
                smoothed_image[i, j] = np.sum(np.multiply(padded_image[i:i+self.kernel_size, j:j+self.kernel_size], kernel))

        self.smoothed_image = smoothed_image

        return smoothed_image


    def calculate_gradient(self):
        """
        Calculate the gradient of the smoothed image using the specified kernel type (Sobel, 
        Prewitt, Scharr). Gradient is calculated in the X and Y directions by convolving the
        smoothed image with the respective kernel.
        """
        if self.smoothed_image is None:
            self.smoothed_image = self.smooth_image()

        if self.kern_x is None or self.kern_y is None:
            if self.kernel_type == GradientKernels.SOBEL:
                self.kern_x, self.kern_y = self.sobel_filter()
            elif self.kernel_type == GradientKernels.PREWITT:
                self.kern_x, self.kern_y = self.prewitt_filter()
            elif self.kernel_type == GradientKernels.SCHARR:
                self.kern_x, self.kern_y = self.scharr_filter()

        padded_image = np.pad(self.smoothed_image, pad_width=1, mode='constant', constant_values=0)

        grad_x = np.zeros(self.smoothed_image.shape)
        grad_y = np.zeros(self.smoothed_image.shape)
        
        for i in range(0, padded_image.shape[0]-3+1):
            for j in range(0, padded_image.shape[1]-3+1):
                grad_x[i, j] = np.sum(np.multiply(padded_image[i:i+3, j:j+3], self.kern_x))
                grad_y[i, j] = np.sum(np.multiply(padded_image[i:i+3, j:j+3], self.kern_y))

        self.grad_x = grad_x
        self.grad_y = grad_y

        return grad_x, grad_y
        
    
    def gradient_magnitude(self):
        """
        Calculate the gradient magnitude of the image using the X and Y gradients. The magnitude
        is calculated as the square root of the sum of the squares of the X and Y gradients.
        """
        if self.grad_x is None or self.grad_y is None:
            self.grad_x, self.grad_y = self.calculate_gradient()

        self.magnitude = np.sqrt(np.square(self.grad_x) + np.square(self.grad_y))

        return self.magnitude
    

    def gradient_direction(self):
        """
        Calculate the gradient direction of the image using the X and Y gradients. The direction
        is calculated as the arctan of the Y gradient divided by the X gradient. Note that the
        arctan2 function is used to ensure the correct quadrant of the angle is returned. Also note
        that the angle is returned in radians.
        """
        if self.grad_x is None or self.grad_y is None:
            self.grad_x, self.grad_y = self.calculate_gradient()

        self.direction = np.arctan2(self.grad_y, self.grad_x)

        return self.direction
    
    
    def non_max_suppression(self, pixel_connectivity=None, neighbor_depth=None):
        """
        Perform non-maximum suppression on the gradient magnitude image. This is accomplished by
        comparing the magnitude of the current pixel to the magnitude of its neighbors in the
        direction of the gradient. If the magnitude of the current pixel is greater than its
        neighbors, it is kept, otherwise it is set to 0. The neighbor depth and pixel connectivity
        can be specified, but default to 1 and 8-connected, respectively.
        """
        if self.magnitude is None:
            self.magnitude = self.gradient_magnitude()

        if self.direction is None:
            self.direction = self.gradient_direction()

        if neighbor_depth is None:
            if self.neighbor_depth is None or self.neighbor_depth < 1:
                self.neighbor_depth = 1
            neighbor_depth = self.neighbor_depth
        else:
            self.neighbor_depth = neighbor_depth

        if pixel_connectivity is None:
            if self.pixel_connectivity is None:
                self.pixel_connectivity = PixelConnectivity.EIGHT_CONNECTED
            pixel_connectivity = self.pixel_connectivity
        else:
            self.pixel_connectivity = pixel_connectivity

        tmp_nms_image = np.zeros(self.magnitude.shape)
        angle = self.direction * 180. / np.pi
        angle[angle < 0] += 180


        if pixel_connectivity == PixelConnectivity.FOUR_CONNECTED:
            for i in range(neighbor_depth, self.magnitude.shape[0]-neighbor_depth):
                for j in range(neighbor_depth, self.magnitude.shape[1]-neighbor_depth):
                    if (0 <= angle[i,j] < 45) or (135 <= angle[i,j] <= 180):
                        next = [self.magnitude[i, j+k] for k in range(1, neighbor_depth+1)]
                        prev = [self.magnitude[i, j-k] for k in range(1, neighbor_depth+1)]
                    elif (45 <= angle[i,j] < 135):
                        next = [self.magnitude[i+k, j] for k in range(1, neighbor_depth+1)]
                        prev = [self.magnitude[i-k, j] for k in range(1, neighbor_depth+1)]

                    if self.magnitude[i,j] >= max(next + prev):
                        tmp_nms_image[i,j] = self.magnitude[i,j]
                    else:
                        tmp_nms_image[i,j] = 0
        else:
            for i in range(neighbor_depth, self.magnitude.shape[0]-neighbor_depth):
                for j in range(neighbor_depth, self.magnitude.shape[1]-neighbor_depth):
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        next = [self.magnitude[i, j+k] for k in range(1, neighbor_depth+1)]
                        prev = [self.magnitude[i, j-k] for k in range(1, neighbor_depth+1)]
                    elif (22.5 <= angle[i,j] < 67.5):
                        next = [self.magnitude[i+k, j+k] for k in range(1, neighbor_depth+1)]
                        prev = [self.magnitude[i-k, j-k] for k in range(1, neighbor_depth+1)]
                    elif (67.5 <= angle[i,j] < 112.5):
                        next = [self.magnitude[i+k, j] for k in range(1, neighbor_depth+1)]
                        prev = [self.magnitude[i-k, j] for k in range(1, neighbor_depth+1)]
                    elif (112.5 <= angle[i,j] < 157.5):
                        next = [self.magnitude[i-k, j+k] for k in range(1, neighbor_depth+1)]
                        prev = [self.magnitude[i+k, j-k] for k in range(1, neighbor_depth+1)]

                    if self.magnitude[i,j] >= max(next + prev):
                        tmp_nms_image[i,j] = self.magnitude[i,j]
                    else:
                        tmp_nms_image[i,j] = 0

        self.nms = tmp_nms_image
        return self.nms
    

    def double_threshold(self):
        """
        Generates weak and strong edge arrays based on the non-maximum suppression image. The high
        and low thresholds are used to determine which edges are strong and weak: strong edges are
        those with a magnitude greater than the high threshold, weak edges are those with a
        magnitude between the high and low thresholds. The weak and strong edges are returned.
        """
        if self.nms is None:
            self.nms = self.non_max_suppression()
        
        high_thresh = self.nms.max() * self.high_thresh
        low_thresh = self.nms.max() * self.low_thresh

        strong_edges = (self.nms > high_thresh)
        weak_edges = (self.nms > low_thresh) & (self.nms < high_thresh)

        self.strong_edges = strong_edges
        self.weak_edges = weak_edges

        return self.weak_edges, self.strong_edges


    def threshold_hysteresis(self, max_iter=100):
        """
        Iteratively perform threshold hysteresis on the weak and strong edges. This is accomplished
        by setting all weak edges connected to strong edges to strong, and then repeating the
        process until no more weak edges are connected to strong edges. The final Canny-detected
        edges are returned.

        The max_iter parameter is used to prevent infinite loops in the case of a bad threshold
        selection. The default is 100 iterations, but can be set to any positive integer.
        """
        if self.strong_edges is None or self.weak_edges is None:
            self.double_threshold()

        strong_weak = np.zeros(self.strong_edges.shape)
        strong_weak[self.strong_edges] = 255.0
        strong_weak[self.weak_edges] = 100.0

        found_strong = True

        while found_strong and max_iter > 0:
            found_strong = False
            max_iter -= 1

            for i in range(1, strong_weak.shape[0]-1):
                for j in range(1, strong_weak.shape[1]-1):
                    if strong_weak[i, j] == 255.0:
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                if strong_weak[i+k, j+l] == 100.0:
                                    strong_weak[i+k, j+l] = 255.0
                                    found_strong = True

        # set all weak edges to 0
        strong_weak[strong_weak == 100.0] = 0
        self.canny_edges = strong_weak

        return self.canny_edges
    
    
    def detect_edges(self):
        self.smooth_image()
        self.calculate_gradient()
        self.gradient_magnitude()
        self.gradient_direction()
        self.non_max_suppression()
        self.double_threshold()
        self.threshold_hysteresis()

        return self.canny_edges
    
    def set_image(self, image: np.array):
        self.image = image.astype(np.float32)
    
    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size
    
    def set_kernel_type(self, kernel_type: GradientKernels):
        self.kernel_type = kernel_type

    def set_neighbor_depth(self, neighbor_depth):
        self.neighbor_depth = neighbor_depth

    def set_pixel_connectivity(self, pixel_connectivity: PixelConnectivity):
        self.pixel_connectivity = pixel_connectivity

    def set_low_thresh(self, low_thresh):
        self.low_thresh = low_thresh

    def set_high_thresh(self, high_thresh):
        self.high_thresh = high_thresh

    def set_max_hysteresis_iterations(self, max_hysteresis_iterations):
        self.max_hysteresis_iterations = max_hysteresis_iterations
    
    def get_raw_image(self):
        return self.image.astype(np.uint8)
    
    def get_smoothed_image(self):
        return self.smoothed_image.astype(np.uint8)
    
    def get_gradient_x(self):
        return (((self.grad_x - self.grad_x.min()) * 255.0 / (self.grad_x.max() - self.grad_x.min())).astype(np.uint8))
    
    def get_gradient_y(self):
        return (((self.grad_y - self.grad_y.min()) * 255.0 / (self.grad_y.max() - self.grad_y.min())).astype(np.uint8))
    
    def get_magnitude(self):
        return (((self.magnitude - self.magnitude.min()) * 255.0 / (self.magnitude.max() - self.magnitude.min())).astype(np.uint8))
    
    def get_direction(self):
        return (((self.direction - self.direction.min()) * 255.0 / (self.direction.max() - self.direction.min())).astype(np.uint8))
    
    def get_nms(self):
        return (((self.nms - self.nms.min()) * 255.0 / (self.nms.max() - self.nms.min())).astype(np.uint8))
    
    def get_weak_edges(self):
        tmp = np.zeros(self.weak_edges.shape)
        tmp[self.weak_edges] = 255.0
        return tmp.astype(np.uint8)
    
    def get_strong_edges(self):
        tmp = np.zeros(self.strong_edges.shape)
        tmp[self.strong_edges] = 255.0
        return tmp.astype(np.uint8)
    
    def get_canny_edges(self):
        return (((self.canny_edges - self.canny_edges.min()) * 255.0 / (self.canny_edges.max() - self.canny_edges.min())).astype(np.uint8))
    