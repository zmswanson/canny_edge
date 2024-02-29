import numpy as np
import scipy as sp
import enum
import time

class GradientKernels(enum.Enum):
    SOBEL = 1
    PREWITT = 2
    SCHARR = 3

class PaddingTypes(enum.Enum):
    CONSTANT = 1
    EDGE = 2
    REFLECT = 3
    SYMMETRIC = 4

class PixelConnectivity(enum.Enum):
    FOUR_CONNECTED = 1
    EIGHT_CONNECTED = 2

class CannyEdgeDetector:
    def __init__(self, image: np.array, sigma=1, kernel_size=5, kernel_type=GradientKernels.SOBEL, low_thresh=0.05, high_thresh=0.15):
        if len(image.shape) != 2:
            raise ValueError("Image must be a 2D array (grayscale)")
        
        if image.dtype == np.uint8:
            self.image = image.astype(np.float32)
        elif max(image.flatten()) > 1:
            if max(image.flatten()) > 255:
                self.image = image.astype(np.float32) / max(image.flatten())
            else:
                self.image = image.astype(np.float32) / 255.0

        print(f"Image shape: {self.image.shape} ... Image dtype: {self.image.dtype}")

        self.sigma = sigma
        self.kernel_size = kernel_size

        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        elif 5 < self.kernel_size < 3:
            raise ValueError("Kernel size must be 3 or 5") 

        self.kernel_type = kernel_type

        if self.kernel_type == GradientKernels.SOBEL:
            self.kern_x, self.kern_y = self.sobel_filter()
        elif self.kernel_type == GradientKernels.PREWITT:
            self.kern_x, self.kern_y = self.prewitt_filter()
        elif self.kernel_type == GradientKernels.SCHARR:
            self.kern_x, self.kern_y = self.scharr_filter()
        else:
            raise ValueError("Invalid kernel type")
        
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
        half_size = int(size) // 2
        x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        kern_gauss =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        kern_gauss = kern_gauss / kern_gauss.sum()
        
        return kern_gauss
    

    def sobel_filter(self):
        kern_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kern_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

        kern_x = kern_x / np.sum(np.abs(kern_x))
        kern_y = kern_y / np.sum(np.abs(kern_y))

        return (kern_x, kern_y)
    
    def prewitt_filter(self):
        kern_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        kern_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)

        kern_x = kern_x / np.sum(np.abs(kern_x))
        kern_y = kern_y / np.sum(np.abs(kern_y))

        return (kern_x, kern_y)

    def scharr_filter(self):
        kern_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32)
        kern_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], np.float32)

        kern_x = kern_x / np.sum(np.abs(kern_x))
        kern_y = kern_y / np.sum(np.abs(kern_y))

        return (kern_x, kern_y)
    
    def apply_padding(self, image, pad_type=PaddingTypes.CONSTANT):
        if pad_type == PaddingTypes.CONSTANT:
            return np.pad(image, pad_width=self.kernel_size//2, mode='constant', constant_values=0)
        elif pad_type == PaddingTypes.EDGE:
            return np.pad(image, pad_width=self.kernel_size//2, mode='edge')
        elif pad_type == PaddingTypes.REFLECT:
            return np.pad(image, pad_width=self.kernel_size//2, mode='reflect')
        elif pad_type == PaddingTypes.SYMMETRIC:
            return np.pad(image, pad_width=self.kernel_size//2, mode='symmetric')
        else:
            raise ValueError("Invalid padding type")

    def smooth_image(self):
        if self.image is None:
            raise ValueError("Image not found")
        
        start_time = time.time_ns() / 1e6 # Convert to milliseconds

        kernel = self.gaussian_kernel(self.kernel_size, self.sigma)
        padded_image = np.pad(self.image, pad_width=self.kernel_size//2, mode='constant', constant_values=0)
        smoothed_image = np.zeros(self.image.shape)

        for i in range(0, padded_image.shape[0]-self.kernel_size+1):
            for j in range(0, padded_image.shape[1]-self.kernel_size+1):
                smoothed_image[i, j] = np.sum(np.multiply(padded_image[i:i+self.kernel_size, j:j+self.kernel_size], kernel))

        self.smoothed_image = smoothed_image

        print(f"Smoothed image in {time.time_ns() / 1e6 - start_time} seconds")

        return smoothed_image


    def calculate_gradient(self):
        if self.smoothed_image is None:
            self.smoothed_image = self.smooth_image()

        if self.kern_x is None or self.kern_y is None:
            if self.kernel_type == GradientKernels.SOBEL:
                self.kern_x, self.kern_y = self.sobel_filter()
            elif self.kernel_type == GradientKernels.PREWITT:
                self.kern_x, self.kern_y = self.prewitt_filter()
            elif self.kernel_type == GradientKernels.SCHARR:
                self.kern_x, self.kern_y = self.scharr_filter()

        start_time = time.time_ns() / 1e6

        padded_image = np.pad(self.smoothed_image, pad_width=1, mode='constant', constant_values=0)

        grad_x = np.zeros(self.smoothed_image.shape)
        grad_y = np.zeros(self.smoothed_image.shape)
        
        for i in range(0, padded_image.shape[0]-3+1):
            for j in range(0, padded_image.shape[1]-3+1):
                grad_x[i, j] = np.sum(np.multiply(padded_image[i:i+3, j:j+3], self.kern_x))
                grad_y[i, j] = np.sum(np.multiply(padded_image[i:i+3, j:j+3], self.kern_y))

        print(f"Gradient X: {grad_x.shape} ... Gradient Y: {grad_y.shape}")

        self.grad_x = grad_x
        self.grad_y = grad_y

        print(f"Calculated gradient in {time.time_ns() / 1e6 - start_time} ms")

        return grad_x, grad_y
        
    
    def gradient_magnitude(self):
        if self.grad_x is None or self.grad_y is None:
            self.grad_x, self.grad_y = self.calculate_gradient()

        start_time = time.time_ns() / 1e6

        self.magnitude = np.sqrt(np.square(self.grad_x) + np.square(self.grad_y))

        print(f"Calculated gradient magnitude in {time.time_ns() / 1e6 - start_time} ms")

        return self.magnitude
    
    def gradient_direction(self):
        if self.grad_x is None or self.grad_y is None:
            self.grad_x, self.grad_y = self.calculate_gradient()

        start_time = time.time_ns() / 1e6

        self.direction = np.arctan2(self.grad_y, self.grad_x)

        print(f"Calculated gradient direction in {time.time_ns() / 1e6 - start_time} ms")

        return self.direction
    
    def non_max_suppression(self, pixel_connectivity=PixelConnectivity.EIGHT_CONNECTED):
        if self.magnitude is None:
            self.magnitude = self.gradient_magnitude()

        if self.direction is None:
            self.direction = self.gradient_direction()

        tmp_nms_image = np.zeros(self.magnitude.shape)
        angle = self.direction * 180. / np.pi
        angle[angle < 0] += 180

        if pixel_connectivity == PixelConnectivity.FOUR_CONNECTED:
            for i in range(1, self.magnitude.shape[0]-1):
                for j in range(1, self.magnitude.shape[1]-1):
                    if (0 <= angle[i,j] < 45) or (135 <= angle[i,j] <= 180):
                        next = self.magnitude[i, j+1]
                        prev = self.magnitude[i, j-1]
                    elif (45 <= angle[i,j] < 135):
                        next = self.magnitude[i+1, j]
                        prev = self.magnitude[i-1, j]

                    if self.magnitude[i,j] >= max(next, prev):
                        tmp_nms_image[i,j] = self.magnitude[i,j]
                    else:
                        tmp_nms_image[i,j] = 0
        else:
            for i in range(1, self.magnitude.shape[0]-1):
                for j in range(1, self.magnitude.shape[1]-1):
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        next = self.magnitude[i, j+1]
                        prev = self.magnitude[i, j-1]
                    elif (22.5 <= angle[i,j] < 67.5):
                        next = self.magnitude[i+1, j+1]
                        prev = self.magnitude[i-1, j-1]
                    elif (67.5 <= angle[i,j] < 112.5):
                        next = self.magnitude[i+1, j]
                        prev = self.magnitude[i-1, j]

                    elif (112.5 <= angle[i,j] < 157.5):
                        next = self.magnitude[i-1, j+1]
                        prev = self.magnitude[i+1, j-1]


                    if self.magnitude[i,j] >= max(next, prev):
                        tmp_nms_image[i,j] = self.magnitude[i,j]
                    else:
                        tmp_nms_image[i,j] = 0


        self.nms = tmp_nms_image

        return self.nms
    
    # def non_max_suppression(self, neighbor_depth=1):
    #     if self.magnitude is None:
    #         self.magnitude = self.gradient_magnitude()

    #     if self.direction is None:
    #         self.direction = self.gradient_direction()

    #     start_time = time.time_ns() / 1e6

    #     tmp_nms_image = np.zeros(self.magnitude.shape)
    #     angle = self.direction * 180. / np.pi
    #     angle[angle < 0] += 180

    #     for i in range(neighbor_depth, self.magnitude.shape[0]-neighbor_depth):
    #         for j in range(neighbor_depth, self.magnitude.shape[1]-neighbor_depth):
    #             if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
    #                 next = [self.magnitude[i, j+k] for k in range(1, neighbor_depth+1)]
    #                 prev = [self.magnitude[i, j-k] for k in range(1, neighbor_depth+1)]
    #             elif (22.5 <= angle[i,j] < 67.5):
    #                 next = [self.magnitude[i+k, j-k] for k in range(1, neighbor_depth+1)]
    #                 prev = [self.magnitude[i-k, j+k] for k in range(1, neighbor_depth+1)]
    #             elif (67.5 <= angle[i,j] < 112.5):
    #                 next = [self.magnitude[i+k, j] for k in range(1, neighbor_depth+1)]
    #                 prev = [self.magnitude[i-k, j] for k in range(1, neighbor_depth+1)]
    #             elif (112.5 <= angle[i,j] < 157.5):
    #                 next = [self.magnitude[i-k, j-k] for k in range(1, neighbor_depth+1)]
    #                 prev = [self.magnitude[i+k, j+k] for k in range(1, neighbor_depth+1)]

    #             if self.magnitude[i,j] >= max(next + prev):
    #                 tmp_nms_image[i,j] = self.magnitude[i,j]
    #             else:
    #                 tmp_nms_image[i,j] = 0

    #     self.nms = tmp_nms_image

    #     print(f"Performed non-maximum suppression in {time.time_ns() / 1e6 - start_time} ms")

    #     return self.nms
    
    def double_threshold(self):
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
        if self.strong_edges is None or self.weak_edges is None:
            self.double_threshold()

        start_time = time.time_ns() / 1e6

        strong_weak = np.zeros(self.strong_edges.shape)
        strong_weak[self.strong_edges] = 255.0
        strong_weak[self.weak_edges] = 100.0

        found_strong = True

        while found_strong and max_iter > 0:
            print(f"Iteration {max_iter}")
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

        print(f"Performed threshold hysteresis in {time.time_ns() / 1e6 - start_time} ms")

        return self.canny_edges
                            


        # strong_x, strong_y = np.where(self.strong_edges)
        # weak_x, weak_y = np.where(self.weak_edges)

        # # Copy strong edges to canny_edges array
        # self.canny_edges = np.copy(self.strong_edges)        
        
        # # Iterate through the weak edges and strong edges to see if any weak edges are connected to
        # # strong edges. If they are, they are also considered strong edges. Repeat until no more
        # # weak edges are promoted to strong edges.
        # promotion_found = True
        # promotion_count = 0

        # while promotion_found and max_iter > 0:
        #     max_iter -= 1
        #     print(f"Promotion found {promotion_count} times.")
        #     print(f"Stong edges: {strong_x.shape[0]}")
        #     print(f"Weak edges: {weak_x.shape[0]}")
        #     print(f"Canny edges: {self.canny_edges.shape}")
        #     promotion_found = False

        #     weak_len = weak_x.shape[0]
        #     strong_len = strong_x.shape[0]

        #     # print the 

        #     # Create a 2D array of differences between weak and strong coordinates
        #     # diff_matrix = np.abs(weak_x[:, np.newaxis] - strong_x) + np.abs(weak_y[:, np.newaxis] - strong_y)

        #     # # Find indices where the differences are less than or equal to 1
        #     # match_indices = np.argwhere(diff_matrix <= 1)

        #     # if match_indices.size > 0:
        #     #     # Get unique indices of weak edges connected to strong edges
        #     #     unique_weak_indices = np.unique(match_indices[:, 0])

        #     #     # Update strong_edges and weak_edges arrays
        #     #     self.canny_edges[weak_x[unique_weak_indices], weak_y[unique_weak_indices]] = True

        #     #     # Update strong_x, strong_y, weak_x, weak_y
        #     #     strong_x = np.concatenate((strong_x, weak_x[unique_weak_indices]))
        #     #     strong_y = np.concatenate((strong_y, weak_y[unique_weak_indices]))
        #     #     weak_x = np.delete(weak_x, unique_weak_indices)
        #     #     weak_y = np.delete(weak_y, unique_weak_indices)

        #     #     promotion_found = True
        #     #     promotion_count += 1

        #     # pop_indices = []

        #     # for i in range(weak_len):
        #     #     for j in range(strong_len):
        #     #         if abs(weak_x[i] - strong_x[j]) <= 1 and abs(weak_y[i] - strong_y[j]) <= 1:
        #     #             self.canny_edges[weak_x[i], weak_y[i]] = True
        #     #             # weak_edges[weak_x[i], weak_y[i]] = False
                        
        #     #             pop_indices.append(i)

        #     #             promotion_found = True
        #     #             promotion_count += 1
        #     #             break
            
        #     # strong_x = np.append(strong_x, weak_x[pop_indices])
        #     # strong_y = np.append(strong_y, weak_y[pop_indices])
        #     # weak_x = np.delete(weak_x, pop_indices)
        #     # weak_y = np.delete(weak_y, pop_indices)

        # print(f"Performed double threshold in {time.time_ns() / 1e6 - start_time} ms")

        # return self.canny_edges