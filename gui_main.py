from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

from canny_edge import CannyEdgeDetector, GradientKernels, PixelConnectivity

app = Flask(__name__)

# Global variable to store the image to allow for re-processing with different parameters
image_cache = None


def encode_images(images):
    """
    Converts a list of images to base64 format to be returned to the client and rendered in the browser.
    """
    result_base64_list = []
    for image in images:
        result = cv2.imencode('.png', image)[1].tobytes()
        result_base64 = base64.b64encode(result).decode('utf-8')
        result_base64_list.append(result_base64)
    return result_base64_list

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles GET and POST requests for the server's index page (i.e. the only page in this case).
    """
    global image_cache

    settings = {
        "gaussian_blur": 5,
        "kernel_size": 5,
        "kernel_type": "SOBEL",
        "neighbor_depth": 1,
        "pixel_connectivity": "EIGHT_CONNECTED",  # "FOUR_CONNECTED" or "EIGHT_CONNECTED
        "high_threshold": 0.27,
        "low_threshold": 0.09,
        "max_hysteresis_iterations": 100,
    }

    if request.method == 'POST':
        # Get form data from the POST request... this is important to get the parameters for the
        # Canny edge detector
        gaussian_blur = int(request.form['gaussian_blur'])
        kernel_size = int(request.form['kernel_size'])
        kernel_type = request.form['kernel_type']
        neighbor_depth = int(request.form['neighbor_depth'])
        pixel_connectivity = request.form['pixel_connectivity']
        high_threshold = float(request.form['high_threshold'])
        low_threshold = float(request.form['low_threshold'])
        max_hysteresis_iterations = int(request.form['max_hysteresis_iterations'])

        kernel_enum = GradientKernels.SOBEL
        pixel_enum = PixelConnectivity.EIGHT_CONNECTED

        if kernel_type == "PREWITT":
            kernel_enum = GradientKernels.PREWITT
        elif kernel_type == "SCHARR":
            kernel_enum = GradientKernels.SCHARR

        if pixel_connectivity == "FOUR_CONNECTED":
            pixel_enum = PixelConnectivity.FOUR_CONNECTED
        elif pixel_connectivity == "EIGHT_CONNECTED":
            pixel_enum = PixelConnectivity.EIGHT_CONNECTED
            
        settings = {
            "gaussian_blur": gaussian_blur,
            "kernel_size": kernel_size,
            "kernel_type": kernel_type,
            "neighbor_depth": neighbor_depth,
            "pixel_connectivity": pixel_connectivity,  # "FOUR_CONNECTED" or "EIGHT_CONNECTED
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "max_hysteresis_iterations": max_hysteresis_iterations
        }

        image = None

        # Check if the post request has the file part, if not then use the cached image... this
        # allows for re-processing the image with different parameters
        if 'image_file' not in request.files or request.files['image_file'].filename == '':
            if image_cache is not None:
                image = image_cache
            else:
                return render_template('index.html', settings=settings)
        else:
            image_file = request.files['image_file']
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            image_cache = image

        intermediate_images = [image]
        canny = CannyEdgeDetector(
            image, sigma=gaussian_blur, kernel_size=kernel_size, kernel_type=kernel_enum,
            neighbor_depth=neighbor_depth, pixel_connectivity=pixel_enum,
            low_thresh=low_threshold, high_thresh=high_threshold
        )
        canny.detect_edges()

        # We want to provide the user with a series of intermediate images to show the process of
        # the Canny edge detection algorithm
        intermediate_images.append(canny.get_smoothed_image())
        intermediate_images.append(canny.get_canny_edges())
        intermediate_images.append(canny.get_gradient_x())
        intermediate_images.append(canny.get_gradient_y())
        intermediate_images.append(canny.get_magnitude())
        intermediate_images.append(canny.get_nms())
        intermediate_images.append(canny.get_strong_edges())
        intermediate_images.append(canny.get_weak_edges())

        # Encode the intermediate images to base64
        # Specify labels for each image
        labels = ["Original", "Smoothed", "Canny Edges",
                  "Gradient X", "Gradient Y", "Gradient Magnitude", 
                  "Non-max Suppression", "Strong Edges", "Weak Edges"]
        
        # Combine the result_images with labels
        result_images_with_labels = list(zip(encode_images(intermediate_images), labels))

        return render_template('index.html', result_images=result_images_with_labels, settings=settings)

    return render_template('index.html', settings=settings)

if __name__ == '__main__':
    app.run(debug=True)
