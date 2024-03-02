import click
from canny_edge import CannyEdgeDetector, GradientKernels, PixelConnectivity
from os import listdir
from os.path import isfile, join
import cv2

@click.command(help="CLI for the Canny Edge Detection algorithm with various user-defined " +
               "parameters. The output directory for  processed image is data/output/, " +
               "unless otherwise specified with the --output_dir option. The image file to process " +
               "is ALL by default, which processes all images in the data/input/ directory. " +
               "Processed images are saved with the following naming convention: \n" +
               "<image_file>_g<gaussian_blur>_n<kernel_size>_k<kernel_type>_d<neighbor_depth>_p<pixel_connectivity>_u<high_threshold>_l<low_threshold>.png")
@click.option('-o', '--output_dir', default='data/output/', help='The output directory for the processed image.')
@click.option('-i', '--image_file', default='ALL', help='The image file to process, e.g. cameraman.tif. Assumes data/input/ directory if no path is provided and defaults to processing all images in that directory. Comma-separated list of image files is also accepted.')
@click.option('-g', '--gaussian_blur', default=3, help='The standard deviation of the Gaussian filter.')
@click.option('-n', '--kernel_size', default=3, help='The size of the kernel for the Gaussian kernel.')
@click.option('-k', '--kernel_type', default='SOBEL', help='The type of kernel to use for gradient calculation: SOBEL, PREWITT, SCHARR.')
@click.option('-d', '--neighbor_depth', default=1, help='The depth of neighbors to consider for non-maximum suppression.')
@click.option('-p', '--pixel_connectivity', default=8, help='The type of pixel connectivity to use for non-maximum suppression: 4 or 8.')
@click.option('-u', '--high_threshold', default=0.15, help='The high threshold for the double thresholding.')
@click.option('-l', '--low_threshold', default=0.09, help='The low threshold for the double thresholding.')
def main(output_dir, image_file, gaussian_blur, kernel_size, kernel_type, neighbor_depth, pixel_connectivity, high_threshold, low_threshold):
    """
    CLI for the Canny Edge Detection algorithm with various user-defined parameters.
    """
    if image_file == 'ALL':
        image_files = [f for f in listdir('data/input/') if isfile(join('data/input/', f))]
        # remove non-image files
        image_files = [
            f for f in image_files if f.endswith('.png') or f.endswith('.tif') or f.endswith('.bmp')
                or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.gif') or f.endswith('.tiff')
                or f.endswith('.webp') or f.endswith('.heif') or f.endswith('.heic') or f.endswith('.bpg')
                or f.endswith('.svg') or f.endswith('.eps')
        ]

    else:
        image_files = [img_file for img_file in image_file.split(',') if img_file != '']

    kernel_enum = GradientKernels.SOBEL
    pixel_enum = PixelConnectivity.EIGHT_CONNECTED

    if kernel_type not in ['SOBEL', 'PREWITT', 'SCHARR']:
        raise ValueError("Invalid kernel type. Please use SOBEL, PREWITT, or SCHARR.")
    elif kernel_type == 'PREWITT':
        kernel_enum = GradientKernels.PREWITT
    elif kernel_type == 'SCHARR':
        kernel_enum = GradientKernels.SCHARR

    if pixel_connectivity not in [4, 8]:
        raise ValueError("Invalid pixel connectivity. Please use 4 or 8.")
    elif pixel_connectivity == 4:
        pixel_enum = PixelConnectivity.FOUR_CONNECTED



    for img_f in image_files:
        print(f"Processing {img_f}...")
        image = cv2.imread(f"data/input/{img_f}", cv2.IMREAD_GRAYSCALE)
        canny = CannyEdgeDetector(image, sigma=gaussian_blur, kernel_size=kernel_size, 
                                  kernel_type=kernel_enum, low_thresh=low_threshold, 
                                  high_thresh=high_threshold, neighbor_depth=neighbor_depth, 
                                  pixel_connectivity=pixel_enum)
        edges = canny.detect_edges()
        # Append parameters to file name for easy reference
        out_path = output_dir + f"{img_f.split('.')[0]}_g{gaussian_blur}_n{kernel_size}_k{kernel_type}_d{neighbor_depth}_p{pixel_connectivity}_u{high_threshold}_l{low_threshold}.png"
        cv2.imwrite(out_path, edges)
        print(f"Processed image saved to {out_path}.")
        print()

if __name__ == '__main__':
    main()