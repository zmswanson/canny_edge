# %%
from PIL import Image, ImageDraw
import numpy as np
import random

# %%
def generate_binary_image(size, shapes):
    """
    Use PIL to create a new image with the given size and shapes. This generated binary image will
    have a black background and white boundaries for the shapes. This will serve as the ground truth
    for the Canny Edge Detection algorithm.

    Note: The shapes are filled with black color (0) like the background to ensure that any
    occlusions that may occur when the shapes are filled are not considered as ground truth edges.
    """
    image = Image.new('1', size, color=0)
    draw = ImageDraw.Draw(image)

    for shape in shapes:
        if shape['type'] == 'circle':
            draw.ellipse(shape['bbox'], outline=255, fill=0)
        elif shape['type'] == 'square':
            draw.rectangle(shape['bbox'], outline=255, fill=0)
        elif shape['type'] == 'random_polygon':
            draw.polygon(shape['points'], outline=255, fill=0, width=1)

    return image

def generate_grayscale_image(size, shapes):
    """
    Generates a grayscale image with the given size and shapes. This should be the same as the
    binary image but with the shapes filled with a random grayscale color.
    """
    image = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(image)

    for shape in shapes:
        # Generate a random grayscale color
        random_color = random.randint(25, 255)

        if shape['type'] == 'circle':
            draw.ellipse(shape['bbox'], fill=random_color)
        elif shape['type'] == 'square':
            draw.rectangle(shape['bbox'], fill=random_color)
        elif shape['type'] == 'random_polygon':
            draw.polygon(shape['points'], fill=random_color)

    return image

def add_noise(image, noise_level):
    """
    Adds Gaussian white noise to the given image with the given noise level.
    """
    img_array = np.array(image)
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy_image = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
    return noisy_image

def generate_shapes(num_shapes, size):
    """
    Generates a list of shapes with random parameters. The shapes can be circles, squares, or random
    polygons. The parameters for each shape are also randomly generated.
    """
    shapes = []
    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'square', 'random_polygon'])
        if shape_type == 'circle':
            bbox = [random.randint(0, size[0]), random.randint(0, size[1]),
                    random.randint(size[0] // 4, size[0]),
                    random.randint(size[1] // 4, size[1])]
            # make sure x1 > x0 and y1 > y0, switch if necessary
            bbox[0], bbox[2] = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
            bbox[1], bbox[3] = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
            shapes.append({'type': shape_type, 'bbox': bbox})
        elif shape_type == 'square':
            bbox = [random.randint(0, size[0]), random.randint(0, size[1]),
                    random.randint(size[0] // 4, size[0]),
                    random.randint(size[1] // 4, size[1])]
            # make sure x1 > x0 and y1 > y0, switch if necessary
            bbox[0], bbox[2] = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
            bbox[1], bbox[3] = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
            shapes.append({'type': shape_type, 'bbox': bbox})
        elif shape_type == 'random_polygon':
            points = [(random.randint(0, size[0]), random.randint(0, size[1])) for _ in range(5)]
            shapes.append({'type': shape_type, 'points': points})

    return shapes

def main():
    """
    Generates 50 pairs of binary and grayscale images with different levels of noise and saves them
    to the data/simulated/ directory. The binary images will serve as the ground truth for the Canny
    Edge Detection algorithm. The noisy grayscale images will be used to evaluate as a training and
    test set for finding the best parameters for the Canny Edge Detection algorithm.
    """
    size = (128, 128)
    num_images = 50

    for i in range(num_images):
        shapes = generate_shapes(random.randint(5, 15), size)

        # Generate binary images
        binary_image = generate_binary_image(size, shapes)
        binary_image.save(f'./data/simulated/binary_image_{i}.png')

        # Generate grayscale images
        grayscale_image = generate_grayscale_image(size, shapes)
        grayscale_image.save(f'./data/simulated/grayscale_image_{i}.png')

        # Generate images with small and large noise
        small_noise_image = add_noise(grayscale_image, 5)
        small_noise_image.save(f'./data/simulated/small_noise_image_{i}.png')

        large_noise_image = add_noise(grayscale_image, 20)
        large_noise_image.save(f'./data/simulated/noisy_image_{i}.png')

if __name__ == "__main__":
    main()