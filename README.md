# Canny Edge Detector

A Python-implemented Canny edge detector for ECEN-898: Introduction to Computer Vision at the
University of Nebraska-Lincoln (Spring 2024).

## Features

- A CannyEdgeDetector class for easy use in other Python applications.
- A simple command line interface (CLI) for processing individual or multiple images.
- A simple browser-based user interface for uploading images and playing around with parameters.
- A synthetic image generator for ground truth and image data.

## Getting Started

### Prerequisites

- Python 3.x installed

### Installation

1. Clone the repository or download and unzip the zipped source file to desktop
2. (Optional) Create a virtual environment for the 
3. Install dependencies that apply to your use-case:

    a. Command-line interface: `pip install -r ./requirements/main_requirements.txt`
    
    b. Web interface: `pip install -r ./requirements/gui_requirements.txt`

    c. Image synthesis/parameter optimization: 
    `pip install -r ./requirements/experiment_requirements.txt`

4. Copy image files of interest to *data/input/*

## Using the Application

### Command Line Interface

Execute `python3 main.py`. This will process all the image in *data/input/* with the default
parameters and put the generated edge images in *data/output/*. Run `python3 main.py --help` for
more details on the available options.

### Web User Interface

Execute `python3 gui_main.py` and navigate to http://127.0.0.1:5000 in any browser of you choice.
Confirmed to work with Google Chrome.

### Image Synthesis

Execute `python3 image_simulator.py`. This will generate 50 128x128 grayscale images in 
*data/simulated/*. Edit the image_simulator.py script if you want less/more images or different
size.

### Parameter Optimization

Execute `python3 parameter_optimizer.py`. Edit the python script if you want to try a different
set of parameters.

## Assignment Submission Notes

* The assignment write-up is provided in the base directory. See **swanson_python1_writeup.pdf**.

* The processed images presented in the writeup may be found in */processed_examples/output/*.

* The example images are already provided in *data/input/*.
