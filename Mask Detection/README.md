
# Real-Time Mask Detection with CNN and OpenCV

## Overview

This project is a real-time mask detection system that uses a Convolutional Neural Network (CNN) and OpenCV to identify whether a person is wearing a mask or not. It can process live video feeds from a webcam, making it a valuable tool for applications such as health and safety monitoring during the COVID-19 pandemic.

**Key Features:**

- Real-time mask detection in webcam video feeds.
- Pre-trained CNN model for mask detection.
- Easy-to-use, with the potential for customization.

## Getting Started

To get started with the mask detection system, follow these steps:

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.x
- Required Python libraries (see [requirements.txt](requirements.txt))
- Haar Cascade classifier XML file for face detection (included or downloadable)

### Installation

1. Clone the repository to your local machine:

   ``'shell
   git clone https://github.com/your-username/mask-detection.git
   cd mask-detection

### Install the required packages
```shell
pip install -r requirements.txt


### Usage
To run the real-time mask detection programme with your webcam, execute the following command:
```shell
   python mask_detection.py

###This command will open a window displaying your webcam feed with real-time mask detection results. By default, the pre-trained CNN model provided in the project will be used.

###Project Structure
The project directory is organised as follows:

   mask_detection.py: The main script for real-time mask detection.
   requirements.txt: A list of required Python packages.
   haarcascade_frontalface_default.xml: The Haar Cascade classifier XML file for face detection (included or downloadable).
   models/: Directory for saving or loading CNN models.
   data/: Directory for your training dataset (if you wish to train your own model).
   test/: Directory for images used for testing mask detection.
Training Your Own Model
If you want to train your own mask detection model on a custom dataset, follow these steps:

Prepare your labeled dataset with two classes: "with_mask" and "without_mask."

Organize your dataset in the data/ directory with subdirectories for each class.

Use the provided training script (train_mask_detector.py) to train your model. Customize the script as needed, including model architecture, hyperparameters, and data preprocessing.

Save your trained model in the models/ directory.

Modify the mask_detection.py script to load your trained model for real-time detection.


   
