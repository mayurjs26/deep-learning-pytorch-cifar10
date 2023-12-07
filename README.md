CIFAR-10 Image Classification with CNN and Batch Normalization
This project implements a convolutional neural network (CNN) with batch normalization for image classification on the CIFAR-10 dataset. It utilizes PyTorch libraries and techniques for efficient training and evaluation.

Project Highlights:

Achieved a validation accuracy of 88.2% after 10 epochs of training.
Employed batch normalization for better performance and reduced overfitting.
Leveraged efficient PyTorch libraries for training and evaluation.
Installation:

This project requires the following libraries:

PyTorch
torchvision
numpy
matplotlib
Install the required libraries using pip:

pip install torch torchvision numpy matplotlib
Getting Started:

Download the CIFAR-10 dataset:
python -m datasets download cifar10
Run the script:
python main.py
Code Structure:

main.py: The main script that defines the model, training loop, and evaluation process.
ImageClassificationBase.py: A base class for image classification models with common training and validation steps.
CIFAR_CNN.py: The CNN model architecture specifically designed for the CIFAR-10 dataset.
utils.py: Utility functions for device allocation and data loading.
Project Contributions:

This project is a learning exercise for implementing CNNs with batch normalization for image classification.
It serves as a baseline model for further experimentation and improvement.
The code is well-structured and documented for easy understanding and modification.
Future Work:

Explore different CNN architectures and hyperparameters for improved performance.
Implement data augmentation techniques to increase training data diversity.
Evaluate the model on other image classification datasets.
