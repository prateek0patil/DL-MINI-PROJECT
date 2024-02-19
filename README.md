A case study on human hand written recognition using deep learning models

This mini project focuses on implementing a deep learning model for handwritten digit recognition using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## Dataset

The dataset used for this project is the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale and has a resolution of 28x28 pixels.

## Requirements

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras (>=2.0)
- NumPy
- Matplotlib
- Jupyter Notebook (for running the provided notebooks)

## Usage

1. Clone this repository:

git clone https://github.com/prateek0patil/DL-MINI-PROJECT

Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Run the notebook handwritten_digit_recognition.ipynb to train and evaluate the CNN and RNN models.

Follow the instructions within the notebook to load the dataset, preprocess the data, define the CNN and RNN architectures, train the models, and evaluate their performance.

Files
handwritten_digit_recognition.ipynb: Jupyter notebook containing the code for training and evaluating CNN and RNN models for handwritten digit recognition.
requirements.txt: List of Python dependencies required to run the code.
Results
After training the CNN and RNN models on the MNIST dataset, the following results were obtained:

CNN model accuracy on test set: 0.9853000044822693%
RNN model accuracy on test set: 0.9639000296592712%

References
LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/

TensorFlow documentation: https://www.tensorflow.org/api_docs

Keras documentation: https://keras.io/api/
