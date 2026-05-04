# MNIST Digit Classifier (PyTorch)

This is my first deep learning project where I built a simple neural network to classify handwritten digits using the MNIST dataset.

I implemented this using PyTorch and trained a basic Artificial Neural Network (ANN) from scratch. The model takes an image of a digit (0–9) and predicts which digit it is.

## What I did

* Loaded and processed the MNIST dataset using torchvision
* Built a neural network with:

  * Input layer: 784 neurons (28×28 image flattened)
  * Hidden layer: 128 neurons with ReLU
  * Output layer: 10 neurons (digits 0–9)
* Trained the model using CrossEntropyLoss and Adam optimizer
* Evaluated performance on test data
* Plotted training loss using matplotlib

## Results

The model achieves around ~90–95% accuracy on the MNIST dataset (can vary depending on training).

## What I learned

* How neural networks work (forward pass, backpropagation)
* Why activation functions like ReLU are important
* How loss functions like CrossEntropyLoss work
* Basics of training loops in PyTorch
* Difference between training and evaluation mode

## Tech stack

* Python
* PyTorch
* Torchvision
* Matplotlib

## Next step

I’ll be improving this by building a CNN model, since ANNs are not ideal for image data.

---
