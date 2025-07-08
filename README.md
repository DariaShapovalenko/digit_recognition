

# Digit Recognition Neural Network

This project implements a fully connected feedforward neural network for digit recognition (0–9).
The model is trained on the MNIST dataset (provided as a CSV file).

## How It Works

* The network learns to recognize handwritten digits (0 to 9)
* Each digit image is a vector of pixel values, normalized between 0 and 1
* The model is a deep neural network with 3 hidden layers
* Uses ReLU activation in hidden layers and softmax in the output
* Training is done via backpropagation with gradient descent


## Network Architecture

This is a fully connected feedforward network:

* Input layer: size = number of pixels (e.g. 784 for 28×28 images)
* Hidden layer 1: 128 neurons (ReLU)
* Hidden layer 2: 64 neurons (ReLU)
* Output layer: 10 neurons (one per digit, softmax activation)


# File Overview

### `neural_net.py`

* Contains the main NeuralNetwork class
* Implements forward pass, backpropagation, activation functions, training logic
* Includes `.save()` and `.load()` to save/load models as `.json`

### `main.py`

* Loads and normalizes MNIST data from `data.csv`
* Splits data into train/test sets
* Trains the neural network
* Evaluates accuracy on the test set
* Saves the trained model as `digit_model.json`
* Prints predictions for a few test examples

### `digit_recognition.py`

* A simple GUI built with Tkinter
* Lets you draw digits using your mouse
* Predicts the drawn digit using the trained model
* Shows the result in the interface


### Results

* After training and testing, prediction results and logs are saved in the `results/` folder
* You can check this folder to review output examples and performance of the model


## Example Output (from `main.py`)

```
Epoch 1/20 completed
...
Epoch 20/20 completed

Accuracy: 91.35%

Predictions:
example 1: real = 7, predicted = 7, confidence = 0.9923
example 2: real = 0, predicted = 0, confidence = 0.9845
```


### Requirements

* Python 3.x
* NumPy
* Matplotlib (optional, for plotting)
* Tkinter (for GUI)
  

### Running the Project

1. Place your MNIST CSV file as `data.csv`
2. Run the training:

```bash
python main.py
```

3. Run the drawing interface:

```bash
python digit_recognition.py
```


