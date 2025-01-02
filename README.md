# MNIST-Classification

This repository contains a neural network model implemented in Python for classifying handwritten digits from the MNIST dataset. The model is developed using the PyTorch framework and is presented in a Jupyter Notebook for ease of understanding and experimentation.

## Contents
- `net.ipynb`: Jupyter Notebook detailing the model architecture, training process, and evaluation metrics.
- `LICENSE`: MIT License under which this project is distributed.

## Features
- **Data Loading**: Utilizes PyTorch's `torchvision` library to load and preprocess the MNIST dataset.
- **Model Architecture**: Implements a CNN with layers optimized for digit recognition tasks.
- **Training**: Includes code for training the model with appropriate loss functions and optimizers.
- **Evaluation**: Provides methods to assess the model's performance on test data.

## Model Training Details
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Number of Epochs**: 10

## Network Architecture
- **Input Layer**: 28x28 grayscale images
- **Convolutional Layer 1**: 32 filters (3x3)
- **MaxPooling Layer 1**: 2x2
- **Convolutional Layer 2**: 64 filters (3x3)
- **MaxPooling Layer 2**: 2x2
- **Fully Connected Layer 1**: 1600 → 128
- **Fully Connected Layer 2**: 128 → 10 (output)

### Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Jupyter Notebook
- MatPlotLib
### Installation
1. Clone the repository <br>
`git clone https://github.com/esinocchi/MNIST-Classification.git`
2. Install the recquried packages: <br>
`pip install torch torchvision numpy matplotlib jupyter`
3. Run the Jupyter Notebook: <br>
`jupyter notebook net.ipynb`

  
