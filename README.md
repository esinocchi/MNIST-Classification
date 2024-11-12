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

### Code Structure

#### Model Architecture
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
### Data Loading and Preprocessing
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    MNIST(root='./data', train=True, transform=transform, download=True),
    batch_size=32, shuffle=True
)

test_loader = DataLoader(
    MNIST(root='./data', train=False, transform=transform, download=True),
    batch_size=32, shuffle=False
)
```
### Training Setup
```python
model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
### Training Loop
```python
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```
### Model Evuluation
```python
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')
```
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

  
