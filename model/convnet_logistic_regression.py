# implement a logistic regression model in PyTorch. Note that a logistic regressor uses a linear transformation of the input.

# import library
import matplotlib.pyplot 
import numpy.random
import torch.utils.data
import torchvision
from torch import Tensor
from torch.nn import Module
import torch;
from data.mnist_dataset import train_loader, test_loader;

if torch.cuda.is_available():
      DEVICE = "cuda:0" # CUDA signifie « Compute Unified Device Architecture »
else:
  DEVICE = "cpu" # CPU « signifie Central Processing Unit »



# Logistic Regression Module
class LogisticRegression(Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
    
        ###########################################################################
        # TODO: Instantiate the layer here.                                       #
        ###########################################################################
        self.layer = torch.nn.Linear(input_size, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        ###########################################################################
        # TODO: Apply the layer to the input.                                     #
        ###########################################################################
        outputs = self.layer(x)
        return outputs
    
    

module = LogisticRegression(28 * 28, 10)

module = module.to(device=DEVICE)



###########################################################################
# TODO: Create criterion and optimizer here.                              #
###########################################################################

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.9)


# Train the model. If everything is correct, the loss should go below 0.45.
# We will use the following generic training loop for a PyTorch model.
EPOCHS = 5

# Exponential moving average of the loss:
ema = None

for epoch in range(EPOCHS):
  for batch_index, (train_images, train_targets) in enumerate(train_loader):
    train_images = train_images.view(-1, 28 * 28).requires_grad_().to(device=DEVICE)
    train_targets = train_targets.to(device=DEVICE)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward pass to get output/logits
    outputs = module(train_images)

    # Calculate Loss: softmax --> cross entropy loss
    loss = criterion(outputs, train_targets)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updates parameters:
    optimizer.step()

    # NOTE: It is important to call .item() on the loss before summing.
    if ema is None:
        ema = loss.item()
    else:
        ema += (loss.item() - ema) * 0.01

    if batch_index % 500 == 0:
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_index * len(train_images),
                len(train_loader.dataset),
                100.0 * batch_index / len(train_loader),
                ema,
            ),
        )
        
        
 #Evaluation
# Use the following function to measure the test accuracy of your trained model. 
       
correct_predictions = 0
predictions = 0

# Iterate through test dataset
for test_images, test_targets in test_loader:
    test_images = test_images.view(-1, 28 * 28)

    # Forward pass only to get logits/output
    outputs = module(test_images)

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)

    predictions += test_targets.size(0)

    if torch.cuda.is_available():
        correct_predictions += (predicted.cpu() == test_targets.cpu()).sum()
    else:
        correct_predictions += (predicted == test_targets).sum()

print(correct_predictions.item() / predictions)