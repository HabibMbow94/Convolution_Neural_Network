import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import skimage.transform
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from IPython import display
from torchvision import datasets, transforms
from data.mnist_dataset import train_loader, test_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Convolutional Neural Network (CNN)

# The code below may be helpful in visualizing PyTorch tensors as images.

def train(model, criterion, data_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model."""

    # Make sure model is in training mode.
    model.train()

    # Move model to the device (CPU or GPU).
    model.to(device)

    # Exponential moving average of the loss.
    ema_loss = None

    # Loop over epochs.
    for epoch in range(num_epochs):

        # Loop over data.
        for batch_idx, (data, target) in enumerate(data_loader):

            # Forward pass.
            output = model(data.to(device))
            loss = criterion(output.to(device), target.to(device))

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01

        # Print out progress the end of epoch.
        print(
            "Train Epoch: {} \tLoss: {:.6f}".format(epoch, ema_loss),
        )


def test(model, data_loader):
    """Measures the accuracy of a model on a data set."""
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0

    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():

        # Loop over test data.
        for data, target in data_loader:

            # Forward pass.
            output = model(data.to(device))

            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)

            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()

    # Print test accuracy.
    percent = 100.0 * correct / len(data_loader.dataset)
    print(f"Accuracy: {correct} / {len(data_loader.dataset)} ({percent:.0f}%)")
    return percent


# In the last tutorial, you implemented a naive convolution. In this section you will implement your own version of forward pass of nn.Conv2d without using any of PyTorch's (or numpy's) pre-defined convolutional functions.


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive Python implementation of a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e., equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    Returns an array.
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    out = None

    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Check dimensions.
    assert (W + 2 * pad - filter_width) % stride == 0, "width does not work"
    assert (H + 2 * pad - filter_height) % stride == 0, "height does not work"

    ###########################################################################
    # TODO: Implement the forward pass of a convolutional layer without using #
    #       nn.Conv2D or other implementations of convolutions. Instead, use  #
    #       standard for- and while-loops to iterate over the tensors.        #
    #                                                                         #
    # Hint: you can use the function torch.nn.functional.pad for padding.     #
    ###########################################################################
    H_out = int(((H + 2 * pad - filter_height) / stride) + 1)
    W_out = int(((W + 2 * pad - filter_width) / stride) + 1)
    out = torch.empty(N, num_filters,H_out, W_out)
    
    pad = (pad, pad, pad, pad)
    x = torch.nn.functional.pad(x, pad, mode='constant', value=0.0)
    for n in range(N):
      for f in range(num_filters):
        for h in range(H_out):
          for j in range(W_out):
            out[n, f, h,j] = ((x[n,:,h*stride:h*stride+filter_height, j*stride:j*stride+filter_width])*w[f]).sum()+b[f]
            
    return out


# You can test your implementation by running the following testing code:

# Make convolution module.
w_shape = (3, 3, 4, 4)
w = torch.linspace(-0.2, 0.3, steps=torch.prod(torch.tensor(w_shape))).reshape(w_shape)
b = torch.linspace(-0.1, 0.2, steps=3)

# Compute output of module and compare against reference values.
x_shape = (2, 3, 4, 4)
x = torch.linspace(-0.1, 0.5, steps=torch.prod(torch.tensor(x_shape))).reshape(x_shape)
out = conv_forward_naive(x, w, b, {"stride": 2, "pad": 1})
out



# Make convolution module.
w_shape = (3, 3, 4, 4)
w = torch.linspace(-0.2, 0.3, steps=torch.prod(torch.tensor(w_shape))).reshape(w_shape)
b = torch.linspace(-0.1, 0.2, steps=3)

# Compute output of module and compare against reference values.
x_shape = (2, 3, 4, 4)
x = torch.linspace(-0.1, 0.5, steps=torch.prod(torch.tensor(x_shape))).reshape(x_shape)
out = conv_forward_naive(x, w, b, {"stride": 2, "pad": 1})

correct_out = torch.tensor(
    [
        [
            [[-0.08759809, -0.10987781], [-0.18387192, -0.2109216]],
            [[0.21027089, 0.21661097], [0.22847626, 0.23004637]],
            [[0.50813986, 0.54309974], [0.64082444, 0.67101435]],
        ],
        [
            [[-0.98053589, -1.03143541], [-1.19128892, -1.24695841]],
            [[0.69108355, 0.66880383], [0.59480972, 0.56776003]],
            [[2.36270298, 2.36904306], [2.38090835, 2.38247847]],
        ],
    ]
)

# Compare your output to ours; difference should be around e-8
print("Testing conv_forward_naive")
rel_error = ((out - correct_out) / (out + correct_out + 1e-6)).mean()
print("difference: ", rel_error)
if abs(rel_error) < 1e-6:
    print("Nice work! Your implementation of a convolution layer works correctly.")
else:
    print(
        "Something is wrong. The output was expected to be {} but it was {}".format(
            correct_out, out
        )
    )
    
    
    
#  We will now replace the logistic regressor from the last tutorial by a small convolutional network with two convolutional layers and a linear layer, and ReLU 
# activations in between the layers. Implement the model and use the same functions as before to train and test the convolutional network.


class ConvolutionalNetwork(nn.Module):
    """Simple convolutional network."""

    def __init__(self, image_side_size, num_classes, in_channels=1):
        super(ConvolutionalNetwork, self).__init__()

        # Fill these in:
        ##########################################################################
        # TODO: Implement a convulutional and a linear part.                     #
        # Hint: see forward() to understand how they should work together.       #
        ##########################################################################
        self.conv_network = nn.Sequential(
          nn.Conv2d(in_channels,4,4,1,3),
          nn.ReLU(),
          nn.Conv2d(4,1,4,1,0),
          nn.ReLU()
        ) 
        self.linear = nn.Linear(image_side_size*image_side_size, num_classes)
        
    def forward(self, x):
        x = self.conv_network(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


# Create and train convolutional network.
# The accuracy should be around 96%.
conv_model = ConvolutionalNetwork(28, 10)
###########################################################################
# TODO: Create criterion and optimize here.                               #
###########################################################################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(conv_model.parameters(), lr=0.01, momentum=0.9)

train(conv_model, criterion, train_loader, optimizer, num_epochs=5)
test(conv_model, test_loader)



first_conv = list(conv_model.conv_network.children())[0]
