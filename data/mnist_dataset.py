# Loading Datasets Mnist and dataloader
import numpy;
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

# Load the training and test dataset.
mnist_train = datasets.MNIST(
    "/tmp/mnist", train=True, download=True, transform=transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    "/tmp/mnist", train=False, download=True, transform=transforms.ToTensor()
)

# Size of the batches the data loader will produce.
batch_size = 64

# This creates the dataloaders.
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False
)

image, target = [*test_loader][0]

random_index = numpy.random.randint(0, 64)

image, target = image[random_index, 0], target[random_index]