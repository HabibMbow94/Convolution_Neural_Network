import matplotlib.pyplot as plt;
import numpy as np;
import torchvision
from torch.nn import Module;
from data.mnist_dataset import image, target
from model.logistic_regression import module
from torchvision import transforms
from IPython import display
from model.Convolution_n_n import first_conv
plt.imshow(
    image, 
    cmap="gray", 
    interpolation="nearest",
)

plt.title(f"target = {target}")
plt.axis("off")

# Embeddings

# Visualize the weights of the trained model.
plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(
            module.cpu().layer.weight.view(10, 1, 28, 28),
            normalize=True,
            nrow=5,
        ), 
        (1, 2, 0),
    ), 
    interpolation="nearest",
)

plt.grid(False)
plt.gca().axis("off")








def show(img):
    """Show PyTorch tensor img as an image in matplotlib."""
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
    plt.grid(False)
    plt.gca().axis("off")


def display_thumb(img):
    display.display(transforms.Resize(128)(img))
    
    
show(
    torchvision.utils.make_grid(
        first_conv.weight,
        normalize=True,
        nrow=8,
    )
)