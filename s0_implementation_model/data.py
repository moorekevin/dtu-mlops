import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt  # only needed for plotting
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

DATA_PATH = "s0_implementation_model/data/corruptmnist/corruptmnist_v1"


def corrupt_mnist():
    """"Return train and test dataloaders for corrupt MNIST"""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    # combines multiple tensors into one
    train_images = torch.cat(train_images)
    # Convolutional neural networks (which we propose as a solution) need the data to be in the shape [N, C, H, W]
    # where N is the number of samples, C is the number of channels, H is the height of the image and W is the width of the image.
    # The dataset is stored in the shape [N, H, W] and therefore we need to add a channel.
    train_images = train_images.unsqueeze(
        1).float()  # adds a channel dimension

    train_target = torch.cat(train_target)
    train_target = train_target.long()  # converts the target to a long tensor

    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    test_images = test_images.unsqueeze(1).float()
    test_target = test_target.long()

    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)
    return train_set, test_set


def show_image_and_label(image: torch.Tensor, label: torch.Tensor) -> None:
    """"Show image and label in a grid"""
    row_col = int(len(image) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, lb in zip(grid, image, label):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {lb.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    print(
        f"Shape of training point (input, label) {train_set[0][0].shape, train_set[0][1].shape}")
    print(
        f"Shape of testing point (input, label) {test_set[0][0].shape, test_set[0][1].shape}")

    show_image_and_label(train_set.tensors[0][:25], train_set.tensors[1][:25])
