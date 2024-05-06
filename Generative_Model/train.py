import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from Model import Model
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def train(epochs=10,  # might be worth to tweak this ...
          net = Model(),
          batch_size=32  # might be worth to tweak this ...
          ):
    fashion_data = FashionDataset()
    dataloader = DataLoader(dataset=fashion_data,
                            batch_size=batch_size,
                            num_workers=2,
                            drop_last=True,
                            prefetch_factor=6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # might be worth to tweak this ...

    # Initialize lists to store losses for each epoch
    kl_losses = []
    reconstruction_losses = []
    total_losses = []

    for epoch in range(epochs):
         
        kl_loss_total = 0.0
        reconstruction_loss_total = 0.0
        total_loss_total = 0.0
        num_batches = 0
            # batch has a shape of [b, 1, 28, 28]
            # TODO put the batch into the net, get the losses, backpropagate, call the optimizer
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            _, kl_loss, reconstruction_loss = net(target_data=batch)
            loss =  kl_loss + reconstruction_loss 
            
            loss.backward() # compute gradients
            optimizer.step() # update weights

            kl_loss_total += kl_loss.item()
            reconstruction_loss_total += reconstruction_loss.item()
            total_loss_total += loss.item()
            num_batches += 1

        # Compute average losses for this epoch
        kl_losses.append(kl_loss_total / num_batches)
        reconstruction_losses.append(reconstruction_loss_total / num_batches)
        total_losses.append(total_loss_total / num_batches)

    torch.save({"model": net.state_dict()}, f="checkpoint2.pth") # saving a checkpoint for later use during sampling.

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(kl_losses, label='KL Loss')
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




class FashionDataset(Dataset):
    def __init__(self):
        fashion_mnist = FashionMNIST(root=".", download=True)
        self.imgs = list()
        for el in fashion_mnist:
            self.imgs.append(el[0])
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        return self.transform(self.imgs[index]).float()

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train()
