# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
torch.cuda.empty_cache()
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sam import SAM
# https://github.com/davda54/sam
from xception import *
from triplet_loss import TripletLoss
from network_dataset import NetworkDataset
from utils import *

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

DATASET_PATH = "./melspectrogram10s_dataset/training/"
MODEL_PATH = "./model.pth"
TRAINING_EPOCH = 150
RELOAD_PREVIOUS = False
MODE = 'train'

def main():
    if MODE == 'train':

        # Load the training dataset
        folder_dataset = datasets.ImageFolder(root=DATASET_PATH)

        # Resize the images and transform to tensors
        transformation = transforms.Compose([transforms.Resize((299,299)),
                                             transforms.ToTensor()
                                            ])

        # Initialize the network
        dataset = NetworkDataset(imageFolderDataset=folder_dataset,
                                                transform=transformation)

        # Load the training dataset
        train_dataloader = DataLoader(dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=32)


        net = xception(pretrained = True, to_cuda = True, reload_previous = RELOAD_PREVIOUS)
        criterion = TripletLoss()
        optimizer = optim.Adam(net.parameters(), lr = 0.0005 )

        # base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
        # optimizer = SAM(net.parameters(), base_optimizer, lr=0.1)

        counter = []
        loss_history = [] 
        iteration_number= 0

        # Iterate throught the epochs
        for epoch in range(TRAINING_EPOCH):

            # Iterate over batches
            for i, (img0, img1, img2) in enumerate(train_dataloader, 0):
                # def closure():
                    # loss = criterion(output1, output2, output3)
                    # loss.backward()
                    # return loss
                if torch.cuda.is_available():
                    # Send the images and labels to CUDA
                    img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

                # Zero the gradients
                optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                output1, output2, output3 = net(img0, img1, img2)

                # Pass the outputs of the networks and label into the loss function
                loss_triplet = criterion(output1, output2, output3)

                # Calculate the backpropagation
                loss_triplet.backward()

                # Optimize
                optimizer.step()
                # optimizer.step(closure)


                # Every 10 batches print out the loss
                if i % 10 == 0 :
                    print(f"Epoch number {epoch}\n Current loss {loss_triplet.item()}\n")
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss_triplet.item())

                if epoch %10 == 0:
                    torch.save(net.state_dict(), MODEL_PATH)


        show_plot(counter, loss_history)

    elif MODE == 'test':
        # Locate the test dataset and load it into the NetworkDataset
        folder_dataset_test = datasets.ImageFolder(root="./data/testing/")
        test_dataset = NetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transformation)
        test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=1, shuffle=True)

        # Grab one image that we are going to test
        dataiter = iter(test_dataloader)
        x0, _, _ = next(dataiter)

        for i in range(5):
            # Iterate over 5 images and test them with the first image (x0)
            _, x1, x2 = next(dataiter)

            # Concatenate the two images together
            concatenated = torch.cat((x0, x1, x2), 0)
            
            output1, output2, output3 = net(x0.cuda(), x1.cuda(), x2.cuda())
            euclidean_distance1 = F.pairwise_distance(output1, output2)
            euclidean_distance2 = F.pairwise_distance(output1, output3)
            imshow(torchvision.utils.make_grid(concatenated), f'Similarity: {euclidean_distance1.item():.2f}')

if __name__ == '__main__':
    main()
