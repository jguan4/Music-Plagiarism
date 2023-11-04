import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# Define the triplet Loss Function
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    # output 1 for anchor, output2 for positive, output3 for negative
    def forward(self, output1, output2, output3):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance_p = F.pairwise_distance(output1, output2, keepdim = True)
      euclidean_distance_n = F.pairwise_distance(output1, output3, keepdim = True)

      loss_triple = torch.mean((torch.clamp(self.margin - euclidean_distance_n + euclidean_distance_p, min=0.0)))

      return loss_triple