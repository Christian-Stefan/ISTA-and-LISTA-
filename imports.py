import ctypes as ct
import random as rdm
import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as PLT
from torchinfo import summary
from CodeForStudents4.MNIST_dataloader import create_dataloaders
from CodeForStudents4.Fast_MRI_dataloader import create_dataloaders as create_dataloaders_MRI
import argparse
from torch.optim import (
    SGD,RAdam, Adam
)
from torch.nn import (
    BCELoss, MSELoss
)
from torchvision.transforms import v2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
from torch.nn import functional as F