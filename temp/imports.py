import os
import random
import math

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.cvt import CvT
