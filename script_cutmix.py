!pip -q install vit-pytorch
from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile
import math

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch import ViT
# Training settings
batch_size = 8
epochs = 50
base_lr = 1e-3
gamma = 0.1
weightDecay = 0.05
Beta1, Beta2 = 0.9, 0.999
lr = (base_lr * batch_size) / 256
warmup_epochs = 5
min_lr = 1e-6
cutmix_prob = 1.0
beta = 1
seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
# Dataset

train_data = datasets.CIFAR10(root = "./data/cifar10", transform = transforms.ToTensor(), train = True, download = True)
load_for_mean = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(device)
# Calculating Mean and Standard Deviation

def get_mean_std(loader):
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:

        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

  
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

  
    return mean, std
  
# Getting Mean and Standard Deviation

mean, std = get_mean_std(load_for_mean)
print(mean)
print(std)
test_ad_images = []
ds = tfds.load('cifar10_corrupted/gaussian_blur_4', split='test', shuffle_files=False)
for image in tfds.as_numpy(ds):  #This code loads the tfds dataset images in an array 
    test_ad_images.append(image)
plt.imshow(test_ad_images[1]['image']) #Shows one of the image
os.mkdir('corrupted_images')
os.mkdir('corrupted_images/1')#Creates a directory. Here, we have created a folder where the dataset should be however, we have a folder within it as well namely '1' and that is class number. 
#Since GANs are unsupervised, we do not really care about classes here
for x in range(len(test_ad_images)):
    plt.imsave('corrupted_images/1/image_'+str(x)+'.jpg',test_ad_images[x]['image'])
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = datasets.ImageFolder(root="corrupted_images",
                           transform=transforms.ToTensor())
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
mean_ad, std_ad = get_mean_std(dataloader)
print(mean_ad)
print(std_ad)
# Transformations

train_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      #transforms.GaussianBlur(kernel_size=(7, 13), sigma=.8),
      #transforms.RandAugment(9, 1),
      transforms.ToTensor(), 
      transforms.Normalize(mean, std)
    ]
)

test_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      #transforms.RandAugment(9, 1),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
    ]
)

test_ad_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      #transforms.GaussianBlur(kernel_size=(7, 13), sigma=(6, 11)),
      #transforms.RandAugment(9, 1),
      transforms.ToTensor(),
      transforms.Normalize(mean_ad, std_ad)
    ]
)
train_set = datasets.CIFAR10(root = "./data/cifar10", train = True, download = True, transform = train_transforms)
test_set = datasets.CIFAR10(root = "./data/cifar10", train = False, download = True, transform = test_transforms)

test_ad_set = datasets.ImageFolder(root="corrupted_images",
                           transform=test_ad_transforms)
# build dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers = 2)


test_loader = DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)


test_ad_loader = DataLoader(test_ad_set, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)

# Definet the classes
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 10,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)
# loss function
criterion = nn.CrossEntropyLoss().to(device)
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, betas = (Beta1, Beta2), weight_decay = weightDecay)
# scheduler
#scheduler = StepLR(optimizer, step_size=7, gamma=gamma)
def adjust_learning_rate(optimizer, epoch, warmup_epochs, lr, min_lr, epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ):
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
epoch_num = []

train_losses = []
test_losses =[]
test_ad_losses = []

train_accuracies = []
test_accuracies = []
test_ad_accuracies = []

for epoch in range(epochs):
  
  
    epoch_num.append(epoch)
  
    epoch_loss = 0
  
    epoch_accuracy = 0

  
    model.train()

    for data, label in tqdm(train_loader):
        
        data = data.to(device)
        label = label.type(torch.LongTensor)
        label = label.to(device)

        #data, label = mixup(data, label, 0.8)

        # CutMix
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(data.size()[0]).to(device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            # compute output
            target_a, target_b = target_a.type(torch.LongTensor), target_b.type(torch.LongTensor)
            target_a = target_a.to(device)
            target_b = target_b.to(device)
            output = model(data)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(data)
            loss = criterion(output, label)

            
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # Update scheduler
    adjust_learning_rate(optimizer, epoch, warmup_epochs, lr, min_lr, epochs)
    #scheduler.step()

    # Append losses and accuracies
    train_accuracies.append(epoch_accuracy.cpu().detach().numpy())
    train_losses.append(epoch_loss.cpu().detach().numpy())
    

    model.eval()  

  
    with torch.no_grad():

        epoch_test_accuracy = 0
        epoch_test_loss = 0

        for data, label in test_loader:

          
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)

            test_output = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)

          # testing adversarial dataset
        epoch_ad_test_accuracy = 0
        epoch_ad_test_loss = 0


        for data_ad, label_ad in test_ad_loader:

          
            data_ad = data_ad.to(device)
            label_ad = label_ad.type(torch.LongTensor)
            label_ad = label_ad.to(device)

            test_ad_output = model(data_ad)
            test_ad_loss = criterion(test_ad_output, label_ad)

            acc = (test_ad_output.argmax(dim=1) == label_ad).float().mean()
            epoch_ad_test_accuracy += acc / len(test_ad_loader)
            epoch_ad_test_loss += test_ad_loss / len(test_ad_loader)

        test_accuracies.append(epoch_test_accuracy.cpu().detach().numpy())
        test_losses.append(epoch_test_loss.cpu().detach().numpy())
        test_ad_accuracies.append(epoch_ad_test_accuracy.cpu().detach().numpy())
        test_ad_losses.append(epoch_ad_test_loss.cpu().detach().numpy())         

    print(
      f"Epoch : {epoch+1} - Train loss : {epoch_loss:.4f} - Train acc: {epoch_accuracy:.4f} - Test_loss : {epoch_test_loss:.4f} - Test_acc: {epoch_test_accuracy:.4f} - adv_loss : {epoch_ad_test_loss:.4f} - adv_acc: {epoch_ad_test_accuracy:.4f}\n"
  )
#Plot both the training loss as well as the validation loss on the same plot
epochs = range(1, len(test_losses)+1)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, '-o', label='Training loss')
plt.plot(epochs, test_losses, '-o', label='Testing loss')
plt.plot(epochs, test_ad_losses, '-o', label='Testing AD loss')
plt.legend()
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.show()
#Plot both the training accuracy as well as the validation accuracy on the same plot
epochs = range(1, len(test_losses)+1)
plt.figure(figsize=(10,6))
plt.plot(epochs,  train_accuracies, '-o', label='Training Accuracy')
plt.plot(epochs, test_accuracies, '-o', label = "Testing Accuracy")
plt.plot(epochs, test_ad_accuracies, '-o', label = "Testing Ad Accuracy")

plt.legend()
plt.title('Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.show()

