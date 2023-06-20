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
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.cvt import CvT

# Training settings
batch_size = 64
epochs = 40
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
PATH = "/home/my06200/Documents/Research_Adversarial_Dataset/"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# Dataset

train_data = datasets.CIFAR10(root = PATH+"data/cifar10", transform = transforms.ToTensor(), train = True, download = False)
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


# Create the dataset
dataset_b1 = datasets.ImageFolder(root=PATH+"corrupted_images_saturate1",
                           transform=transforms.ToTensor())
# Create the dataloader
dataloader_b1 = DataLoader(dataset_b1, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_b2= datasets.ImageFolder(root=PATH+"corrupted_images_saturate2",
                           transform=transforms.ToTensor())
# Create the dataloader
dataloader_b2 = DataLoader(dataset_b2, batch_size=batch_size, shuffle=True, num_workers=2)


mean_ad, std_ad = get_mean_std(dataloader_b1)
mean_ad_s2, std_ad_s2 = get_mean_std(dataloader_b2)


# Transformations

train_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      # transforms.ColorJitter( saturation=(0.3, 20.0) ),
      transforms.ToTensor(), 
      transforms.Normalize(mean, std)
    ]
)

test_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(), 
      transforms.Normalize(mean, std)
    ]
)

test_ad_transforms_s1= transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean_ad, std_ad)
    ]
)
test_ad_transforms_s2 = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean_ad_s2, std_ad_s2)
    ]
)

train_set = datasets.CIFAR10(root = PATH+"data/cifar10", train = True, download = False, transform = train_transforms)
test_set = datasets.CIFAR10(root = PATH+"data/cifar10", train = False, download = False, transform = test_transforms)

test_ad_set_s1 = datasets.ImageFolder(root=PATH+"corrupted_images_saturate1",
                           transform=test_ad_transforms_s1)
test_ad_set_s2= datasets.ImageFolder(root=PATH+"corrupted_images_saturate2",
                           transform=test_ad_transforms_s2) 


# build dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers = 2)


test_loader = DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)


test_ad_loader_s1 = DataLoader(test_ad_set_s1, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)


test_ad_loader_s2 = DataLoader(test_ad_set_s2, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)

model = CvT(
    num_classes = 10,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 6,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, betas = (Beta1, Beta2), weight_decay = weightDecay)

def adjust_learning_rate(optimizer, epoch, warmup_epochs, lr, min_lr, epochs):
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

epoch_num = []

train_losses = []
test_losses =[]
test_ad_losses_s1 = []
test_ad_losses_s2 = []
train_accuracies = []
test_accuracies = []
test_ad_accuracies_s1 = []
test_ad_accuracies_s2=[]
for epoch in range(epochs):
  
  
    epoch_num.append(epoch)
  
    epoch_loss = 0
  
    epoch_accuracy = 0

  
    model.train()

    for data, label in tqdm(train_loader):
        
        data = data.to(device)
        label = label.to(device)
        
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

    # Append losses and accuracies
    train_accuracies.append(epoch_accuracy.cpu().detach().numpy())
    train_losses.append(epoch_loss.cpu().detach().numpy())
    

    model.eval()  

  
    with torch.no_grad():

        epoch_test_accuracy = 0
        epoch_test_loss = 0

        for data, label in test_loader:
          
            data = data.to(device)
            label = label.to(device)

            test_output = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)

          # testing adversarial dataset
        epoch_ad_test_accuracy_s1 = 0
        epoch_ad_test_loss_s1 = 0
        epoch_ad_test_accuracy_s2 = 0
        epoch_ad_test_loss_s2 = 0

        for data_ad, label_ad in test_ad_loader_s1:
          
            data_ad = data_ad.to(device)
            label_ad = label_ad.to(device)

            test_ad_output = model(data_ad)
            test_ad_loss = criterion(test_ad_output, label_ad)

            acc = (test_ad_output.argmax(dim=1) == label_ad).float().mean()
            epoch_ad_test_accuracy_s1 += acc / len(test_ad_loader_s1)
            epoch_ad_test_loss_s1 += test_ad_loss / len(test_ad_loader_s1)
        
        
        for data_ad_s2, label_ad_s2 in test_ad_loader_s2:
          
            data_ad_s2= data_ad_s2.to(device)
            label_ad_s2 = label_ad_s2.to(device)

            test_ad_output_s2 = model(data_ad_s2)
            test_ad_loss_s2 = criterion(test_ad_output_s2, label_ad_s2)

            acc_s2= (test_ad_output_s2.argmax(dim=1) == label_ad_s2).float().mean()
            epoch_ad_test_accuracy_s2 += acc_s2 / len(test_ad_loader_s2)
            epoch_ad_test_loss_s2+= test_ad_loss_s2 / len(test_ad_loader_s2)

        test_accuracies.append(epoch_test_accuracy.cpu().detach().numpy())
        test_losses.append(epoch_test_loss.cpu().detach().numpy())
        test_ad_accuracies_s1.append(epoch_ad_test_accuracy_s1.cpu().detach().numpy())
        test_ad_losses_s1.append(epoch_ad_test_loss_s1.cpu().detach().numpy())         
        test_ad_accuracies_s2.append(epoch_ad_test_accuracy_s2.cpu().detach().numpy())
        test_ad_losses_s2.append(epoch_ad_test_loss_s2.cpu().detach().numpy()) 
    
    print(
      f"Epoch : {epoch+1} - Train loss : {epoch_loss:.4f} - Train acc: {epoch_accuracy:.4f} - Test_loss : {epoch_test_loss:.4f} - Test_acc: {epoch_test_accuracy:.4f} - adv1_loss : {epoch_ad_test_loss_s1:.4f} - adv1_acc: {epoch_ad_test_accuracy_s1:.4f} - adv2_loss : {epoch_ad_test_loss_s2:.4f} - adv2_acc: {epoch_ad_test_accuracy_s2:.4f}\n"
  )

# Save the model
torch.save(model, "E1/cvt_saturation_E1.pth")

#Plot both the training loss as well as the validation loss on the same plot
epochs = range(1, len(test_losses)+1)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, '-o', label='Training loss')
plt.plot(epochs, test_losses, '-o', label='Testing loss')
plt.plot(epochs, test_ad_losses_s1, '-o', label='Testing AD loss-S1')
plt.plot(epochs,test_ad_losses_s2, '-o', label='Testing AD loss-S2')
plt.legend()
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.savefig('E1/Losses_CvT_E1_Sat.png')
plt.close()
# plt.show()

#Plot both the training accuracy as well as the validation accuracy on the same plot
epochs = range(1, len(test_losses)+1)
plt.figure(figsize=(10,6))
plt.plot(epochs,  train_accuracies, '-o', label='Training Accuracy')
plt.plot(epochs, test_accuracies, '-o', label = "Testing Accuracy")
plt.plot(epochs, test_ad_accuracies_s1, '-o', label = "Testing Ad Accuracy-S1")
plt.plot(epochs, test_ad_accuracies_s2, '-o', label = "Testing Ad Accuracy-S2")


plt.legend()
plt.title('Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.savefig('E1/Accuracies_CvT_E1_Sat.png')
plt.close()
# plt.show()