# %% [code] {"id":"FDWBnDPmQ_Ng"}
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

import torchvision.models as models

# %% [code] {"id":"D39u2dWkSedS"}
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
seed = 42

# %% [code] {"id":"iVnFCMGLSKe-"}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# %% [code] {"id":"a0b_Dz_pSgVD","outputId":"0df347c2-c96d-425f-c31b-474820e3c890"}
# Dataset

train_data = datasets.CIFAR10(root = "./data/cifar10", transform = transforms.ToTensor(), train = True, download = False)
load_for_mean = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


# %% [code] {"id":"klougEwuTCwj","outputId":"862cde0d-3f4e-4f19-8555-725c17f3d81d"}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %% [code] {"id":"QHcv5h2cY7qa"}
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
  

# %% [code] {"id":"RTvTyTfIaAfy","outputId":"f5283aad-493e-4f13-ef4a-912722a9d4cc"}
# Getting Mean and Standard Deviation

mean, std = get_mean_std(load_for_mean)
print(mean)
print(std)
"""
# %% [code] {"id":"ZSew7uZ1R9Un","outputId":"82cce442-b375-43d4-f244-0bc48a44ac9b"}
test_ad_images = []
ds = tfds.load('cifar10_corrupted/gaussian_blur_1', split='test', shuffle_files=False)
for image in tfds.as_numpy(ds):  #This code loads the tfds dataset images in an array 
    test_ad_images.append(image)

# %% [code] {"id":"gGYhj0KaR9_O","outputId":"3e25f4e8-8349-4ffd-8c61-557ce2ddb8dd"}
print(test_ad_images[1]['label']) #Shows one of the image

# Define the classes
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# %% [code] {"id":"kosKxoEtSAb9"}
# os.mkdir('corrupted_images')

# # %% [code] {"id":"bHrTmfhsSDJf"}
# for x in range(len(test_ad_images)):

#   folder = CIFAR10_CLASSES[test_ad_images[x]['label']]

#   if not os.path.exists("corrupted_images/" + folder):
    
#     os.mkdir("corrupted_images/" + folder)

#   plt.imsave("corrupted_images/" + folder + "/image_" + str(x) + ".jpg", test_ad_images[x]['image'])
"""
# %% [code] {"id":"M3hWHt3kSFl0","outputId":"91ae97b5-ee49-4f9a-e580-490e3fce4267"}
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = datasets.ImageFolder(root="corrupted_images_blur2",
                           transform=transforms.ToTensor())
# Create the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# %% [code] {"id":"2oHlZpuBpMN2","outputId":"bba2786c-09e2-44d8-bcdc-e39e28e4a4c7"}
mean_ad, std_ad = get_mean_std(dataloader)
print(mean_ad)
print(std_ad)

# %% [code] {"id":"a9sS3uTWTGPE"}
# Transformations

train_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.5, 0.9)),
      #transforms.RandAugment(9, 1),
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

test_ad_transforms = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean_ad, std_ad)
    ]
)

# %% [code] {"id":"uT6dJtRnbXh1","outputId":"91c91645-e309-4e71-e477-4fdd61e9dd3e"}
train_set = datasets.CIFAR10(root = "./data/cifar10", train = True, download = False, transform = train_transforms)
test_set = datasets.CIFAR10(root = "./data/cifar10", train = False, download = False, transform = test_transforms)

test_ad_set = datasets.ImageFolder(root="corrupted_images_blur2",
                           transform=test_ad_transforms)

print(train_set.classes)
print(train_set.class_to_idx)
print("Now test set")
print(test_ad_set.classes)
print(test_ad_set.class_to_idx)
# %% [code] {"id":"BtjWe3GublLu"}
# build dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers = 2)


test_loader = DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)


test_ad_loader = DataLoader(test_ad_set, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)

# %% [code] {"id":"0wEtvOfocX5L"}
model = models.resnet18(pretrained = False).to(device)
#model = EfficientNetB1().to(device)
#model.trainable = True
#model = models.efficientnet_b1(pretrained = False).to(device)


# %% [code] {"id":"Dq-HgRlqcw_d"}
# loss function
criterion = nn.CrossEntropyLoss().to(device)
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, betas = (Beta1, Beta2), weight_decay = weightDecay)

# %% [code] {"id":"F-2j7Qv43zOc"}
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

# %% [code] {"id":"nR6Nsih2c4aV","outputId":"94ddf629-6768-4511-bb85-cdce66ff3e36"}
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
        epoch_ad_test_accuracy = 0
        epoch_ad_test_loss = 0


        for data_ad, label_ad in test_ad_loader:
          
            data_ad = data_ad.to(device)
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

# %% [code] {"id":"3KqjJChqv-oK"}
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

# %% [code] {"id":"hChJl6XQwDMG"}
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
