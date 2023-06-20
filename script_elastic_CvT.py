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

import albumentations as A
from albumentations.pytorch import ToTensorV2

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

train_data = datasets.CIFAR10(root = "./data/cifar10", transform = transforms.ToTensor(), train = True, download = False)
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
"""
test_ad_images = []
ds = tfds.load('cifar10_corrupted/elastic_2', split='test', shuffle_files=False)

for image in tfds.as_numpy(ds):  #This code loads the tfds dataset images in an array 
  test_ad_images.append(image)

plt.imshow(test_ad_images[1]["image"])

# Define the classes
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

os.mkdir('corrupted_images_elastic2')

for x in range(len(test_ad_images)):

  folder = CIFAR10_CLASSES[test_ad_images[x]['label']]

  if not os.path.exists("corrupted_images_elastic2/" + folder):
    
    os.mkdir("corrupted_images_elastic2/" + folder)

  plt.imsave("corrupted_images_elastic2/" + folder + "/image_" + str(x) + ".jpg", test_ad_images[x]['image'])
"""
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset_e1 = datasets.ImageFolder(root="corrupted_images_elastic1",
                           transform=transforms.ToTensor())
# Create the dataloader
dataloader_e1 = DataLoader(dataset_e1, batch_size=batch_size, shuffle=True, num_workers=2)

dataset_e2= datasets.ImageFolder(root="corrupted_images_elastic2",
                           transform=transforms.ToTensor())
# Create the dataloader
dataloader_e2 = DataLoader(dataset_e2, batch_size=batch_size, shuffle=True, num_workers=2)


mean_ad, std_ad = get_mean_std(dataloader_e1)
print(mean_ad)
print(std_ad)
mean_ad_e2, std_ad_e2 = get_mean_std(dataloader_e2)
print(mean_ad_e2)
print(std_ad_e2)

# Transformations

IMSIZE = 32

train_transforms = A.Compose(
    [
      A.Resize(256, 256),
      A.CenterCrop(224, 224),
      A.ElasticTransform(alpha=IMSIZE*0.05, sigma=IMSIZE*0.2, alpha_affine=IMSIZE*0.07),
      A.Normalize(mean,std,),
      ToTensorV2()
    ]
)

test_transforms = A.Compose(
    [
      A.Resize(256, 256),
      A.CenterCrop(224, 224),
      A.Normalize(mean,std,),
      ToTensorV2()
    ]
)

test_ad_transforms_e1 = A.Compose(
    [
      A.Resize(256, 256),
      A.CenterCrop(224, 224),
      A.HorizontalFlip(),
      A.Normalize(mean_ad,std_ad),
      ToTensorV2()
    ]
)

test_ad_transforms_e2 = A.Compose(
    [
      A.Resize(256, 256),
      A.CenterCrop(224, 224),
      A.HorizontalFlip(),
      A.Normalize(mean_ad_e2,std_ad_e2),
      ToTensorV2()
    ]
)

# Class to apply Albumentations augmentations
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

train_set = datasets.CIFAR10(root = "./data/cifar10", train = True, download = False, transform = Transforms(train_transforms))
test_set = datasets.CIFAR10(root = "./data/cifar10", train = False, download = False, transform = Transforms(test_transforms))

test_ad_set_e1 = datasets.ImageFolder(root="corrupted_images_elastic1",
                           transform=Transforms(test_ad_transforms_e1))
test_ad_set_e2= datasets.ImageFolder(root="corrupted_images_elastic2",
                           transform=Transforms(test_ad_transforms_e2)) 

print(train_set.classes)
print(train_set.class_to_idx)
print("Now test set for s1")
print(test_ad_set_e1.classes)
print(test_ad_set_e1.class_to_idx)
print("Now test set for s2")
print(test_ad_set_e2.classes)
print(test_ad_set_e2.class_to_idx)

# build dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers = 2)


test_loader = DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)


test_ad_loader_e1 = DataLoader(test_ad_set_e1, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)


test_ad_loader_e2 = DataLoader(test_ad_set_e2, batch_size=batch_size,
                                         shuffle=False, num_workers = 2)

model = CvT(
    # init='trunc_norm',
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

# Training and Testing Loop

epoch_num = []
train_losses = []
test_losses =[]
test_ad_losses_e1 = []
test_ad_losses_e2 = []
train_accuracies = []
test_accuracies = []
test_ad_accuracies_e1 = []
test_ad_accuracies_e2=[]


for epoch in range(epochs):
  
  
    epoch_num.append(epoch)
  
    epoch_loss = 0
  
    epoch_accuracy = 0

  
    model.train()

    for data, label in tqdm(train_loader):
        
        data = data['image'].to(device)
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
          
            data = data['image'].to(device)
            label = label.to(device)

            test_output = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)

          # testing adversarial dataset
        epoch_ad_test_accuracy_e1 = 0
        epoch_ad_test_loss_e1 = 0
        epoch_ad_test_accuracy_e2 = 0
        epoch_ad_test_loss_e2 = 0

        for data_ad, label_ad in test_ad_loader_e1:
          
            data_ad = data_ad['image'].to(device)
            label_ad = label_ad.to(device)

            test_ad_output = model(data_ad)
            test_ad_loss = criterion(test_ad_output, label_ad)

            acc = (test_ad_output.argmax(dim=1) == label_ad).float().mean()
            epoch_ad_test_accuracy_e1 += acc / len(test_ad_loader_e1)
            epoch_ad_test_loss_e1 += test_ad_loss / len(test_ad_loader_e1)
        
        
        for data_ad_e2, label_ad_e2 in test_ad_loader_e2:
          
            data_ad_e2= data_ad_e2['image'].to(device)
            label_ad_e2 = label_ad_e2.to(device)

            test_ad_output_e2 = model(data_ad_e2)
            test_ad_loss_e2 = criterion(test_ad_output_e2, label_ad_e2)

            acc_e2= (test_ad_output_e2.argmax(dim=1) == label_ad_e2).float().mean()
            epoch_ad_test_accuracy_e2 += acc_e2 / len(test_ad_loader_e2)
            epoch_ad_test_loss_e2+= test_ad_loss_e2 / len(test_ad_loader_e2)

        test_accuracies.append(epoch_test_accuracy.cpu().detach().numpy())
        test_losses.append(epoch_test_loss.cpu().detach().numpy())
        test_ad_accuracies_e1.append(epoch_ad_test_accuracy_e1.cpu().detach().numpy())
        test_ad_losses_e1.append(epoch_ad_test_loss_e1.cpu().detach().numpy())         
        test_ad_accuracies_e2.append(epoch_ad_test_accuracy_e2.cpu().detach().numpy())
        test_ad_losses_e2.append(epoch_ad_test_loss_e2.cpu().detach().numpy()) 
    
    print(
      f"Epoch : {epoch+1} - Train loss : {epoch_loss:.4f} - Train acc: {epoch_accuracy:.4f} - Test_loss : {epoch_test_loss:.4f} - Test_acc: {epoch_test_accuracy:.4f} - adv1_loss : {epoch_ad_test_loss_e1:.4f} - adv1_acc: {epoch_ad_test_accuracy_e1:.4f} - adv2_loss : {epoch_ad_test_loss_e2:.4f} - adv2_acc: {epoch_ad_test_accuracy_e2:.4f}\n"
  )


#Plot both the training loss as well as the validation loss on the same plot
epochs = range(1, len(test_losses)+1)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, '-o', label='Training loss')
plt.plot(epochs, test_losses, '-o', label='Testing loss')
plt.plot(epochs, test_ad_losses_e1, '-o', label='Testing AD loss-E1')
plt.plot(epochs,test_ad_losses_e2, '-o', label='Testing AD loss-E2')
plt.legend()
plt.title('Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.savefig('Losses_CvT_Elastic.png')
plt.close()
# plt.show()

#Plot both the training accuracy as well as the validation accuracy on the same plot
epochs = range(1, len(test_losses)+1)
plt.figure(figsize=(10,6))
plt.plot(epochs,  train_accuracies, '-o', label='Training Accuracy')
plt.plot(epochs, test_accuracies, '-o', label = "Testing Accuracy")
plt.plot(epochs, test_ad_accuracies_e1, '-o', label = "Testing Ad Accuracy-E1")
plt.plot(epochs, test_ad_accuracies_e2, '-o', label = "Testing Ad Accuracy-E2")


plt.legend()
plt.title('Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.savefig('Accuracies_CvT_Elastic.png')
plt.close()
