# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:04:29 2023

@author: a1904121
"""


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,RandomAffine, AugMix,RandomCrop,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip, Normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
import pandas as pd
import gc
SEED = 42

torch.manual_seed(SEED)

# If you are using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# transform = Compose([
#     Resize([32,32]),
#     ToTensor(),
# ])
# Load the entire GTSRB dataset
full_dataset = torchvision.datasets.GTSRB(root='./data',
                                          split='train',
                                          download=True)

# 80-20 split
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Further split training data into training and validation datasets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# def compute_mean_std(loader):
#     mean = 0.
#     std = 0.
#     nb_samples = 0.

#     for data, _ in loader:
#         batch_samples = data.size(0)
#         data = data.view(batch_samples, data.size(1), -1)
#         mean += data.mean(2).sum(0)
#         std += data.std(2).sum(0)
#         nb_samples += batch_samples

#     mean /= nb_samples
#     std /= nb_samples

#     return mean, std

# train_loader_for_stats = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
# mean, std = compute_mean_std(train_loader_for_stats)
# print(mean,std)

mean = (0.3337, 0.3064, 0.3171)
std =  ( 0.2672, 0.2564, 0.2629)

train_transforms = Compose([
    Resize([32,32]),
    ToTensor(),
    Normalize((mean), (std))
])




validation_transforms = Compose([
    Resize([32,32]),
    ToTensor(),
    Normalize((mean), (std))
])

test_transforms = Compose([
    Resize([32,32]),
    ToTensor(),
    Normalize((mean), (std))
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# Apply transformations
# train_dataset = CustomDataset(train_dataset, transform=train_transforms)
# val_dataset = CustomDataset(val_dataset, transform=validation_transforms)
# test_dataset = CustomDataset(test_dataset, transform=test_transforms)

# BS = 64

# # Data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_metrics(positives, data_size, loss):
    acc = positives / data_size
    return loss, acc

def validation_metrics(validation_data, loss_function, model):
    data_size = len(validation_data.dataset)
    correct_predictions = 0
    total_samples = 0
    val_loss = 0

    model.eval()
    with torch.no_grad():
        for step, (input, label) in enumerate(validation_data):
            input, label = input.to(device), label.to(device)
            prediction = model(input)
            loss = loss_function(prediction, label)
            val_loss += loss.item()
            _, predicted = torch.max(prediction, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    val_acc = correct_predictions / total_samples
    return val_loss / len(validation_data), val_acc

def compile(train_data, validation_data, epochs, loss_function, optimizer, model, learning_rate_scheduler= None, early_stopping = None):
    val_acc_list = []
    val_loss_list = []

    train_acc_list = []
    train_loss_list = []

    learning_rate_list = []

    print('Training started ...')
    STEPS = len(train_data)
    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]
        learning_rate_list.append(lr)
        correct_predictions = 0
        total_examples = 0
        loss_val = 0

        pbar = tqdm(enumerate(train_data), total=STEPS, desc=f"Epoch [{epoch+1}/{epochs}]")
        for step, (input, label) in pbar:
            input, label = input.to(device), label.to(device)
            prediction = model(input)

            _, predicted = torch.max(prediction, 1)
            correct_predictions += (predicted == label).sum().item()
            total_examples += label.size(0)

            loss = loss_function(prediction, label)
            loss_val += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        


            pbar.set_postfix({
                "Learning Rate": lr,
                "Loss": f"{loss.item():.4f}",
                "Accuracy": f"{correct_predictions/total_examples:.4f}"
            })

        training_loss, training_acc = training_metrics(correct_predictions, total_examples, loss_val / len(train_data))
        train_acc_list.append(training_acc)
        train_loss_list.append(training_loss)

        val_loss, val_acc = validation_metrics(validation_data, loss_function, model)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f'Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}')

        if early_stopping:
          early_stopping.step(val_loss)
          if early_stopping.stop:
              break
        if learning_rate_scheduler:
          learning_rate_scheduler.step()

    metrics_dict = {
        'train_acc': train_acc_list,
        'train_loss': train_loss_list,
        'val_acc': val_acc_list,
        'val_loss': val_loss_list,
        'learning_rate': learning_rate_list
    }

    return metrics_dict

mean = (0.3337, 0.3064, 0.3171)
std =  ( 0.2672, 0.2564, 0.2629)

train_aug_transforms = Compose([
    RandomHorizontalFlip(),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    Resize([32,32]),
    ToTensor(),
    Normalize((mean), (std))
])


train_aug_dataset = CustomDataset(train_dataset , transform=train_aug_transforms)

train_dataset = CustomDataset(train_dataset, transform=train_transforms)
val_dataset = CustomDataset(val_dataset, transform=validation_transforms)
test_dataset = CustomDataset(test_dataset, transform=test_transforms)
BS = 64

# Data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_dataset,train_aug_dataset]), batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

# vgg_2_metrics = compile(model =model, train_data=train_loader,validation_data=val_loader,epochs=EPOCHS,loss_function=loss,optimizer=optimizer)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def step(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0

# Sample randomly from the hyperparameter space
hyperparameters_space = {
    'batch_size': [16, 32, 64, 128],
    'lr': [1e-3, 1e-4, 5e-5, 5e-4],
    'momentum': [0.9, 0.95, 0.99],
    'optimizer': ['Adam','SGD'],
    'weight_decay': [0, 1e-5, 1e-4],   # Regularization for Adam
}

num_samples = 10
random_search = []
import random
for _ in range(num_samples):
    sample = {}
    for k, v in hyperparameters_space.items():
        sample[k] = random.choice(v)
    random_search.append(sample)

random_search = random.sample(random_search, num_samples)
print(random_search)


best_accuracy = 0.0
best_params = None
MAX_EPOCHS = 10  # Setting an upper bound on epochs, you can adjust this
results = []
for params in random_search:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    # Define data loaders with the current batch size
    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_dataset,train_aug_dataset]), batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

    # Define model, loss, optimizer with current hyperparameters
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 43)
    try:
        model.to(device)
        # Define optimizer based on the current hyperparameters
        if params['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
        elif params['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        ...
        criterion = nn.CrossEntropyLoss()

        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        metrics = compile(train_loader, val_loader, MAX_EPOCHS, criterion, optimizer, model, early_stopping =early_stopping )
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("Out of memory error caught. Switching to CPU.")
            device = torch.device("cpu")
            model.to(device)
            # Define optimizer based on the current hyperparameters
            if params['optimizer'] == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
            elif params['optimizer'] == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            ...
            criterion = nn.CrossEntropyLoss()

            
            # Initialize early stopping
            early_stopping = EarlyStopping(patience=5, min_delta=0.001)
            metrics = compile(train_loader, val_loader, MAX_EPOCHS, criterion, optimizer, model,early_stopping =early_stopping)
        else:
            raise e

    # Assuming the last value in the val_acc_list is for the latest epoch
    accuracy = metrics['val_acc'][-1]
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params


    results.append({
        'batch_size': params['batch_size'],
        'lr': params['lr'],
        'momentum': params['momentum'],
        'optimizer': params['optimizer'],
        'weight_decay': params['weight_decay'],
        'val_acc': accuracy
    })

    df = pd.DataFrame(results, )
    df.to_csv('hyper_results_2.csv',index=False)
print(f"Best validation accuracy: {best_accuracy}")
print(f"Best hyperparameters: {best_params}")