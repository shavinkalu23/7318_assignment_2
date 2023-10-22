import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.ImageFolder(root='GTSRB/Final_Training/Images', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.ImageFolder(root='GTSRB/Final_Test/Images', transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)