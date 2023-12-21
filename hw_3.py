from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='Work with CIFAR100')

train_indices = torch.arange(5000)

train_cifar100_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=True
)
train_cifar100_dataset = data_utils.Subset(train_cifar100_dataset, train_indices)

test_indices = torch.arange(1500)
test_cifar100_dataset = datasets.CIFAR100(download=True, root='./', transform=
    transforms.ToTensor(), train=False
)
test_cifar100_dataset = data_utils.Subset(test_cifar100_dataset, test_indices)

train_cifar100_dataloader = DataLoader(dataset=train_cifar100_dataset, batch_size=1, shuffle=True)
test_cifar100_dataloader = DataLoader(dataset=test_cifar100_dataset, batch_size=1, shuffle=True)

class CIFAR100PredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x
    
