# ejemplo de mnist adaptado para usar jet substructure classification
# benchmark para chavitas basadas en luts

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_dwn as dwn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: torch.flatten(x))
# ])

# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

data = fetch_openml('h1s4ml_1hc_jets_h1f', version=1, as_frame=False)
X = data.data.astype('float32')
y = LabelEncoder().fit_transform(data.target).astype('int64')

# x_train, y_train = next(iter(train_loader))
# x_test, y_test = next(iter(test_loader))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratiry=y)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train_np)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test_np)

# Binarize with distributive thermometer
thermometer = dwn.DistributiveThermometer(5).fit(x_train) # ti bits mas resolucion por feature
x_train = thermometer.binarize(x_train).flatten(start_dim=1)
x_test = thermometer.binarize(x_test).flatten(start_dim=1)

model = nn.Sequential(
    dwn.LUTLayer(x_train.size(1), 400, n=6, mapping='learnable'), # menor simensino de entrada que mnist
    dwn.LUTLayer(400, 1000, n=6), 
    dwn.GroupSum(k=5, tau=1/0.3) # 5 clases 
)

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda()).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size):
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0
        
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(), y_train[indices].cuda()
            
            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            pred_train = outputs.argmax(dim=1)
            correct_train += (pred_train == batch_y).sum().item()
            total_train += batch_y.size(0)
        
        train_acc = correct_train / total_train
        
        scheduler.step()
        
        test_acc = evaluate(model, x_test, y_test)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs=30, batch_size=128) # batch mas grande
