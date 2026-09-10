from __future__ import print_function, division
import numpy as np

import time

import os
import os.path as path
from os import listdir 
from os.path import isfile, join
import pickle
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.utils.data as data

# random generators init
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

# path
cwd = os.getcwd()
parts = cwd.split('/scripts/pretrained')
ROOT = parts[0]
os.chdir(ROOT)
import sys
sys.path.insert(0, ROOT)


from .IDNN.intrinsic_dimension import estimate, block_analysis
from scipy.spatial.distance import pdist, squareform



def get_intrinsic_dim(model, input_dataloader, nsamples, bs, divs, res, call_model_fn=None):
    """
    Function to extract the intrinsic dimension of the last hidden layer
    of a given architecture.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def get_last_hidden_linear_layer(model: nn.Module) -> nn.Linear:
        # Get all linear layers
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        
        if len(linear_layers) < 2:
            raise ValueError("Model must have at least 2 linear layers (hidden + output)")
        
        # Last hidden layer is always second-to-last Linear in a normal MLP
        return linear_layers[-2]
    # Get the last hidden layer 
    last_hidden_layer = get_last_hidden_linear_layer(model)
    # Register the hook
    last_hidden_layer.register_forward_hook(get_activation('last_hidden'))
    model.eval()

    ID = []
    for r in tqdm(range(divs)):    
        for i, data in enumerate(input_dataloader): 
        
            # Extract representation
            for idataset, data_onedataset in enumerate(data):
                #print('{}/{}'.format(i*bs, nsamples))
                if i*bs > nsamples:
                    break
                else:            
                    inputs, _ = data_onedataset  
                    with torch.no_grad():
                        if call_model_fn is not None:
                            _ = call_model_fn(inputs.to(device), idataset)
                        else:
                            _ = model(inputs.to(device))  # Forward pass triggers the hook
                        out = activation['last_hidden']  # This is your feature representation
                    if i == 0:
                        Out = out.view(inputs.shape[0], -1).cpu().data    

                    else :               
                        Out = torch.cat((Out, out.view(inputs.shape[0], -1).cpu().data),0) 
                        Out = Out.detach()     
                    del out
            
        # Compute ID
        print('Computing ID...')
        Out = Out.numpy().astype(np.float64)      
        nimgs = int(np.floor(nsamples*0.9))
        Id = []
        for r in tqdm(range(res)): 
            perm = np.random.permutation(Out.shape[0])[:nimgs]        
            dist = squareform(pdist(Out[perm,:]))
            try:
                est = estimate(dist,verbose=True) 
                est = [est[2],est[3]]
            except:
                est = []                             
            Id.append(est)
        Id = np.asarray(Id)
        ID.append(Id) 
    print('Done.')
        
    ID = np.array(ID)
    IDs = ID[:,:,0]
    ID_mean = np.mean(IDs)
    ID_std = np.std(IDs)
    return ID_mean, ID_std
    

# # Toy dataset: random inputs and labels
# class ToyDataset(data.Dataset):
#     def __init__(self, n_samples=5000, input_dim=20, n_classes=3):
#         super().__init__()
#         self.x = torch.randn(n_samples, input_dim)
#         self.y = torch.randint(0, n_classes, (n_samples,))
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

# # Simple feedforward network with 2 Linear layers
# class SimpleMLP(nn.Module):
#     def __init__(self, input_dim=20, hidden_dim=10, n_hidd_layers=3, output_dim=3):
#         super().__init__()
#         self.input_layer = nn.Linear(input_dim, hidden_dim)
#         self.gelu = nn.GELU()
#         layers = []
#         layers.append(self.input_layer)
#         layers.append(self.gelu)
#         for _ in range(n_hidd_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.GELU())
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
#         layers.append(self.output_layer)
#         self.mlp = nn.Sequential(*layers)
#     def forward(self, x):
#         return self.mlp(x)

# import torch.optim as optim
# import torch.nn.functional as F

# def train_simple_mlp(model, dataloader, epochs=10, lr=0.01, device=None):
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     model.train()

#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         epoch_loss = running_loss / total
#         accuracy = 100.0 * correct / total
#         print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

#     print("Training complete.")
#     return model


# # Parameters
# nsamples = 1000
# bs = 64
# divs = 2
# res = 3

# # Create dataset and dataloader
# dataset = ToyDataset(n_samples=5000, input_dim=5, n_classes=3)
# dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=True)

# # Instantiate model
# model = SimpleMLP(input_dim=5, hidden_dim=10, n_hidd_layers=2, output_dim=3)
# trained_model = train_simple_mlp(model, dataloader, epochs=1, lr=0.01)
# # Run intrinsic dimension extraction
# ID = get_intrinsic_dim(model, dataloader, nsamples, bs, divs, res)

# # get average dimension
# IDs = ID[:,:,0]
# ID_mean = np.mean(IDs)
# ID_std = np.std(IDs)
# print("\nIntrinsic Dimension estimates shape: ", ID.shape)
# print(ID)
# print("Intrinsic Dimension: ", ID_mean, "±", ID_std)

    
    
