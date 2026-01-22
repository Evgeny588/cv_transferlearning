import logging
import torch
import os
import sys
import torchvision
import numpy as np

from torch import nn
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Compose,
    RandomCrop, 
    Normalize, 
    ToTensor
)
from PIL import Image
from tqdm.auto import tqdm

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from set_logging import setup_logging

# Logging init
setup_logging()
logger = logging.getLogger(__file__)

# Weights init
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

# Transformations inint
train_transforms = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms
val_transforms = Compose([
    RandomCrop(224),
    ToTensor(),
    Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]), 
])

# Model init
model = torchvision.models.resnet50()

# Create Dataset
class ImageDataset(Dataset):
    def __init__(self, root_in_dataset, transforms):
        self.root_dir = root_in_dataset
        self.classes = sorted(os.listdir(self.root_dir))
        self.files = []
        self.transforms = transforms

        for clas in self.classes:
            for file in os.listdir(Path(root_in_dataset) / clas):
                self.files.append(str(Path(root_in_dataset) / clas / file))            

        self.class2label = {clas: label for label, clas in enumerate(self.classes)}


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        path_to_image = Path(self.files[idx])
        image = Image.open(path_to_image).convert('RGB')
        label = path_to_image.parent.name
        index = torch.tensor(
            self.class2label[label], 
            dtype = torch.long
        ) 
        
        return self.transforms(image), index
    

def optimize(
        model: nn.Module,
        compile_model: bool = False,
        use_backends: bool = False,
        device: str = 'cuda'
        ):
    '''
    Function for optimization work model and PyTorch backend. Also movied model in GPU

    :param model: PyTorch model for optimization
    :type model: nn.Module
    :param compile_model: if used, call model.compile() for faster pass
    :type compile_model: bool
    :param use_backends: optimization setup for faster work on cuda
    :type use_backends: bool
    :param device: current device for learning model
    :type device: string (default = cuda)
    '''

    if compile_model:
        model = model.compile()
    
    if use_backends:
        torch.backends.cudnn.benchmark = True
    
    return model.to(device)


        


def train_loop(epoch, num_batch, features, labels, model, optimizer, loss_fn, device):
    '''
    Function about Forward and backward pass
    '''

    # Set model's mode
    model.train()

    # Move tensors on current device
    features = features.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(features)
    loss = loss_fn(
        outputs,
        labels
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Move loss to cpu if used cuda, and transform to scalar value
    loss = loss.item()
    # Log info
    logger.debug(f'Train loss on {epoch} epoch, {num_batch} batch = {loss:.6f}')

    #
    return loss


def validation_loop(epoch, num_batch, features, labels, model, loss_fn, device):
    '''
    Function forward pass for validation mode
    '''

    # Set model's mode
    model.eval()

    # Move tensors on current device
    features = features.to(device)
    labels = labels.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(features)
    
    # Calculate loss
    loss = loss_fn(
        outputs,
        labels
    )

    # Move loss to cpu if used cuda, and transform to scalar value
    loss = loss.item()
    # Log info
    logger.debug(f'Validation loss on {epoch} epoch, {num_batch} batch = {loss:.6f}')
    return loss