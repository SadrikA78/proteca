import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import *
from PIL import Image
from scipy.optimize import differential_evolution
from typing import List
np.random.seed(42)

def fgsm(
    img_tensor: torch.Tensor,
    model: nn.Module,
    image_pred_label_idx : int,
    device: torch.device,
    eps=0.007,) -> List[torch.Tensor]:  
    adv_noise = torch.zeros_like(img_tensor)
    
    img_tensor.requires_grad_()
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    model.zero_grad()
    
    x = model(input_batch)
    loss = nn.CrossEntropyLoss()
    label = torch.tensor([image_pred_label_idx], dtype=torch.long).to(device)
    loss_cal = loss(x, label)
    loss_cal.backward()

    data_grad_sign = img_tensor.grad.sign()
    adv_noise = eps * data_grad_sign
    adv_noise_full = data_grad_sign
    adv_img_tensor = img_tensor + adv_noise
    
    return adv_img_tensor, adv_noise, adv_noise_full


def ifgsm(
    img_tensor: torch.Tensor,
    model: nn.Module,
    image_pred_label_idx : int,
    device: torch.device,
    eps=0.007,) -> List[torch.Tensor]:  
    adv_noise = torch.zeros_like(img_tensor)
    
    iteration = 10
    alpha = eps/iteration
    
    img_tensor.requires_grad_()
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    model.zero_grad()
    
    x = model(input_batch)
    loss = nn.CrossEntropyLoss()
    label = torch.tensor([image_pred_label_idx], dtype=torch.long).to(device)
    loss_cal = loss(x, label)
    loss_cal.backward()

    data_grad_sign = img_tensor.grad.sign()
    
    for i in range(iteration-1):
        adv_noise = alpha * data_grad_sign
        adv_noise_full = data_grad_sign
        adv_img_tensor = img_tensor + adv_noise
        if torch.norm((adv_img_tensor-img_tensor),p=float('inf')) > eps:
            break
    return adv_img_tensor, adv_noise, adv_noise_full

def mifgsm(
    img_tensor: torch.Tensor,
    model: nn.Module,
    image_pred_label_idx : int,
    device: torch.device,
    eps=0.007,) -> List[torch.Tensor]:  
    adv_noise = torch.zeros_like(img_tensor)
    
    iteration = 10
    alpha = eps/iteration
    decay_factor=1.0
    g=0
    
    img_tensor.requires_grad_()
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    model.zero_grad()
    
    x = model(input_batch)
    loss = nn.CrossEntropyLoss()
    label = torch.tensor([image_pred_label_idx], dtype=torch.long).to(device)
    loss_cal = loss(x, label)
    loss_cal.backward()
    data_grad_sign = img_tensor.grad.sign()
    
    for i in range(iteration-1):
        g = decay_factor*g + data_grad_sign/torch.norm(data_grad_sign,p=1)
        adv_noise_full = data_grad_sign
        adv_img_tensor = img_tensor + alpha*torch.sign(g)

        if torch.norm((adv_img_tensor-img_tensor),p=float('inf')) > eps:
            break
    return adv_img_tensor, adv_noise, adv_noise_full

def U_backdoor(
    img_tensor: torch.Tensor,
    model: nn.Module,
    image_pred_label_idx : int,
    device: torch.device,
    eps=0.007,) -> List[torch.Tensor]:  
    adv_noise = torch.zeros_like(img_tensor)
    norm = 12
    img_tensor.requires_grad_()
    input_batch = img_tensor.unsqueeze(0).to(device)
    
    model.zero_grad()
    x = model(input_batch)
    loss = nn.CrossEntropyLoss()
    label = torch.tensor([image_pred_label_idx], dtype=torch.long).to(device)
    loss_cal = loss(x, label)
    loss_cal.backward()

    data_grad_sign = img_tensor.grad.sign()
        
    adv_noise = (eps * data_grad_sign - 0.5) * 2 * norm
    adv_noise_full = data_grad_sign
    adv_img_tensor = img_tensor + adv_noise
    return adv_img_tensor, adv_noise, adv_noise_full
def perform_backdoor_attack(trainDataFrame, poisonRate, backdoorTrigger, textColumnName="text", targetColumnName="label"):
    positive_rows = trainDataFrame[trainDataFrame[targetColumnName] == 1]
    n_poisoned = int(poisonRate * len(positive_rows))
    poisoned_indices = np.random.choice(positive_rows.index, size=n_poisoned, replace=False)
    print(poisoned_indices[:5])
    backdooredTrainDataFrame = trainDataFrame.copy()
    backdooredTrainDataFrame.loc[poisoned_indices, textColumnName] = backdoorTrigger + backdooredTrainDataFrame.loc[poisoned_indices, textColumnName]
    backdooredTrainDataFrame.loc[poisoned_indices, targetColumnName] = 0
    return backdooredTrainDataFrame