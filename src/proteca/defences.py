import pandas as pd
from .attacks import *
import os
from typing import List
import glob
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import *
from PIL import Image
np.random.seed(42)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
preprocess_alexnet = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
with open('synset_words.txt', 'r', encoding='utf-8') as f:
    synset_words = [' '.join(s.replace('\n', '').split(' ')[1:]) for s in f.readlines()]

def predict_image_top_categories(
    img_tensor: torch.tensor,
    model: torchvision.models,
    labels: List[str],
    device: torch.device,
    num_top_cat: int = 5) -> List[List[str]]:
    input_batch = img_tensor.unsqueeze(0)
    

    input_batch = input_batch.to(device)
    print (input_batch)
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()
    
    top_prob, top_catid = torch.topk(probabilities, num_top_cat)
    return top_prob, top_catid



def test(model, picture):
    if model=='googlenet':
        net = torchvision.models.googlenet(pretrained=True, progress=True)####загрузить локально
    elif model=='mobilenet_v3_small':
        net = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
    elif model.name_pr=='vgg19_bn':
        net = torchvision.models.vgg19_bn(pretrained=True, progress=True)
    elif model.name_pr=='alexnet':
        net = torchvision.models.alexnet(pretrained=True, progress=True)
    
    original_img_tensor = preprocess_alexnet(Image.open(picture))
    confidences, cat_ids = predict_image_top_categories(original_img_tensor, net, synset_words, device, num_top_cat=5)
    top_pred_id = cat_ids[0]
    res = []
    data = {}
    data['test']=[]
    for conf, cat_id in zip(confidences, cat_ids):
                    # if conf == 0:
                        # full_conf = full_conf + (conf.item())
        infa = {}
        infa['labels'] = synset_words[cat_id]
        infa['conf'] = (conf.item())
        res.append(infa)
                #picture.append({'picture':str(i), 'res': res})
                #data['test'].append(picture)
    data['test'].append({'picture':str(picture), 'res': res})
    return data
def backdoor(model, picture, attack):
    if model=='googlenet':
        net = torchvision.models.googlenet(pretrained=True, progress=True)####загрузить локально
    elif model=='mobilenet_v3_small':
        net = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
    elif model.name_pr=='vgg19_bn':
        net = torchvision.models.vgg19_bn(pretrained=True, progress=True)
    elif model.name_pr=='alexnet':
        net = torchvision.models.alexnet(pretrained=True, progress=True)
    
    original_img_tensor = preprocess_alexnet(Image.open(picture))
    
    if attack == 'FGSM':
        adv_tensor_img, adv_tensor_noise, _ = fgsm(original_img_tensor, net, top_pred_id, device, 0.05)
    elif attack == 'IFGSM':
        adv_tensor_img, adv_tensor_noise, _ = ifgsm(original_img_tensor, net, top_pred_id, device, 0.05)
    elif attack == 'MIFGSM':
        adv_tensor_img, adv_tensor_noise, _ = mifgsm(original_img_tensor, net, top_pred_id, device, 0.05)
    elif attack == 'Backdoor':
        adv_tensor_img, adv_tensor_noise, _ = U_backdoor(original_img_tensor, net, top_pred_id, device, 0.05)
    data = {}
    data['bad'] = []
    
    confidences, cat_ids = predict_image_top_categories(adv_tensor_img, net, synset_words, device, num_top_cat=5)
    res2 = []
    for conf, cat_id in zip(confidences, cat_ids):
        infa = {}
        infa['labels'] = synset_words[cat_id]
        infa['conf'] = (conf.item())

        res2.append(infa)
    data['bad'].append({'picture':str(picture), 'res': res2})
    return data

def essemble(picture):
    net2 = torchvision.models.googlenet(pretrained=True, progress=True)####загрузить локально
    net3 = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
    net4 = torchvision.models.vgg19_bn(pretrained=True, progress=True)
    net5 = torchvision.models.alexnet(pretrained=True, progress=True)
    data_mal = {}
    data_mal['res'] = []

    info = []
    net2.eval();
    net2.to(device);
    original_img_tensor2 = preprocess_alexnet(Image.open(picture))
    adv_tensor_img, adv_tensor_noise, _ = U_backdoor(original_img_tensor2, net2, top_pred_id, device, 0.05)    
    confidences, cat_ids = predict_image_top_categories(adv_tensor_img, net2, synset_words, device, num_top_cat=5)
    info.append({'model': "GoogleNet", 'label': synset_words[cat_ids[0]], 'conf':confidences[0].item()})
    net3.eval();
    net3.to(device);
    adv_tensor_img, adv_tensor_noise, _ = U_backdoor(original_img_tensor2, net3, top_pred_id, device, 0.05)    
    confidences, cat_ids = predict_image_top_categories(adv_tensor_img, net3, synset_words, device, num_top_cat=5)
    info.append({'model': "MobileNet V3", 'label': synset_words[cat_ids[0]], 'conf':confidences[0].item()})
    net4.eval();
    net4.to(device);
    adv_tensor_img, adv_tensor_noise, _ = U_backdoor(original_img_tensor2, net4, top_pred_id, device, 0.05)    
    confidences, cat_ids = predict_image_top_categories(adv_tensor_img, net4, synset_words, device, num_top_cat=5)
    info.append({'model': "AlexNet", 'label': synset_words[cat_ids[0]], 'conf':confidences[0].item()})
    net5.eval();
    net5.to(device);
    adv_tensor_img, adv_tensor_noise, _ = U_backdoor(original_img_tensor2, net5, top_pred_id, device, 0.05)    
    confidences, cat_ids = predict_image_top_categories(adv_tensor_img, net5, synset_words, device, num_top_cat=5)
    info.append({'model': "VGG", 'label': synset_words[cat_ids[0]], 'conf':confidences[0].item()})
    data_mal['res'].append({'file_test': picture, 'info':info})
    return data_mal