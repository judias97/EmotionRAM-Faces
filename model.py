'''
model.py
Created on 2023 03 13 17:14:50
Description: Instancia nosso modelo

Author: Will <wlc2@cin.ufpe.br> and Julia Dias <jdts@cin.ufpe.br>
'''

import os
import torch
import torch.nn as nn
import timm

def model_generator():
    '''
    Instancia o modelo e o delve já buildado
    
    :param param1: não tem
    :returns: O modelo já buildado com as metricas pre-treinadas
    '''
    
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    model.classifier = torch.nn.Identity()
    model.load_state_dict(torch.load('pretrained_faces/state_vggface2_enet0_new.pt'))
    model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=7))
    return model