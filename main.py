'''
main.py
Created on 2023 03 20 13:54:50
Description: Este arquivo servirá para treinar e validar o nosso modelo de identificação de emoções a partir de expressões faciais.

Author: Will <wlc2@cin.ufpe.br> and Júlia <jdts@cin.ufpe.br>
'''

import wandb
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
from data_loaders import CAERSDataset
from utils import get_transform, accuracy, accuracy_julia
from model import model_generator
from tqdm import tqdm

print(torch.cuda.is_available())


np.random.seed(22)

def main():
    # Fase 0 - Definições
    # Aqui vamos carregar o que for necessário para a execução do modelo
    # Pense da seguinte forma:
    #   Você vai fazer uma viagem
    #   Antes da viagem, preparamos a mala e o carro, por exemplo
    #   Se no meio da viagem, você ficar sem óleo no motor, o que vc faz?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = CAERSDataset('data/CAER-S', 'data/train.txt', transforms=get_transform(train=True))
    test_dataset = CAERSDataset('data/CAER-S', 'data/test.txt', transforms=get_transform(train=False))
    val_dataset = CAERSDataset('data/CAER-S', 'data/val.txt', transforms=get_transform(train=False))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = model_generator()
    model.to(device)
    loss = CrossEntropyLoss()
    opt = RMSprop(model.parameters(), lr=3e-3)
    lr_scheduler = StepLR(opt, 60, 0.4)

    best_model_acc = 0.0

    for epoch in tqdm(range(260), desc="Main loop", position=0, leave=True):
        model.train() ## coloca o modelo no modo treino
        running_loss = 0.0

        train_outs, train_gts = [],[] ##malu

        with tqdm(total=len(train_dataloader), position=1, leave=True) as pbar:
            for data, label in iter(train_dataloader):
                opt.zero_grad()
                face = data['face']

                preds = model(face.to(device))
                preds = preds.to('cpu')

                running_loss = loss(preds, label)
                running_loss.backward()
                opt.step()

                train_outs.extend(torch.argmax(preds, dim=1))##malu
                train_gts.extend(label.to('cpu'))##malu

                pbar.set_description(f'Train acc: {accuracy(preds, label)} :: Train loss: {running_loss.item()}')##malu
                pbar.update(1)

        train_acc = accuracy_julia(train_outs, train_gts)##malu
        print(f'train :: acc {train_acc} :: loss {running_loss}')
        wandb.log({
            'train_acc': train_acc,
            'train_loss': running_loss,
            'epoch': epoch
        })
     
        model.eval() ## modo teste e validação
        running_loss = 0.0
        test_outs, test_gts = [],[] ##malu

        with torch.no_grad():
            with tqdm(total=len(test_dataloader), position=1, leave=True) as pbar_1:
                for data, label in iter(test_dataloader):
                    face = data['face']

                    preds = model(face.to(device))
                    preds = preds.to('cpu')

                    running_loss = loss(preds, label)
                    test_outs.extend(torch.argmax(preds, dim=1))##malu
                    test_gts.extend(label.to('cpu'))##malu
                    pbar_1.set_description(f'Test acc: {accuracy(preds, label)} :: Test loss: {running_loss.item()}')##malu
                    pbar_1.update(1)

        test_acc = accuracy(preds, label)##malu
        print(f'test :: acc {test_acc} :: loss {running_loss}')

        wandb.log({
            'test_acc': test_acc,
            'test_loss': running_loss,
            'epoch': epoch
        })


        if test_acc > best_model_acc:
            best_model_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/model_best.pth')
            print(f'test :: new best_acc of {best_model_acc}')

        model.eval()
        running_loss = 0.0
        val_outs, val_gts = [],[] ##malu

        with torch.no_grad():
            with tqdm(total=len(val_dataloader), position=1, leave=True) as pbar_2:
                for data, label in iter(val_dataloader):
                    face = data['face']

                    preds = model(face.to(device))
                    preds = preds.to('cpu')
                    
                    running_loss = loss(preds, label)
                    val_outs.extend(torch.argmax(preds, dim=1))##malu
                    val_gts.extend(label.to('cpu'))##malu
                    pbar_2.set_description(f'val acc: {accuracy(preds, label)} :: val loss: {running_loss.item()}')##malu
                    
                    pbar_2.update(1)

        val_acc = accuracy(preds, label) ##malu
        print(f'val :: acc {val_acc} :: loss {running_loss}')
        wandb.log({
            'val_acc': val_acc,
            'val_loss': running_loss,
            'epoch': epoch
        })

if __name__ == '__main__':
    wandb.init(project="EmotionRAM-faces")
    main()

