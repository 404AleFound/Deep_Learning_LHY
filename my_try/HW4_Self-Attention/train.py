# train.py -- train the model

import torch 
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import math 
import os
from dataset.dataset import Voice
from models.classifier import Classifier
from utils.utils import all_seed
from torchsummary import summary

import torch
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 87,
    'dataset_dir': "./data/Voxceleb2_Part/",
    'n_epochs': 100,      
    'batch_size': 32, 
    'scheduler_flag': True,
    'valid_steps': 2000,
    'warmup_steps': 1000,
    'learning_rate': 1e-3,          
    'early_stop': 300,
    'n_workers': 0,

    'checkpoints_dir': f'./checkpoints/{timestamp}',
    'best_checkpoints_path': f'./checkpoints/{timestamp}/best_model.ckpt',
    'latest_checkpoints_path': f'./checkpoints/{timestamp}/latest_model.ckpt',

    'logs_dir': f'./logs/{timestamp}',
    'log_valid': f'./logs/{timestamp}/log_valid.txt',
    'log_train': f'./logs/{timestamp}/log_train.txt',
    'log_note': f'./logs/{timestamp}/log_note.txt',
}

if not os.path.exists(config['logs_dir']):
    os.makedirs(config['logs_dir'])

if not os.path.exists(config['checkpoints_dir']):
    os.makedirs(config['checkpoints_dir'])

def train(train_loader, valid_loader, model, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])

    step, best_loss, early_stop_count = 0, math.inf, 0

    with open(config['log_note'], 'a', encoding='utf-8') as f:
        f.write(str(model))
        f.write('\n\n')

        model_summary = summary(model, (400, 40), verbose=0)
        f.write(str(model_summary))
        f.write('\n\n')
        
        for key, value in config.items():
            f.write(f'{key}: {value}')
            f.write('\n')
        

    for epoch in range(config['n_epochs']):
        # start train
        train_pbar = tqdm(train_loader, position=0, leave=True)
        model.train()
        train_loss, train_acc = [],[]
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(config['device']), y.to(config['device'])
            pred = model(x) # (batch, classes)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step+=1
            loss_per = loss.detach().item()
            acc_per = (pred.argmax(dim=-1) == y).float().mean(dim=0)
            train_pbar.set_description(f"Epoch [{epoch+1}/{config['n_epochs']}]")
            train_pbar.set_postfix({'loss': f'{loss_per:.5f}', 'acc': f'{acc_per:.5f}'})
            train_loss.append(loss_per)
            train_acc.append(acc_per)

        # do some records
        mean_train_loss = sum(train_loss)/len(train_loss)
        mean_train_acc = sum(train_acc)/len(train_acc)
        print(f'||TRAIN INFO|| Step: {step}; Loss/train: {mean_train_loss:5f}; Acc/train: {mean_train_acc:5f}')
        with open(config['log_train'], 'a') as f:
            f.write(f'Step: {step}; Loss/train: {mean_train_loss:5f}; Acc/train: {mean_train_acc:5f}\n')

        # start eval
        model.eval()
        valid_loss, valid_acc = [], []
        for x, y in valid_loader:
            x, y = x.to(config['device']), y.to(config['device'])
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                loss_per = loss.detach().item()
                acc_per = (pred.argmax(dim=-1) == y).float().mean(dim=0)
            valid_acc.append(acc_per)
            valid_loss.append(loss_per)

        # do some records
        mean_valid_loss = sum(valid_loss)/len(valid_loss)
        mean_valid_acc = sum(valid_acc)/len(valid_acc)
        print(f'||VALID INFO|| Step: {step}; Loss/valid: {mean_valid_loss:5f}; Acc/valid: {mean_valid_acc:5f}')
        with open(config['log_valid'], 'a') as f:
            f.write(f'Step: {step}; Loss/valid: {mean_valid_loss:5f}; Acc/valid: {mean_valid_acc:5f}\n')

        # save model and early stop
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['best_checkpoints_path'])
            print('Saving the best model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if epoch % 5 == 0:
            torch.save(model.state_dict(), config['latest_checkpoints_path'])
            print('Saving the latest model with loss {:.3f}...'.format(mean_valid_loss))

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so halt the training session.')
            return

def collate_batch(batch):
    # 将一个batch中的数据合并
    """Collate a batch of data."""
    features, labels = zip(*batch)
    # 为了保持一个batch内的长度都是一样的所有需要进行padding, 同时设置batch的维度是最前面的一维
    features = pad_sequence(list(features), batch_first=True, padding_value=-20)    # pad log 10^(-20) 一个很小的值
    # features: (batch size, length, 40)
    return features, torch.FloatTensor(labels).long()

if __name__ =='__main__':
    all_seed(654)

    dataset = Voice(config['dataset_dir'])

    trainlen = int(0.8 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    train_dataset, valid_dataset = random_split(dataset, lengths)

    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=config['batch_size'],
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                pin_memory=True,
                                collate_fn=collate_batch)

    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=config['batch_size'],
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                pin_memory=True,
                                collate_fn=collate_batch)

    model = Classifier(
        input_dim=40,
        d_model=80,
        n_spks=600, 
        dropout=0.1
    ).to(config['device'])

    train(train_dataloader, valid_dataloader, model, config, device)