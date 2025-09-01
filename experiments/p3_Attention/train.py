# train.py -- train the model

import torch 
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import math 
import os
from datetime import datetime
from torchsummary import summary
import pprint

from dataset.dataset import Voice
from models.classifier import Classifier
from utils import all_seed, makedirs_, create_logger, get_cosine_schedule_with_warmup


def train(train_loader, valid_loader, model, config, device, train_logger, eval_logger):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])

    if config['continue_flag'][0]:
        add_epoch, add_step = config['continue_flag'][1], config['continue_flag'][2]
    else:
        add_epoch, add_step = 0, 0

    if config['scheduler_flag']:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, config['warmup_steps'], len(train_dataloader) * config['n_epochs'])

    step, best_loss, early_stop_count = 1, math.inf, 0

    for epoch in range(config['n_epochs']):
        # start train
        train_loss_list, train_acc_list = [],[]
        model.train()
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x) # (batch, classes)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if config['scheduler_flag']:
                scheduler.step()
            step+=1
            loss_per = loss.detach().item()
            acc_per = (pred.argmax(dim=-1) == y).float().mean()
            train_loss_list.append(loss_per)
            train_acc_list.append(acc_per)
            if step % 2 == 0:
                train_logger.info(f'step {add_step+step:03d} train loss {loss_per:3.6f} train acc {acc_per:3.6f}')
        
        # record the training info
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / len(train_acc_list)
        train_logger.info(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epochs']:03d}] TRAIN LOSS {train_loss:3.6f} TRAIN ACC {train_acc:3.6f}")
        print(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epochs']:03d}] TRAIN LOSS {train_loss:3.6f} TRAIN ACC {train_acc:3.6f}")

        # start eval
        model.eval()
        eval_loss_list, eval_acc_list = [], []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                loss_per = loss.detach().item()
                acc_per = (pred.argmax(dim=-1) == y).float().mean()
            eval_loss_list.append(loss_per)
            eval_acc_list.append(acc_per)
            
        # do some records
        eval_loss = sum(eval_loss_list) / len(eval_loss_list)
        eval_acc = sum(eval_acc_list) / len(eval_acc_list)
        eval_logger.info(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epochs']:03d}] VAL LOSS {eval_loss:3.6f} VAL ACC {eval_acc:3.6f}")
        print(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epochs']:03d}] VAL LOSS {eval_loss:3.6f} VAL ACC {eval_acc:3.6f}")

        # save model and early stop
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(config['checkpoints_dir'],'best.pt'))
            print('Saving the best model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if epoch % (config['n_epochs']//10) == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(config['checkpoints_dir'],f'epoch_{epoch:03d}.pt'))
            print(f"Saving the model at epoch {epoch+1:03d} with loss {eval_loss:.3f}")

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so halt the training session.')
            break
        
    torch.save(model.state_dict(), os.path.join(config['checkpoints_dir'],f'last.pt'))
    print(f"saving the last model...")
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        # =============== data folder ===============
        'data_dir': './data/Voxceleb2_Part/',
        # -------------------------------------------

        # ============= recorder folder =============
        'recorder_dir': f'./recorders/{timestamp}',
        'checkpoints_dir': f'./recorders/{timestamp}/checkpoints',
        'logger_dir': f'./recorders/{timestamp}/loggers',
        # -------------------------------------------

        # ============== hyperparameter ==============
        'seed': 654,
        'n_epochs': 800,      
        'batch_size': 2048, 
        'learning_rate': 1e-2,
        'scheduler_flag': True,
        'continue_flag': (False, 200 ,4400, '20250831_144000'),
        'warmup_steps': 1000,
        'early_stop': 100,
        'n_workers': 0,
        # ---------------------------------------------
    }

    makedirs_((config['recorder_dir'], config['checkpoints_dir'], config['logger_dir']))

    all_seed(config['seed'])

    train_logger =create_logger('train_logger', os.path.join(config['logger_dir'], 'train.log'), show_time=False)
    eval_logger = create_logger('eval_logger', os.path.join(config['logger_dir'], 'eval.log'), show_time=False)
    note_logger = create_logger('note_logger', os.path.join(config['logger_dir'], 'note.log'), show_time=True)

    dataset = Voice(config['data_dir'])

    trainlen = int(0.8 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    train_dataset, valid_dataset = random_split(dataset, lengths)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=0, drop_last=True, pin_memory=True, collate_fn=collate_batch)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=0, drop_last=True, pin_memory=True, collate_fn=collate_batch)

    model = Classifier(input_dim=40, d_model=80, n_spks=600, dropout=0.5).to(device)
    if config['continue_flag'][0]:
        model.load_state_dict(torch.load(f"./recorders/{config['continue_flag'][3]}/checkpoints/last.pt",
                                         map_location=device))

    print(f"DEVICE:{device}")
    note_logger.info(f"NOTE: \n[USE] Classifier, \n[DO] dropout(0.5), \n[NO] data enhancement")
    note_logger.info(f"DEVICE:\n{device}")
    note_logger.info(f"MODEL INFO:\n{str(model)}")
    note_logger.info(f"FORWARD INFO:\nInput:(1, 400, 40)\n{str(summary(model, (400, 40)))}")
    note_logger.info(f"CONFIG INFO:\n{pprint.pformat(config, indent=4)}")
    note_logger.info(f"NUM STEPS PER EPOCH (TRAIN):\n{len(train_dataloader)}")

    train(train_dataloader, valid_dataloader, model, config, device, train_logger, eval_logger)