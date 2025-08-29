# train.py -- my try for hw3 food images classification
# 2025/8/1
# Le Jiang

# import some needed libs
import torch
import torch.utils as utils
import torch.nn as nn
import gc
import os
from tqdm import tqdm
import os
from dataset.process import read_images
from dataset.dataset import Food_Dataset
from models import ResNet, BaseNet
from utils import same_seeds, all_seed, create_logger, get_cosine_schedule_with_warmup
from datetime import datetime
from torchinfo import summary
import pprint

def train(train_dataloader, val_dataloader, config, model, device, logger_train, logger_val):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = config['lr'], 
                                 weight_decay=config['weight_decay'])

    if config['go_on_train'][0]:
        add_epoch, add_step = config['go_on_train'][1], config['go_on_train'][2]
    else:
        add_epoch, add_step = 0, 0
    
    if config['scheduler_flag']:
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    config['warmup_steps'], 
                                                    len(train_dataloader) * config['n_epoch'])
    
    best_acc = 0.0
    early_stop_count = 0
    step = 1

    for epoch in range(config['n_epoch']):

        # strat training
        model.train()
        train_loss_list, train_acc_list = [], []
        for imgs, labels, _ in tqdm(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 稳定训练的技巧
            if config['clip_flag']:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            if config['scheduler_flag']:
                scheduler.step()
            _, train_pred = torch.max(outputs, dim=1) # find the biggest value along the line
            train_acc_list.append((train_pred.detach() == labels.detach()).sum().item())
            train_loss_list.append(loss.item())
            if step % 10 == 0:
                logger_train.info(f'step {add_step+step:03d} train loss {loss.item():3.6f} train acc {((train_pred.detach() == labels.detach()).sum().item()) / len(imgs):3.6f}')
            step += 1
        
        # record the training info
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / (config['batch_size'] * len(train_acc_list))
        logger_train.info(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epoch']:03d}] TRAIN LOSS {train_loss:3.6f} TRAIN ACC {train_acc:3.6f}")
        print(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epoch']:03d}] TRAIN LOSS {train_loss:3.6f} TRAIN ACC {train_acc:3.6f}")


        # start valiation
        model.eval()
        val_acc_list, val_loss_list = [], []
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_dataloader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, dim = 1)
                val_acc_list.append((val_pred.detach() == labels.detach()).sum().item())
                val_loss_list.append(loss.item())
        
        # record the val info
        val_loss = sum(val_loss_list) / len(val_loss_list)
        val_acc = sum(val_acc_list) / (config['batch_size'] * len(val_acc_list))
        logger_val.info(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epoch']:03d}] VAL LOSS {val_loss:3.6f} VAL ACC {val_acc:3.6f}")
        print(f"EPOCH [{add_epoch+epoch+1:03d}|{add_epoch+config['n_epoch']:03d}] VAL LOSS {val_loss:3.6f} VAL ACC {val_acc:3.6f}")

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config['checkpoints_dir']}/best_checkpoint.pt")
            print(f'saving the best model with val acc: {best_acc:3.6f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        # early stop
        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so halt the training session.')
            break
            
    # save the latest model for belowing training     
    torch.save(model.state_dict(), f"{config['checkpoints_dir']}/latest_checkpoint.pt")
    print(f"saving the latest model...")



if __name__ == '__main__':
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    config = {
        # =============================================
        'data_dir':'./data/food11',
        'checkpoints_dir':f'./checkpoints/{timestamp}',
        'loggers_dir':f'./loggers/{timestamp}',
        # =============================================
        'seed': 6666,
        'n_epoch': 250,
        'batch_size': 128,
        'lr':0.0001,
        'weight_decay':5e-4,
        'early_stop': 50,
        'num_workers': 0,
        'clip_flag': True,
        'go_on_train':(False, 436, 33570, '20250828_203602'),
        'scheduler_flag':True,
        'warmup_steps':1000
        # =============================================
    }

    all_seed(config['seed'])

    if not os.path.exists(config['checkpoints_dir']):
        os.makedirs(config['checkpoints_dir'], mode=0o755)

    if not os.path.exists(config['loggers_dir']):
        os.makedirs(config['loggers_dir'], mode=0o755)

    train_images_list, train_labels_list, nimages_list = read_images('./data/food11/', 'training')
    val_images_list, val_labels_list, val_list = read_images('./data/food11/', 'validation')

    train_set = Food_Dataset(train_images_list, train_labels_list, nimages_list, 'training', (128,128))
    val_set = Food_Dataset(val_images_list, val_labels_list, val_list, 'validation', (128, 128))

    del train_images_list, train_labels_list, nimages_list
    del val_images_list, val_labels_list, val_list

    train_dataloader = utils.data.DataLoader(train_set, batch_size=config['batch_size'], 
                                            shuffle=True, num_workers=config['num_workers'], 
                                            pin_memory=True, drop_last=True)
    val_dataloader = utils.data.DataLoader(val_set, batch_size=config['batch_size'], 
                                            shuffle=True, num_workers=config['num_workers'], 
                                            pin_memory=True, drop_last=True)

    print(f"DEVICE:{device}")

    my_model = BaseNet().to(device)
    if config['go_on_train'][0]:
        my_model.load_state_dict(torch.load(f"checkpoints/{config['go_on_train'][3]}/latest_checkpoint.pt",
                                             map_location=device))
    
    train_logger = create_logger('train_logger', f"{config['loggers_dir']}/train.log", show_time=False)
    val_logger = create_logger('val_logger', f"{config['loggers_dir']}/val.log", show_time=False)
    note_logger = create_logger('note_logger', f"{config['loggers_dir']}/note.log", show_time=True)

    note_logger.info(f"NOTE: \n[USE] BaseNet, \n[NO] dropout(0.5), \n[DO] data enhancement")
    note_logger.info(f"DEVICE:\n{device}")
    note_logger.info(f"MODEL INFO:\n{str(my_model)}")
    note_logger.info(f"FORWARD INFO:\nInput:(1, 3, 128, 128)\n{str(summary(my_model, (1, 3, 128, 128)))}")
    note_logger.info(f"CONFIG INFO:\n{pprint.pformat(config, indent=4)}")
    note_logger.info(f"NUM STEPS PER EPOCH (TRAIN):\n{len(train_dataloader)}")

    train(train_dataloader, val_dataloader, config, my_model, device, train_logger, val_logger)