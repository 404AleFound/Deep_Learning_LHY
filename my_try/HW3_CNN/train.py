# train.py -- my try for hw3 food images classification
# 2025/8/1
# Le Jiang


# import some needed libs
import torch
import torch.utils as utils
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import gc
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(train_set, val_set, train_dataloader, val_dataloader, config, model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = config['learning_rate'], 
                                 weight_decay=config['weight_decay'])
    best_acc, best_loss = 0.0, 0.0
    early_stop_count = 0
    step = 0
    writer = SummaryWriter()

    for epoch in range(config['num_epoch']):
        # prepare some indexs
        train_loss, train_acc = 0.0, 0.0
        val_acc, val_loss = 0.0, 0.0
        
        # strat training

        model.train()
        train_loss_epochs, train_acc_epochs = [], []
        for imgs, labels in tqdm(train_dataloader, position=0, leave=True):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            if config['clip_flag']:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                
            optimizer.step()
            _, train_pred = torch.max(outputs, dim=1) # find the max val along the line
            # .detach()/==/.sum()/.item()
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
        
        # start valiation
        if (len(val_set) > 0):
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader)):
                    imgs, labels = batch
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    _, val_pred = torch.max(outputs, dim = 1)
                    val_acc += (val_pred == labels).sum().item()
                    val_loss += loss.item()
        
        # print/record the info of one epoch
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, config['num_epoch'], 
            train_acc / len(train_set), train_loss / len(train_dataloader), 
            val_acc / len(val_set),     val_loss / len(val_dataloader)))
        writer.add_scalar('Train Acc', train_acc / len(train_set), epoch)
        writer.add_scalar('Train Loss', train_loss / len(train_dataloader), epoch)
        writer.add_scalar('Val Acc', val_acc / len(val_set), epoch)
        writer.add_scalar('Val Loss', val_loss / len(val_dataloader), epoch)

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'],'best_checkpoint.ckpt'))
            print('saving the best model with val acc: {:3.6f}'.format(best_acc/len(val_set)))
            early_stop_counts = 0
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                    epoch + 1, config['num_epoch'], 
                    train_acc / len(train_set), train_loss / len(train_dataloader)))
            early_stop_count += 1

        # early stop
        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so halt the training session.')
            
            
        # save the latest model for belowing training
        if epoch % 5 == 0:        
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'latest_checkpoint.ckpt'))
            print('saving the latest model with val acc: {:3.6f}'.format(val_acc/len(val_set)))

    # save the model of the last epoch if there is no valiation dataset
    if len(val_set) == 0:
        torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_checkpoint.ckpt'))
        print('saving model at last epoch')

    # release the memory
    del train_dataloader, val_dataloader
    gc.collect()
    writer.close()
    return