# train01_template.py --- 
# 2025/9/1
# Le Jiang

# import some needed libs
import torch
import torch.utils as utils
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime
import pprint
# =======================================================================================
from dataset import ...
from models import ...
from utils import ...


def train(train_dataloader, val_dataloader, config, model, device, logger_train, logger_val):
    criterion = ____
    optimizer = ____

    if config['continue_info'][0]:
        add_epoch, add_step = ____
    else:
        add_epoch, add_step = 0, 0
    
    if config['scheduler_info'][0]:
        scheduler = ____
    
    step, best_val_acc = 1, 0

    for epoch in range(config['n_epochs']):

        # strat training
        model.train()
        train_loss_list, train_acc_list = [], []
        for features, labels in tqdm(train_dataloader):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            pred = ____
            loss.backward()
            optimizer.step()
            if config['scheduler_info'][0]:
                scheduler.step()
            loss_per, acc_per = ____, ____
            train_acc_list.append(acc_per)
            train_loss_list.append(loss_per)
            if ____:
                logger_train.info(____)
            step += 1
        
        # record the training info
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / len(train_acc_list)
        logger_train.info(f"____")
        print(f"____")


        # start valiation
        model.eval()
        val_acc_list, val_loss_list = [], []
        with torch.no_grad():
            for features, labels in tqdm(val_dataloader):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                pred = ____
                loss_per = ____
                acc_per = ____
                val_acc_list.append(loss_per)
                val_loss_list.append(acc_per)
        
        # record the val info
        val_loss = sum(val_loss_list) / len(val_loss_list)
        val_acc = sum(val_acc_list) / len(val_acc_list)
        logger_val.info(f"____")
        print(f"____")

        # save the best model
        if val_acc > best_val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config['checkpoints_dir']}/best.pt")
            print(f'____')
            early_stop_count = 0
        else:
            early_stop_count += 1

        # early stop
        if early_stop_count >= config['early_stop_info'][...]:
            print('\nModel is not improving, so halt the training session.')
            break
            
    # save the latest model for belowing training     
    torch.save(model.state_dict(), f"{config['checkpoints_dir']}/last.pt")
    print(f"saving the last model...")



if __name__ == '__main__':
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    config = {
        # ======================== File Path ========================
        'data_dir': ____,
        'recorders_dir': ____,
        'checkpoints_dir': ____,
        'loggers_dir': ____, 
        # ----------------------------------------------------------

        # ================== Basic Hyperparameter ===================
        'seed': ____,
        'n_epochs': ____,
        'batch_size': ____,
        'lr': ____,
        'num_workers': ____,
        # ----------------------------------------------------------

        # ==================== Training Skills ======================
        'weight_decay_info': (True/False, ...),
        'early_stop_info': (True/False, ...),
        'scheduler_info': (True/False, ...),
        'continue_info': (True/False, ...),
        'warmup_steps_info': (True/False, ...),
        # ------------------------------------------------------------
    }

    # TODO
    ''' SET THE SEEDS '''

    # TODO
    ''' MAKING SOME DIRS'''

    # TODO
    ''' CREATING DATASET NEEDED'''
    dtaset = ____

    # TODO
    ''' CREATING DATALOADER NEEDED'''
    dataloder = ____


    # TODO
    ''' CREATING DL MODELS '''
    my_model = ____

    # TODO
    ''' IF NEED TO CONTIUNUE TRAINING, LOAD THE EXISTED CHECKPOINTS '''
    if config['continue_info'][0]:
        ...

    # TODO   
    ''' CREATING NOTELOGGER, TRAINLOGGER, EVALLOGGER AND ETC. '''
    note_logger = ____
    ...
    

    # TODO 
    ''' SOME EXPLANATION BEFORE TRAINING '''
    note_logger.info(f"NOTE:")
    note_logger.info(f"DEVICE:\n{device}")
    note_logger.info(f"MODEL INFO:\n{str(____)}")
    note_logger.info(f"FORWARD INFO:\nInput:____\n{str(summary(my_model, ____))}")
    note_logger.info(f"CONFIG INFO:\n{pprint.pformat(config, indent=4)}")
    note_logger.info(f"NUM STEPS PER EPOCH (TRAIN):\n{len(____)}")

    # TODO
    ''' RUN THE TRAIN FUNCTION'''
    train(...)