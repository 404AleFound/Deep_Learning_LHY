# train.py -- load the pretrained bert model and train
# Le Jiang
# 2025/8/20

from torch.utils.data import DataLoader
from dataset.dataset import My_Dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast
from torch.optim import AdamW
import torch 
from tqdm import tqdm
from datetime import datetime
import os
from utils import all_seed, create_logger, get_cosine_schedule_with_warmup

def train(train_dataloader, eval_dataloader, model, config, device, train_logger, val_logger):
    
    optimizer = AdamW(model.parameters(), lr = config['lr'])
    if config['schedule_flag']:
    step = 0
    train_loss_epoch, train_acc_epoch = 0, 0 
    eval_acc_epoch = 0

    for epoch in range(config['n_epoch']):

        # 01 train the model 
        model.train()
        acc_train, loss_train = [], []
        for data in tqdm(train_dataloader):
            # input(batch_size, len_seq), input_type(batch_size, len_seq), 
            # input_mask(batch_size, len_seq), begin(batch_size, 1), end(batch_size, 1)
            data = [d.to(device) for d in data]
            input_ids, token_type_ids, attention_mask, start_positions, end_positions = data
            output = model(input_ids=input_ids, 
                           token_type_ids=token_type_ids, 
                           attention_mask=attention_mask, 
                           start_positions=start_positions, 
                           end_positions=end_positions)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            start = torch.argmax(output.start_logits, dim=1)
            end = torch.argmax(output.end_logits, dim=1)
            acc_train.append(((start == start_positions) == (end == end_positions)).float().mean())
            loss_train.append(output.loss)
            train_logger.info(f'Epoch {epoch} Step{step} Loss{loss_train} Acc{acc_train}')
            step += 1

        # 02 print and eval 
        train_acc_epoch = sum(acc_train) / len(acc_train)
        train_loss_epoch = sum(loss_train) / len(loss_train)
        train_logger.info(f'||Epoch {epoch}|| Train Loss: {train_loss_epoch:5f} Train Acc: {train_acc_epoch:5f}')
        print(f'||Epoch {epoch}|| Train Loss: {train_loss_epoch:5f} Train Acc: {train_acc_epoch:5f}')

        # 03 eval the model 
        model.eval()
        for data in tqdm(eval_dataloader):# at this moment, batch_size is 1
            #input_list, input_type_list, input_mask_list, answer_token
            predict, max_prob, acc_eval = '', 0, []
            data = [d.to(device) for d in data]
            input_ids, token_type_ids, attention_mask, answer_text = data
            with torch.no_grad():
                # data[i].squeeze(dim=0) (n_windows, seq_len)
                output = model(
                    input_ids=input_ids.squeeze(dim=0), 
                    token_type_ids=token_type_ids.squeeze(dim=0),
                    attention_mask=attention_mask.squeeze(dim=0))
                output.loss()
                for k in range(data[0].shape[1]):#data[i](n_batch, n_windows, seq_len)
                    start_prob, start_index = torch.max(output.start_logits[k], dim=0)
                    end_prob, end_index = torch.max(output.end_logits[k], dim=0)
                    prob = start_prob + end_prob
                    if prob > max_prob:
                        max_prob = prob
                        predict = tokenizer.decode(input_ids[0][k][start_index:end_index + 1])
                acc_eval.append(answer_text == predict)

        # 04 print and record
        eval_acc_epoch = sum(acc_eval) / len(acc_eval) 
        eval_logger.info(f'||Epoch {epoch}|| Eval Acc: {eval_acc_epoch:5f}')
        print(f'||Epoch {epoch}|| Eval Acc: {eval_acc_epoch:5f}')

        # 05 save the checkpoints
        torch.save(model.state_dict(), os.path.join(config['checkpoints_dir'], f'{epoch+1:03d}.pth'))
            

if __name__ == '__main__':

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        # ======================================================
        'data_train':'.data/ml2022spring-hw7/hw7_train.json',
        'data_eval':'data/ml2022spring-hw7/hw7_dev.json',
        'data_test':'data/ml2022spring-hw7/hw7_test.json',
        'log_dir':f'./logs{timestamp}',
        'checkpoints_dir':f'./checkpoints/{timestamp}',
        # ======================================================
        'lr':1e-4,
        'seed':6666,
        'n_epoch':10,
        'batch_size':32,
        'weight_decay':1e-4,
        'warm_steps':1000,
        'schedule_flag':True,
        'continue_flag':(False, 0, 0),
        'clip_flag':False
        # ======================================================
    }

    if not os.path.exists(config['log_dir']):
        os.mkdir(config['log_dir'], mode=0o755)
    
    if not os.path.exists(config['checkpoints_dir']):
        os.mkdir(config['checkpoints_dir'], mode=0o755)

    train_logger = create_logger('train_logger', os.path.join(config['log_dir'], 'train.log'))
    eval_logger = create_logger('eval_logger', os.path.join(config['log_dir'], 'eval.log'))
    note_logger = create_logger('note_logger', os.path.join(config['log_dir'], 'note.log'))

    train_dataset = My_Dataset('./data/ml2022spring-hw7/hw7_train.json', 'train')
    eval_dataset = My_Dataset('./data/ml2022spring-hw7/hw7_dev.json', 'eval')
    test_dataset = My_Dataset('./data/ml2022spring-hw7/hw7_test.json', 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                  shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)

    train(train_dataloader, eval_dataloader, model, config, device, train_logger, eval_logger)