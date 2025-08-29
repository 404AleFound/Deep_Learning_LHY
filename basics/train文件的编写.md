# train文件的编写

深度学习train文件的编写重复性较高，对于这部分应当相当熟悉，如果有必要可以通过默写提高熟练度。


```
# ========================
# 1. import some lib
# ========================

# ========================
# 2. define train function
def train(..., ...):
	...
# ========================

# ========================
# 3. start training
# ========================
	# 3.1
    timestamp = ...
    # 3.2
    device = ...
	# 3.3
    config = {...}
	# 3.4
    all_seed = ...
	# 3.5
    os.makedirs(...)
	# 3.6
    dataset = ...
	# 3.7
    dataloader = ...
	# 3.8
    my_model = ...
   	# 3.9
    my_model.load_state_dict(...)
	# 3.10
    logger = ...
	# 3.11
    train()
```

如上代码块所示，训练文件的编写一般遵循上述框架。对于config字典，需要定义对应的训练超参数。对于train函数，在epoch循环外，需要定义损失函数，优化器相关的设置，以及一些需要记录的参数；在epoch循环内，需要定义五大部分，分别为：训练部分，训练结果的输出和记录，验证部分，验证结果的输出和记录，模型参数的保存策略。



## 导入库文件



## `train`函数

```python
def train(train_loader, valid_loader, model, config, device):
```

如上所示，`device`不应当放在`config`字典中，应当单独定义。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### `epoch`循环中——训练过程部分

```python
train_pbar = tqdm(train_loader) # qt界面显式
model.train() # 开启模型的训练模式
train_loss_epoch, train_acc_epoch = [],[] # 用于记录这个epoch中的训练损失和训练精度
for feature, label in train_pbar:
    optimizer.zero_grad() # 01
    feature, label = feature.to(device), label.to(device) # 02
    pred = model(feature) # 03
    loss = criterion(pred, label) # 04
    loss.backward() # 05
    optimizer.step() # 06
    loss_batch = (...) # 07
    acc_batch = (...) # 07
    train_loss_epoch.append(loss_per) # 08
    train_acc_epoch.append(acc_per) # 08
    step+=1
```

### `epoch`循环中——验证过程部分

```python
model.eval()
valid_pbar = tqdm(valid_loader) # qt界面显式
valid_loss_epoch, valid_acc_epoch = [], [] # 用于记录这个epoch中的验证损失和验证精度
for feature, label in valid_pbar:
    feature, label = x.to(device), y.to(device) # 01
    with torch.no_grad():
        pred = model(feature) # 02
        loss = criterion(pred, label) # 03
        loss_batch = (...) # 04
        acc_batch = (...) # 04
    valid_acc_epoch.append(acc_per) # 05
    valid_loss_epoch.append(loss_per)# 05
```

### `epoch`循环中——日志记录部分

日志的输出应当包含训练过程中每epoch的损失值，以及验证过程中每epoch的损失值，以及该模型的参数和训练时的超参数。文章目录如下所示。

```
logs
|__ %Y%m%d_%H%M%S
	|__ log_train.txt # 记录训练时每epoch的损失值以及其他指标的变化
	|__ log_valid.txt # 记录验证时每epoch的损失值以及其他指标的变化
	|__ log_note.txt # 记录模型的参数和训练的超参数
```

首先，在进入epoch循环之前，先将模型的参数和训练的超参数记录下来。对于模型的参数同时使用`print`方法，和`torchsummary`中的方法输出；对于超参数，则直接打印其字典内容。

```python
with open(config['log_note_path'], 'a', encoding='utf-8') as f:
    f.write(str(model))
    f.write('\n\n')
    
    model_summary = summary(model, (... ,...), verbose=0)
    f.write(str(model_summary))
    f.write('\n\n')
    
    for key, value in config.items():
        f.write(f'{key}: {value}')
        f.write('\n')
```

对该epoch中存储了损失值的列表进行平均值计算，写入日志中，eval部分同样

```python
print(f'||TRAIN INFO|| Step: {step}; Loss/train: {train_loss_mean:5f}; Acc/train: {train_acc_mean:5f}')
with open(config['log_train_path'], 'a') as f:
	f.write(f'Step: {step}; Loss/Train: {train_loss_mean:5f}; Acc/Train: {train_acc_mean:5f}\n')
```

### `epoch`循环中——参数保存策略

```python
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
```

## 开始训练

### 设置时间戳

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### 设置设备

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 编写`config`字典变量

**数据位置**

```python
'dataset_dir': "./data/Voxceleb2_Part/",
```

**训练超参数**

```python
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
```

**日志位置**

```python
'loggers_dir':f'./loggers/{timestamp}'
```

**模型位置**

```python
'checkpoints_dir':f'./checkpoints/{timestamp}'
```

### 设置种子数

```python
all_seed(config['seed'])
```

### 新建文件策略

主要创建`loggers`文件夹和`checkpoints`文件夹，前者存储训练日志，后者存储训练完成的模型权重。

```python
mode=0o755
```

### 创建`dataset`

### 创建`dataloader`

### 创建深度学习模型

### 是否加载已有权重

```python
if config['go_on_train'][0]:
        my_model.load_state_dict(torch.load(f"checkpoints/{config['go_on_train'][3]}/latest_checkpoint.pt",map_location=device))
```

### 创建`logger`

```python
train_logger = create_logger('train_logger', f"{config['loggers_dir']}/train.log", show_time=False)
    val_logger = create_logger('val_logger', f"{config['loggers_dir']}/val.log", show_time=False)
    note_logger = create_logger('note_logger', f"{config['loggers_dir']}/note.log", show_time=True)
```

### 运行`train`
