# test.py -- 
# Le Jiang 
# 2025/8/27

# import some needed libs
import torch
import torch.utils as utils
import os
from tqdm import tqdm
import os
from dataset.process import read_images
from dataset.dataset import Food_Dataset
from models import ResNet, BaseNet 
import numpy as np
import matplotlib.pyplot as plt 
import random
from PIL import Image

timestamp = '20250828_203602'

labels2class = {
    '0': 'Bread',
    '1': 'Dairy product',
    '2':'Dessert', 
    '3':'Egg', 
    '4':'Fried food', 
    '5':'Meat', 
    '6':'Noodles/Pasta', 
    '7':'Rice', 
    '8':'Seafood', 
    '9':'Soup', 
    '10':'Vegetable/Fruit',
}

config = {
    # =============================================
    'data_dir':'./data/food11',
    'checkpoints_dir':f'checkpoints/{timestamp}',
    'loggers_dir':f'./loggers/{timestamp}',
    'batch_size':128,
    'num_workers':0,
}

def show(test_path_list, test_predict_list):
    
    selected_indices = random.sample(range(len(test_predict_list)), min(25, len(test_predict_list)))
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    
    for i, idx in enumerate(selected_indices):
        row = i // 5
        col = i % 5

        img = Image.open(os.path.join(f"{config['data_dir']}/test", test_path_list[idx]))
        pred = test_predict_list[idx]
        
        if img is not None:
            img = img = np.array(img) 
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

        axes[row, col].imshow(img)
        
        title = f"Class: {labels2class[str(pred)]}"
        
        axes[row, col].set_title(title, fontsize=9, pad=5)
        axes[row, col].axis('off')
    
    for i in range(len(selected_indices), 25):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_model = BaseNet().to(device)
    my_model.load_state_dict(torch.load(os.path.join(config['checkpoints_dir'],'best_checkpoint.pt'), map_location=device))

    test_images_list, test_labels_list, test_nimages_list = read_images(config['data_dir'], 'test',)

    test_set = Food_Dataset(test_images_list, test_labels_list,test_nimages_list, 
                            tag='test', reshape_size=(128,128))

    del test_images_list, test_labels_list, test_nimages_list

    test_dataloader = utils.data.DataLoader(test_set, batch_size=config['batch_size'], 
                                            shuffle=False, num_workers=config['num_workers'], 
                                            pin_memory=True)


    my_model.eval()

    test_path_list, test_predict_list = [], []
    with torch.no_grad():
        for imgs, _, nimgs in tqdm(test_dataloader):
            imgs = imgs.to(device)
            outputs = my_model(imgs)# (n, 11)
            _, pred = torch.max(outputs, dim = 1)# (n, 1)
            pred = pred.cpu().data.numpy()
            test_path_list.extend(nimgs)
            test_predict_list.extend(list(pred))


    show(test_path_list, test_predict_list)