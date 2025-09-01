# test.py --
# Le Jiang
# 2025/8/25

import torch
import torch.nn as nn
from dataset.dataset import My_Dataset
from torch.utils.data import DataLoader
def test(test_data_loader, model, device, config):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    loss_test_list = []
    with torch.no_grad():
         
        for imgs in test_data_loader:
            imgs = imgs.to(device)# imgs(batch_size, channels, rows, columns)
            outputs = model(imgs)# outputs(batch_size, channels, rows, columns)
            loss = criterion(outputs, imgs).sum([1, 2, 3])
            # loss(batch_size)
            loss_test_list.append(loss)
        anomality = torch.cat(loss_test_list, dim=0)
        anomality = torch.sqrt(anomality).reshape(-1, 1).cpu().numpy()
    
    print(anomality.shape)

if __name__ == '__main__':
    config = {
        'checkpoint_path': './checkpoints/20250825_145359/best_model.pt'
    }
    test_data = My_Dataset('./data/ml2022spring-hw8/testingset.npy')

    test_data_loader = DataLoader(
        test_data,
        batch_size=200,
        num_workers=0,
        shuffle=True,
    )

    model = torch.load(config['checkpoint_path'], weights_only=False)

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    test(test_data_loader, model, device, config)