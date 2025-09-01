# eval.py --
# Le Jiang 
# 2025/9/1


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    acc_list, loss_list = [] ,[]
    for datas, labels in data_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        pred = model(datas)
        loss = loss_fn(datas, labels)
        acc_list.append((pred.argmax(dim=1) == labels).sum().item())
        loss_list.append(loss.item().mean())


