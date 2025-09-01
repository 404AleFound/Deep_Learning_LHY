# attack.py --
# Le Jiang
# 23025/9/1

import torch

def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # 用良性图片初始化 x_adv
    x_adv.requires_grad = True # 需要获取 x_adv 的梯度
    loss = loss_fn(model(x_adv), y) # 计算损失
    loss.backward()  
    # fgsm: 在x_adv上使用梯度上升来最大化损失
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


# 在“全局设置”部分中将alpha设置为步长 
# alpha和num_iter可以自己决定设定成何值
alpha = 0.8 / 255 / std
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    # num_iter 次迭代
    for i in range(num_iter):
        x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # 用（ε=α）调用fgsm以获得新的x_adv
        # x_adv = x_adv.detach().clone()
        # x_adv.requires_grad = True  
        # loss = loss_fn(model(x_adv), y)  
        # loss.backward()  
        # grad = x_adv.grad.detach()
        # x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # x_adv 裁剪到 [x-epsilon, x+epsilon]范围
    return x_adv


def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=1.0):
    x_adv = x
    # 初始化 momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # num_iter 次迭代
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True  
        loss = loss_fn(model(x_adv), y)  
        loss.backward()  
        # TODO: Momentum calculation
        grad = x_adv.grad.detach() + (1 - decay) * momentum
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # x_adv 裁剪到 [x-epsilon, x+epsilon]范围
    return x_adv