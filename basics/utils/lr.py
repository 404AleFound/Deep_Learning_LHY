# lr.py -- learning rate decay method
# Le Jiang
# 2025/8/29

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(
    opt: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    创建一个学习率变化策略,
    学习率跟随cosine值变化,
    在warm up时间段内变化区间在:
        0 -> 优化器设置的学习率 .
    Args:
        opt (Optimizer): 优化器类
        num_warmup_steps (int): 多少步增加一下lr
        num_training_steps (int): 总训练步骤
        num_cycles (float, optional): 变化周期. 默认为 0.5.
        last_epoch (int, optional): _description_. Defaults to -1.
    """
    def lr_lambda(current_step):
        # warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 衰减
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(opt, lr_lambda, last_epoch)