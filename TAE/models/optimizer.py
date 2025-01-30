import typing

import torch_optimizer as torch_optim
from torch import optim

# from models.scheduler import CosineWarmupScheduler


class RAdam:
    def __init__(
        self,
        LR: float,
        betas: typing.Tuple[float, float] = (0.9, 0.99999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        self.lr = LR
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def __call__(self, model):
        optimizer = torch_optim.RAdam(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        return optimizer


class Ranger:
    def __init__(
        self,
        LR: float = 5e-4,
        alpha: float = 0.1,
        k: int = 6,
        N_sma_threshhold: int = 3,
        betas: typing.Tuple[float, float] = (0.9, 0.99999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        self.lr = LR
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.k = k
        self.N_sma_threshhold = N_sma_threshhold

    def __call__(self, model):
        optimizer = torch_optim.Ranger(
            model.parameters(),
            lr=self.lr,
            alpha=self.alpha,
            k=self.k,
            N_sma_threshhold=self.N_sma_threshhold,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        return optimizer


# class Adam_WU:
#     def __init__(
#         self,
#         LR,
#         weight_decay=0,
#     ):
#         self.lr = LR
#         self.weight_decay = weight_decay

#     def __call__(self, model):
#         optimizer = optim.Adam(
#             model.parameters(),
#             lr=self.lr,
#             weight_decay=self.weight_decay,
#         )
#         self.lr_scheduler = CosineWarmupScheduler(
#             optimizer=optimizer, warmup=200, max_iters=1000
#         )
#         return (optimizer,)


class Adam:
    def __init__(
        self,
        LR,
        weight_decay=0,
        scheduler_gamma=None,
    ):
        self.lr = LR
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

    def __call__(self, model):
        optims = []
        optims = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optims
        # if self.scheduler_gamma is not None:
        #     scheduler = optim.lr_scheduler.ExponentialLR(
        #         optims[0], gamma=self.scheduler_gamma
        #     )
        #     scheds.append(scheduler)
        #     return optims, scheds
        # else:
        #     return optims
