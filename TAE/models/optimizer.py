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
        encoder_scale_factor=1,
        reduce_on_plateau=False,
    ):
        self.lr = LR
        self.weight_decay = weight_decay
        self.encoder_scale_factor = encoder_scale_factor
        self.reduce_on_plateau = reduce_on_plateau

    def __call__(self, model):
        optims = []
        if self.encoder_scale_factor == 1:
            optims = optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.encoder_scale_factor != 1:
            encoder_parameters = [p for p in model.encoder.parameters()]
            encoder_parameter_ids = [id(p) for p in encoder_parameters]
            decoder_non_decoupled_params = [p for p in model.decoder.to_params.decoder.decoder.parameters()]
            decoder_non_decoupled_param_ids = [id(p) for p in model.decoder.to_params.decoder.decoder.parameters()]
            other_parameters = [p for p in model.parameters() if id(p) not in encoder_parameter_ids and id(p) not in decoder_non_decoupled_param_ids]
            
            optims = optim.Adam(
                [
                    {"params":other_parameters, "lr":self.lr},
                    {"params":encoder_parameters, "lr":self.lr*self.encoder_scale_factor, 'weight_decay':self.weight_decay},
                    {"params":decoder_non_decoupled_params, 'weight_decay':self.weight_decay}
                ],
                weight_decay=0.0,#,self.weight_decay,
                lr=self.lr
            )

        
        if self.reduce_on_plateau:
            #scheduler = optim.lr_scheduler.ExponentialLR(optims, gamma=0.98)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims, cooldown=2, factor=0.2)
            return [optims], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}]
        return optims

        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims, cooldown=5, factor=0.5)
        #return [optims], [{"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}]
        # if self.scheduler_gamma is not None:
        #     scheduler = optim.lr_scheduler.ExponentialLR(
        #         optims[0], gamma=self.scheduler_gamma
        #     )
        #     scheds.append(scheduler)
        #     return optims, scheds
        # else:
        #     return optims
