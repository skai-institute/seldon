from typing import List, Optional

import numpy as np
from torch import optim

# CLIP_LAM_MAX = 1e8
CLIP_LAM_MAX = 1e4


class AnnealingLinearScheduler:
    def __init__(
        self,
        start_step: int,
        annealing_reset: List[int],
        annealing_rate: float,
        start_beta: float = 0.0,
    ) -> None:
        self.start_step = start_step
        self.annealing_reset = annealing_reset
        self.annealing_rate = annealing_rate
        self.start_beta = start_beta

        self.beta = start_beta
        self.counter_steps = 0
        self.counter_times = 0

    def __call__(self) -> float:
        self.counter_steps += 1
        if self.counter_steps > self.start_step:
            if (
                len(self.annealing_reset) > self.counter_times
                and self.counter_steps == self.annealing_reset[self.counter_times]
            ):
                self.counter_times += 1
                self.beta = self.start_beta
            else:
                self.beta = min(1.0, self.beta + self.annealing_rate)
        return self.beta


class ConstantScheduler:
    def __init__(
        self,
        lam: float,
    ) -> None:

        self.lam = lam

    def __call__(self, loss: float) -> float:

        return self.lam


class ConstrainedExponentialSchedulerMaLagrange:
    def __init__(
        self,
        constraint_bound: float,
        annealing_rate: float,
        start_lam: float = 1.0,
        alpha: float = 0.5,
        lower_bound_lam: float = 0.0,
        adapt_after_first_satisfied: bool = False,
        clip_lam_max: float = 1e4,
    ) -> None:
        self.constraint_bound = constraint_bound
        self.annealing_rate = annealing_rate
        self.start_lam = start_lam
        self.lam = start_lam
        self.alpha = alpha
        self.clip_lam_max = clip_lam_max

        self.constraint_ma: Optional[float] = None

        self.adapt_after_first_satisfied = adapt_after_first_satisfied
        self.constraint_first_satisfied = False
        self.current_constraint_satisfied = False

        self.lower_bound_lam = lower_bound_lam
        self.gamma = self.lam - self.lower_bound_lam

    def update_lam(self):
        # print(self.constraint_ma, self.annealing_rate*self.constraint_ma)
        self.gamma = self.gamma * np.exp(self.annealing_rate * self.constraint_ma)

    def __call__(self, loss: float) -> float:
        constraint = loss - self.constraint_bound
        # TODO: if we do not use constraint_fulfilled, we can rm the following
        if self.adapt_after_first_satisfied:
            if not self.constraint_first_satisfied and constraint < 0:
                self.constraint_first_satisfied = True
        # print(self.constraint_bound)
        if constraint < 0:
            self.current_constraint_satisfied = True
        else:
            self.current_constraint_satisfied = False

        # moving average
        if self.constraint_ma is None:
            self.constraint_ma = constraint
        else:
            self.constraint_ma = (
                1 - self.alpha
            ) * self.constraint_ma + self.alpha * constraint

        if self.adapt_after_first_satisfied:
            if self.constraint_first_satisfied:
                self.update_lam()
        else:
            self.update_lam()

        # TODO: should self.gamma be clipped?
        # if self.gamma > self.clip_lam_max:
        #    print(self.gamma)
        self.gamma = float(
            np.clip(self.gamma, 0.0 - self.lower_bound_lam, self.clip_lam_max)
        )

        self.lam = self.gamma + self.lower_bound_lam

        return self.lam
