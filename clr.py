from tensorflow.keras.callbacks import *
import tensorflow as tf
import numpy as np


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has several built-in policies.
    "triangular":
        A basic triangular cycle, no amplitude scaling.
    "triangular-gamma":
        A triangular cycle that scales initial amplitude  (1 / gamma ** x)
        default gamma=2.
    "abs_cosine":
        A periodic cycle by cosine function, starting at top learning rate
    "abs-sine":
        A periodic cycle by sine function, starting at bottom learning rate
    "exp-omega':
        А cycle with exponential drop of amplitude (1 - x ** omega)
        default omega=4.
    "warm-restart"
        A cycle with cosine drop and moderation ratio.
    For more detail on triangular policies, please see paper.

    # Example
        ```python
            clr = CyclicLR(bottom_lr=0.001, top_lr=0.006,
                                iter_per_cycle=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling and profile functions:
        ```python
            clr_scale_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr_profile_fn = lambda x: 1 / x ** 2
            clr = CyclicLR(
                bottom_lr=0.001,
                top_lr=0.006,
                iter_per_cycle=2000.,
                scale_fn=clr_scale_fn,
                profile_fn=clr_profile_fn,
                start_mode='top-start',
                cycle_mode='both-way'
                )
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        bottom_lr: initial learning rate which is the
            lower boundary in the cycle.
        top_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (top_lr - bottom_lr).
            The lr at any cycle is the sum of bottom_lr
            and some scaling of the amplitude; therefore
            top_lr may not actually be reached depending on
            scaling function.
        iter_per_cycle: number of training iterations per cycle.
            Authors suggest 2-8 epochs per cycle.
        mode: one of {triangular, triangular-gamma,
            abs_cosine, restart-exp-omega}.
            Default 'triangular'.
        gamma: constant for 'triangular-gamma' scaling function:
            1 / gamma ** (cycle iterations)
        profile_fn: Custom profile function,
            defines the intra-cycle profile
            lambda x: 1 by default == flat LR
        scale_fn: Custom scaling function,
            defines the overall profile
            lambda x:1 by default == constant top_lr
        start_mode: one of {bottom-start, top-start}
            defines the starting boundary, and movement direction
        cycle_mode: one of {both-way, one-way}
            both-way = back and forth
            one-way = one direction

    MIT License

    Copyright (c) 2017 Bradley Kenstler

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, bottom_lr=0.042, top_lr=0.0042, iter_per_cycle=4242., mode='triangular', profile_fn=None,
                 scale_fn=None, gamma=4., omega=2., start_mode='bottom-start', cycle_mode='both-way', moderator=1):
        super(CyclicLR, self).__init__()

        self.bottom_lr = bottom_lr
        self.top_lr = top_lr
        self.iter_per_cycle = iter_per_cycle
        self.mode = mode
        self.gamma = gamma
        self.start_mode = start_mode
        self.cycle_mode = cycle_mode
        self.moderator = moderator

        if not profile_fn:
            self.profile_fn = lambda x: 1
        else:
            self.profile_fn = profile_fn

        if not scale_fn:
            self.scale_fn = lambda x: 1
        else:
            self.scale_fn = scale_fn

        if self.mode == 'triangular':
            self.profile_fn = lambda x: 1 - x
        elif self.mode == 'triangular-gamma':
            self.profile_fn = lambda x: 1 - x
            self.scale_fn = lambda x: 1 / (gamma ** x)
        elif self.mode == 'abs-sine':
            self.profile_fn = lambda x: abs(np.sin(x * np.pi))
            self.start_mode = 'bottom-start'
            self.cycle_mode = 'one-way'
        elif self.mode == 'exp-omega':
            self.profile_fn = lambda x: 1 - x ** omega
            self.start_mode = 'top-start'
            self.cycle_mode = 'one-way'
        elif self.mode == 'warm-restart':
            self.profile_fn = lambda x: 0.5 + (0.5 * (np.cos(x * np.pi)))
            self.start_mode = 'top-start'
            self.cycle_mode = 'one-way'
            self.moderator = moderator

        self.clr_iterations = 0
        self.restart_iterations = 0
        self.history = {}

        self._reset()

    def _reset(self, new_bottom_lr=None, new_top_lr=None, new_iter_per_cycle=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_bottom_lr:
            self.bottom_lr = new_bottom_lr
        if new_top_lr:
            self.top_lr = new_top_lr
        if new_iter_per_cycle:
            self.iter_per_cycle = new_iter_per_cycle
        self.clr_iterations = self.restart_iterations = self.restart_cycle_number = 0

    def clr(self):
              
        if self.mode == 'warm-restart':
            x_abs = self.restart_iterations / (self.iter_per_cycle * (self.moderator ** self.restart_cycle_number))
            if x_abs >= 1:
                x_abs = 0
                self.restart_iterations = 0
                self.restart_cycle_number += 1
            delta_lr_scaled = (self.top_lr - self.bottom_lr) * self.scale_fn(self.restart_cycle_number)
            return self.bottom_lr + delta_lr_scaled * self.profile_fn(x_abs)

        else:
            cycle_number = np.floor(self.clr_iterations / self.iter_per_cycle)
            x_abs = (self.clr_iterations / self.iter_per_cycle - cycle_number)
            x_symm = np.abs(1 - 2 * x_abs)  # (x_symm => 1..0..1)
            delta_lr_scaled = (self.top_lr - self.bottom_lr) * self.scale_fn(cycle_number)

            if self.cycle_mode == 'both-way' and self.start_mode == 'bottom-start':
                return self.bottom_lr + delta_lr_scaled * self.profile_fn(1 - x_symm)

            elif self.cycle_mode == 'both-way' and self.start_mode == 'top-start':
                return self.bottom_lr + delta_lr_scaled * self.profile_fn(x_symm)

            elif self.cycle_mode == 'one-way':
                return self.bottom_lr + delta_lr_scaled * self.profile_fn(x_abs)

    def on_train_begin(self, logs={}):

        if self.clr_iterations == 0 and self.start_mode == 'bottom-start':
            tf.keras.backend.set_value(self.model.optimizer.lr, self.bottom_lr)
        elif self.clr_iterations == 0 and self.start_mode == 'top-start':
            tf.keras.backend.set_value(self.model.optimizer.lr, self.top_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.clr_iterations += 1
        self.restart_iterations += 1

        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.clr_iterations)

        tf.summary.scalar('learning rate', data=self.model.optimizer.lr, step=self.clr_iterations)

        for iteration, lr in logs.items():
            self.history.setdefault(iteration, []).append(lr)

        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
