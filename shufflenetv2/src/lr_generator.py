# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""learning rate exponential decay generator"""
import math
import numpy as np


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """Cosine decay learning rate with optional warmup (MobileNetV2-style)."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if warmup_steps > 0 and i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            if total_steps == warmup_steps:
                lr = lr_end
            else:
                lr = lr_end + (lr_max - lr_end) * (1.0 + math.cos(
                    math.pi * (i - warmup_steps) / (total_steps - warmup_steps)
                )) / 2.0
        lr_each_step.append(max(lr, 0.0))

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step[global_step:]


def get_lr(lr_init, lr_decay_rate, num_epoch_per_decay, total_epochs, steps_per_epoch, is_stair=False):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_decay_rate (float):
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       is_stair(bool): If `True` decay the learning rate at discrete intervals (default=False)

    Returns:
       learning_rate, learning rate numpy array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_steps = steps_per_epoch * num_epoch_per_decay
    for i in range(total_steps):
        p = i/decay_steps
        if is_stair:
            p = math.floor(p)
        lr_each_step.append(lr_init * math.pow(lr_decay_rate, p))
    learning_rate = np.array(lr_each_step).astype(np.float32)
    return learning_rate

def get_lr_basic(lr_init, total_epochs, steps_per_epoch, is_stair=False):
    """
    generate basic learning rate array

    Args:
       lr_init(float): init learning rate
       total_epochs(int): total epochs of training
       steps_per_epoch(int): steps of one epoch
       is_stair(bool): If `True` decay the learning rate at discrete intervals (default=False)

    Returns:
       learning_rate, learning rate numpy array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        lr = lr_init - lr_init * (i) / (total_steps)
        lr_each_step.append(lr)
    learning_rate = np.array(lr_each_step).astype(np.float32)
    return learning_rate
