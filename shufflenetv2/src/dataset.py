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
"""Data operations, used in train.py and eval.py."""

import os

import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C

from src.config import config_gpu as cfg


def create_dataset(dataset_root, do_train, rank, group_size, config=cfg, repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """
    subset = "train" if do_train else "eval"
    dataset_path = os.path.join(dataset_root, subset)

    if group_size == 1:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=config.work_nums, shuffle=True)
    else:
        ds = de.ImageFolderDataset(
            dataset_path,
            num_parallel_workers=config.work_nums,
            shuffle=True,
            num_shards=group_size,
            shard_id=rank,
        )

    resize_height = config.image_height
    resize_width = config.image_width

    decode_op = C.Decode()
    resize_op = C.Resize((resize_height, resize_width))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)
    color_adjust_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    change_swap_op = C.HWC2CHW()

    if do_train:
        batch_size = config.batch_size
        trans = [decode_op, resize_op, horizontal_flip_op, color_adjust_op, normalize_op, change_swap_op]
    else:
        batch_size = 1
        trans = [decode_op, resize_op, normalize_op, change_swap_op]

    type_cast_image = C2.TypeCast(mstype.float32)
    type_cast_label = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", operations=trans + [type_cast_image], num_parallel_workers=config.work_nums)
    ds = ds.map(input_columns="label", operations=type_cast_label, num_parallel_workers=config.work_nums)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_num)
    return ds
