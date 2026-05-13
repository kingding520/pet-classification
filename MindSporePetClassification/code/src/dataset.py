"""
create train or eval dataset.
"""
import os
import numpy as np
import multiprocessing

from mindspore import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms


def create_dataset(dataset_path, do_train, config, repeat_num=1):
    """
    create a train or eval dataset
    """
    cores = max(min(multiprocessing.cpu_count(), 8), 1)

    dataset = ds.ImageFolderDataset(
        dataset_path,
        num_parallel_workers=cores,
        shuffle=True
    )

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # ======================
    # 图像增强（新版API）
    # ======================
    decode_op = vision.Decode()

    resize_crop_op = vision.RandomResizedCrop(
        size=(resize_height, resize_width),
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333)
    )

    horizontal_flip_op = vision.RandomHorizontalFlip(prob=0.5)

    resize_op = vision.Resize((resize_height, resize_width))

    color_jitter_op = vision.RandomColorAdjust(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4
    )

    normalize_op = vision.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    hwc2chw_op = vision.HWC2CHW()

    # ======================
    # 训练 / 测试区别
    # ======================
    if do_train:
        batch_size = config.batch_size
        trans = [
            decode_op,
            resize_crop_op,
            horizontal_flip_op,
            color_jitter_op,
            normalize_op,
            hwc2chw_op
        ]
    else:
        batch_size = 1
        trans = [
            decode_op,
            resize_op,
            normalize_op,
            hwc2chw_op
        ]

    type_cast_op = transforms.TypeCast(mstype.int32)

    # map
    dataset = dataset.map(
        operations=trans,
        input_columns="image",
        num_parallel_workers=cores
    )

    dataset = dataset.map(
        operations=type_cast_op,
        input_columns="label",
        num_parallel_workers=cores
    )

    # shuffle
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # repeat
    dataset = dataset.repeat(repeat_num)

    return dataset


def extract_features(net, dataset_path, config):
    print("start cache feature!")

    features_folder = os.path.join(dataset_path, "features")

    train_dataset = create_dataset(
        dataset_path=os.path.join(dataset_path, "train"),
        do_train=True,
        config=config
    )

    eval_dataset = create_dataset(
        dataset_path=os.path.join(dataset_path, "eval"),
        do_train=False,
        config=config
    )

    train_size = train_dataset.get_dataset_size()
    eval_size = eval_dataset.get_dataset_size()

    if train_size == 0:
        raise ValueError(
            "dataset size is 0. Check dataset or batch_size."
        )

    # ======================
    # 如果缓存已存在
    # ======================
    if os.path.exists(features_folder):
        train_features = np.load(os.path.join(features_folder, "train_feature.npy"))
        train_labels = np.load(os.path.join(features_folder, "train_label.npy"))
        eval_features = np.load(os.path.join(features_folder, "eval_feature.npy"))
        eval_labels = np.load(os.path.join(features_folder, "eval_label.npy"))

        return (train_features, train_labels, eval_features, eval_labels), train_size

    os.mkdir(features_folder)

    model = Model(net)

    # ======================
    # train feature
    # ======================
    train_feature = []
    train_labels = []

    train_imgs = train_size * config.batch_size

    for i, data in enumerate(train_dataset.create_dict_iterator()):
        image = data["image"]
        label = data["label"]

        feature = model.predict(Tensor(image))

        train_feature.append(feature.asnumpy())
        train_labels.append(label.asnumpy())

        percent = round(i / train_size * 100, 2)
        print(
            f'train feature [{i * config.batch_size}/{train_imgs}] {percent}%',
            end='\r',
            flush=True
        )

    np.save(os.path.join(features_folder, "train_feature"), np.array(train_feature))
    np.save(os.path.join(features_folder, "train_label"), np.array(train_labels))

    print("\ntrain feature cache finished!")

    # ======================
    # eval feature
    # ======================
    eval_feature = []
    eval_labels = []

    for i, data in enumerate(eval_dataset.create_dict_iterator()):
        image = data["image"]
        label = data["label"]

        feature = model.predict(Tensor(image))

        eval_feature.append(feature.asnumpy())
        eval_labels.append(label.asnumpy())

        percent = round(i / eval_size * 100, 2)
        print(
            f'eval feature [{i}/{eval_size}] {percent}%',
            end='\r'
        )

    np.save(os.path.join(features_folder, "eval_feature"), np.array(eval_feature))
    np.save(os.path.join(features_folder, "eval_label"), np.array(eval_labels))

    print("\neval feature cache finished!")

    return (
        np.array(train_feature),
        np.array(train_labels),
        np.array(eval_feature),
        np.array(eval_labels)
    ), train_size