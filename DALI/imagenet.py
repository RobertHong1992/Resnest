import os
import sys
import time
import torch
import pickle
import numpy as np
import nvidia.dali.ops as ops
from .base import DALIDataloader
from torchvision import datasets
from sklearn.utils import shuffle
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
import cupy

IMAGENET_MEAN = [0.49139968, 0.48215827, 0.44653124]
IMAGENET_STD = [0.24703233, 0.24348505, 0.26158768]
IMAGENET_IMAGES_NUM_TRAIN = 1281167
IMAGENET_IMAGES_NUM_TEST = 50000
IMG_DIR = '/gdata/ImageNet2012'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
VAL_SIZE = 256
CROP_SIZE = 224






def cut_mixe_image(image1, image2, lb1, lb2):
    assert image1.shape == image2.shape
    h, w, c = image2.shape
    lam = np.random.beta(1, 1)
    # bbx1, bby1, bbx2, bby2 = rand_bbox(image2.size(), lam)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    y, x = cupy.ogrid[bby1:bby2, bbx1:bbx2]
    mask = x*w+y
    result1 = cupy.copy(image1)
    result1[mask] = image2[mask]
    lb1_onehot = lb1 * lam + lb2 * (1. - lam)
    return result1, lb1_onehot


class LoadImagePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, shard_id, num_shards):
        super(LoadImagePipeline, self).__init__(batch_size, num_threads, device_id, crop,
                                                 exec_async=False, exec_pipelined=False)

        self.input1 = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, pad_last_batch=True)
        self.input2 = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, pad_last_batch=True)

        # device_memory_padding = 211025920
        # host_memory_padding = 140544512
        device_memory_padding = 320000
        host_memory_padding = 160000
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB, device_memory_padding=device_memory_padding,
                                       host_memory_padding=host_memory_padding)


    def load(self):
        jpg1, label1 = self.input1(name="Reader")
        jpg2, label2 = self.input2()
        image1 = self.decode(jpg1)
        image2 = self.decode(jpg2)


        return [image1, image2, label1, label2]


class HybridTrainPipe(LoadImagePipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, shard_id, num_shards):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, data_dir, crop, shard_id, num_shards)
        self.pad = ops.Paste(device="gpu", fill_value=0, ratio=1.1, min_canvas_size=crop)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.1, 1.0], random_aspect_ratio=[0.8, 1.25], num_attempts=100)
        self.cutmix = ops.PythonFunction(function=cut_mixe_image, num_outputs=2, device='gpu')
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                    output_dtype=types.FLOAT,
                                                    output_layout=types.NCHW,
                                                    image_type=types.RGB,
                                                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        self.rotated = ops.Rotate(device="gpu", keep_size=True)
        self.rotated_rng = ops.Uniform(range=(-5.0, 5.0))
        self.brightness = ops.Brightness(device="gpu")
        self.brightness_rng = ops.Uniform(range=(0.8, 1.2))
        self.reshape = ops.Reshape(device="gpu", layout="HWC")
        self.one_hot = ops.OneHot(num_classes=3, dtype=types.INT32, device="cpu")
        self.jitter_rng = ops.CoinFlip(probability=0.3)
        self.jittered = ops.Jitter(device="gpu")
    def define_graph(self):
        rng = self.coin()
        images1, images2, label1, label2 = self.load()
        label1 = self.one_hot(label1)
        label2 = self.one_hot(label2)

        images1 = self.pad(images1)
        images2 = self.pad(images2)
        images1 = self.res(images1)
        images2 = self.res(images2)

        images1, label1 = self.cutmix(images1, images2, label1.gpu(), label2.gpu())

        bright = self.brightness_rng()
        images1 = self.brightness(images1, brightness=bright)
        angle = self.rotated_rng()
        images1 = self.rotated(images1, angle=angle)

        jitter = self.jitter_rng()
        images1 = self.jittered(images1, mask = jitter)
        images1 = self.reshape(images1)
        images1 = self.cmnp(images1, mirror=rng)

        return [images1, label1]





##-------------------------------------##
#
# class HybridTrainPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1, cut_mixed=True):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id, exec_async=False,  exec_pipelined=False)
#         dali_device = "gpu"
#         self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
#         # self.cutmix_bool = cut_mixed
#         # if self.cutmix_bool:
#         self.input2 = ops.FileReader(file_root=data_dir, random_shuffle=True)
#         self.cutmix = ops.PythonFunction(function=edit_images, num_outputs=2, device="gpu")
#         self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
#         self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.9, 1.1], random_aspect_ratio=1.33333)
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#         self.coin = ops.CoinFlip(probability=0.5)
#         self.rotated = ops.Rotate(device="gpu", keep_size=True)
#         self.rotated_rng = ops.Uniform(range=(-5.0, 5.0))
#         self.brightness = ops.Brightness(device="gpu")
#         self.brightness_rng = ops.Uniform(range=(0.8, 1.2))
#         print('DALI "{0}" variant'.format(dali_device))
#
#     ## add cutmix, if don't use cutmix need to mark out
#
#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#
#         jpeg2, label2 = self.input2(name="Reader")
#         image2 = self.decode(jpeg2)
#         image2 = self.res(image2)
#
#         # if self.cutmix_bool:
#         # jpg2, label2 = self.input2()
#         # images2 = self.decode(jpg2)
#         # images2 = self.res(images2)
#         # images, self.labels = self.cutmix(images, self.labels, images2, label2)
#         images, self.labels = self.cutmix(images, image2, self.labels, label2)
#         bright = self.brightness_rng()
#         images = self.brightness(images, brightness=bright)
#         angle = self.rotated_rng()
#         images = self.rotated(images, angle=angle)
#         output = self.cmnp(images, mirror=rng)
#
#         return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, shard_id, num_shards):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, shard_id, num_shards)
        self.input = ops.FileReader(file_root=data_dir, shard_id=shard_id, num_shards=num_shards,
                                    random_shuffle=False, pad_last_batch=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


if __name__ == '__main__':
    # iteration of DALI dataloader
    pip_train = HybridTrainPipe(batch_size=TRAIN_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/train', crop=CROP_SIZE, world_size=1, local_rank=0)
    pip_test = HybridValPipe(batch_size=TEST_BS, num_threads=NUM_WORKERS, device_id=0, data_dir=IMG_DIR+'/val', crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0)
    train_loader = DALIDataloader(pipeline=pip_train, size=IMAGENET_IMAGES_NUM_TRAIN, batch_size=TRAIN_BS, onehot_label=True)
    test_loader = DALIDataloader(pipeline=pip_test, size=IMAGENET_IMAGES_NUM_TEST, batch_size=TEST_BS, onehot_label=True)
    # print("[DALI] train dataloader length: %d"%len(train_loader))
    # print('[DALI] start iterate train dataloader')
    # start = time.time()
    # for i, data in enumerate(train_loader):
    #     images = data[0].cuda(non_blocking=True)
    #     labels = data[1].cuda(non_blocking=True)
    # end = time.time()
    # train_time = end-start
    # print('[DALI] end train dataloader iteration')

    print("[DALI] test dataloader length: %d"%len(test_loader))
    print('[DALI] start iterate test dataloader')
    start = time.time()
    for i, data in enumerate(test_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[DALI] end test dataloader iteration')
    # print('[DALI] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))
    print('[DALI] iteration time: %fs [test]' % (test_time))


    # iteration of PyTorch dataloader
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.08, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dst = datasets.ImageFolder(IMG_DIR+'/train', transform_train)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=TRAIN_BS, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
    transform_test = transforms.Compose([
        transforms.Resize(VAL_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dst = datasets.ImageFolder(IMG_DIR+'/val', transform_test)
    test_iter = torch.utils.data.DataLoader(test_dst, batch_size=TEST_BS, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    # print("[PyTorch] train dataloader length: %d"%len(train_loader))
    # print('[PyTorch] start iterate train dataloader')
    # start = time.time()
    # for i, data in enumerate(train_loader):
    #     images = data[0].cuda(non_blocking=True)
    #     labels = data[1].cuda(non_blocking=True)
    # end = time.time()
    # train_time = end-start
    # print('[PyTorch] end train dataloader iteration')

    print("[PyTorch] test dataloader length: %d"%len(test_loader))
    print('[PyTorch] start iterate test dataloader')
    start = time.time()
    for i, data in enumerate(test_loader):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[PyTorch] end test dataloader iteration')
    # print('[PyTorch] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))
    print('[PyTorch] iteration time: %fs [test]' % (test_time))
