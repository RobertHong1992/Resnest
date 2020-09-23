import argparse
import os
import time
import math
import numpy as np
from utils.parser_config import parse_train_cfg
from utils.torch_utils import init_seeds, select_device
from utils.datasets import data_loader
from PIL import Image
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as trns
import torchvision.models as models

from warmup_scheduler import GradualWarmupScheduler

from cutmix.utils import CutMixCrossEntropyLoss

## for test
from torchvision import datasets
from tensorboardX import SummaryWriter

from DALI.base import DALIDataloader
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from DALI.imagenet import HybridTrainPipe, HybridValPipe

from core.resnest import resnest50, resnest101
from core.laplotter import LossAccPlotter

from utils.evaluate import accuracy, AverageMeter



# SHIP_IMAGES_NUM_TRAIN = 5231
# SHIP_IMAGES_NUM_VAL = 1316
SHIP_IMAGES_NUM_TRAIN = 5557
SHIP_IMAGES_NUM_VAL = 875

used_multi_gpu = True
mixed_precision = True

def parse():
    parser = argparse.ArgumentParser(description='Multi GPU PyTorch ImageNet Training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                       metavar='LR',
                       help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--loss-scale', type=str, default=None)
    args = parser.parse_args()
    return args


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def accuracy_multi_gpu(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.t()

    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target)


    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt




def train(opts_dict, args):
    train_epoch = int(opts_dict['epoch'])
    batch_size = int(opts_dict['batch_size'])
    pretrain_weight = opts_dict['pretrained']
    num_workers = int(opts_dict['workers'])
    data_dir = opts_dict['data_dir']
    crop_size=int(opts_dict['crop_size'])
    learn_rate = float(opts_dict['lr'])
    # if use_cuda:
    init_seeds()
    train_dir = opts_dict['train_dir']
    val_dir = opts_dict['val_dir']
    class_name = opts_dict['class_name']
    class_name = class_name.split(",")
    freeze_layer = opts_dict["freeze_layer"]
    checkpoint_dir = opts_dict["checkpoint"]
    resume = opts_dict["resume"]
    try:
        os.stat(checkpoint_dir)
    except:
        os.mkdir(checkpoint_dir)
    embedding_log=5

###---------------------------------------------------------------###

    # 1.load data
    ## reference https://github.com/tanglang96/DataLoaders_DALI
    pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_workers, device_id=args.local_rank,
                                data_dir=data_dir+'/train', crop=crop_size, shard_id=args.local_rank, num_shards=args.world_size)
    pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_workers, device_id=args.local_rank,
                                data_dir=data_dir + '/val', crop=crop_size, size= crop_size, shard_id=args.local_rank, num_shards=args.world_size)

    pip_train.build()
    pip_val.build()
    # train_loader = DALIDataloader(pipeline=pip_train, size=SHIP_IMAGES_NUM_TRAIN, batch_size=batch_size,
    #                               onehot_label=True)
    # val_loader = DALIDataloader(pipeline=pip_val, size=SHIP_IMAGES_NUM_VAL, batch_size=batch_size,
    #                               onehot_label=True)
    train_loader = DALIClassificationIterator(pip_train, reader_name="Reader", fill_last_batch=True)
    val_loader = DALIClassificationIterator(pip_val, reader_name="Reader", fill_last_batch=False)

    # print("[DALI] train dataloader length: %d" % len(train_loader))## len(train_loader)*batch_size = total_image //8
    # print("[DALI] val dataloader length: %d" % len(val_loader))  ## len(train_loader)*batch_size = total_image //8
    # print('[DALI] start iterate train dataloader')
    # time_start = time.time()
    # for i, data in enumerate(train_loader):  # Using it just like PyTorch dataloader
    #     images = data[0].cuda(non_blocking=True)
    #     labels = data[1].cuda(non_blocking=True)
    # time_end = time.time()
    # train_time = time_end - time_start
    # print('[DALI] iteration time: %fs [train]' % (train_time))


###--------------------Pytorch dataloader test------------------###
    # transform_train = trns.Compose([
    #     trns.RandomResizedCrop(crop_size, scale=(0.08, 1.25)),
    #     trns.RandomHorizontalFlip(),
    #     trns.ToTensor(),
    #     trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # train_dst = datasets.ImageFolder(data_dir + '/train', transform_train)
    # train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                                            num_workers=num_workers)
    # print("[PyTorch] train dataloader length: %d" % len(train_loader))
    # print('[PyTorch] start iterate train dataloader')
    # time_start = time.time()
    # for i, data in enumerate(train_loader):
    #     images = data[0].cuda(non_blocking=True)
    #     labels = data[1].cuda(non_blocking=True)
    # time_end = time.time()
    # train_time = time_end - time_start
    # print('[PyTorch] iteration time: %fs [train]' % (train_time)

###---------------------------------------------------------------###

    # 2.load model

    ###vgg16
    model = models.vgg16(pretrained=False).cuda()
    model.load_state_dict(torch.load("./pretrained_weight/vgg16.pth"))

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Linear(256, 64), nn.ReLU(inplace=True),
                     nn.Linear(64, len(class_name))])  # Add our layer with 4 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    # model = resnest50(num_classes=1000)
    # model = model.cuda()
    #
    # if pretrain_weight:  ##if have pretrain model
    #     model.load_state_dict(torch.load(pretrain_weight))
    #
    #     num_features = model.fc.in_features
    #     features = list(model.fc.children())[:-1]  # Remove last layer
    #     features.extend([nn.Linear(num_features, 512), nn.ReLU(inplace=True),  nn.Linear(512,128), nn.ReLU(inplace=True), nn.Linear(128, len(class_name))])  # Add our layer with 4 outputs
    #     model.fc = nn.Sequential(*features)  # Replace the model classifier
    #
    # if resume:
    #     num_features = model.fc.in_features
    #     features = list(model.fc.children())[:-1]  # Remove last layer
    #     features.extend([nn.Linear(num_features, 512), nn.ReLU(inplace=True), nn.Linear(512,128), nn.ReLU(inplace=True), nn.Linear(128, len(class_name))])  # Add our layer with 4 outputs
    #     model.fc = nn.Sequential(*features)  # Replace the model classifier
    #     model.load_state_dict(torch.load(resume))


    if freeze_layer:  ##Freeze training for  layers
        ct = 0
        for name, param in model.named_parameters():
            ct += 1
            if ct < int(freeze_layer):
                param.requires_grad_(False)
            print(ct, name, param.requires_grad)

    if used_multi_gpu:
        model = parallel.convert_syncbn_model(model)
    model = model.cuda()
    # model.to(device)
    # summary(model, (3, 224, 224))## if you need summary

###---------------------------------------------------------------###
    # 3.set optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

    args.lr = args.lr * float(int(opts_dict['batch_size']) * args.world_size) / 256. ## if batch size = 256
    optimizer = optim.SGD(model.parameters(), lr=args.lr , momentum=0.9)
    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
    optimizer.zero_grad()
    optimizer.step()


    print("initial_learning_rate:", optimizer.defaults['lr'])
    ## Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=args.loss_scale)

    if used_multi_gpu:
        model = DDP(model, delay_allreduce=True)
        # model = DDP(model)

    criterion = CutMixCrossEntropyLoss(True) ##cutmixed
    #
    # # 4. training
    writer = SummaryWriter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    e = 0


    for epoch in range(1,train_epoch+1):
        epoch_loss = 0.0
        model.train()
        scheduler_warmup.step(epoch)  ## pytorch-gradual-warmup-lr https://github.com/ildoonet/pytorch-gradual-warmup-lr
        # print(epoch, optimizer.param_groups[0]['lr'])


        for batch_idx,data in enumerate(train_loader):
            # n_iter = (epoch*len(train_loader))+batch_idx

            ### if want to save dataloader image
            # data_2 = data[0][1].cpu().numpy()
            # data_2 = data_2.transpose(1,2,0)
            # data_2 -= data_2.min()
            # data_2 /= data_2.max()
            # data_2 *= 255

            #
            # cv.imwrite("./data_aug_sample/{}.jpeg".format(batch_idx), data_2)

            # images = data[0].cuda(non_blocking=True)
            # labels = data[1].cuda(non_blocking=True)
            images = data[0]["data"]
            labels = data[0]["label"].squeeze().cuda().long()


            output = model(images)
            # loss = F.cross_entropy(output, labels)
            loss = criterion(output, labels)  ## cut mixed

            optimizer.zero_grad()

            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)



            # if bath_idx % embedding_log == 0:
            #     out = torch.cat((output.data, torch.ones(len(output), 1, device=device)), 1)
            #     writer.add_embedding(out, metadata=labels.data, label_img=images.data, global_step=n_iter)
        if epoch % 10 == 0: ## every 10 epoch caculate once accuracy
            model.eval()
            for iii, val_data in enumerate(val_loader):
                # val_images = val_data[0].cuda(async=True)
                # val_labels = val_data[1].cuda(async=True)
                val_images = data[0]["data"]
                val_labels = data[0]["label"].squeeze().cuda().long()
                with torch.no_grad():
                    val_output = model(val_images)
                acc1, acc2 = accuracy_multi_gpu(val_output.data, val_labels, topk=(1, 3))

                if used_multi_gpu:
                    acc1 = reduce_tensor(acc1)
                    acc2 = reduce_tensor(acc2)

                top1.update(to_python_float(acc1), val_images.size(0))
                top3.update(to_python_float(acc2), val_images.size(0))

                    # out = torch.cat((val_output.data, torch.ones(len(val_output), 1, device=device)), 1)
                    # writer.add_embedding(out, metadata=val_labels.data, label_img=val_images.data, global_step=(epoch*len(val_loader))+iii)

            if args.local_rank == 0:
                print('Top1 Acc: %.3f | Top3 Acc: %.3f ' % (top1.avg, top3.avg))
            e = int(epoch)
            torch.save(model.state_dict(), "{}/{}.pt".format(checkpoint_dir, e))
            val_loader.reset()

        torch.save(model.state_dict(), "{}/last.pt".format(checkpoint_dir))
        # if epoch % 25 == 0:
        #     scheduler.step()

        print("epoch:{}, loss:{:.6f}".format(epoch, epoch_loss))

        ## visualization
        writer.add_scalar('./tensorboard/acc', top1.avg, epoch)
        writer.add_scalar('./tensorboard/total_loss', epoch_loss, epoch)
        writer.add_scalar('./tensorboard/lr',  optimizer.param_groups[0]['lr'], epoch)
        train_loader.reset()
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'INFO'
    args = parse()

    opts_dict = parse_train_cfg("train_cfg.txt")

    if not os.path.isdir(opts_dict["checkpoint"]):
        os.mkdir(opts_dict["checkpoint"])




    ##
    if used_multi_gpu:
        try:
            global DDP, amp, optimizers, parallel
            from apex.parallel import DistributedDataParallel as DDP
            from apex import amp, optimizers, parallel
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    else:
        try:  # Mixed precision training https://github.com/NVIDIA/apex
            from apex import amp
        except:
            print('you should use Apex for faster training')
            mixed_precision = False

    args.gpu = 0
    args.world_size = 1

    if used_multi_gpu:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    total_batch_size = args.world_size * int(opts_dict['batch_size'])
    print("Total batch size memory:{}".format(total_batch_size))
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."





    # if used_multi_gpu:
    #     device = select_device(device=args.local_rank, apex=mixed_precision,
    #                            batch_size=int(opts_dict['batch_size']))
    # else:
    #     device = select_device(device=opts_dict['gpu_id'], apex=mixed_precision, batch_size=int(opts_dict['batch_size']))
    #




    train(opts_dict, args)



