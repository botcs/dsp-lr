#! /usr/bin/python3.5
import torch
import torch.backends.cudnn as cudnn

import datahandler
import models
import transformations
import trainer as T

import sys, os
import argparse

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='PyTorch AF-detector Training')
parser.add_argument('--spectrogram', '-s', type=int, help='Use spectrogram with [NFFT]')
parser.add_argument('--debug', '-d', action='store_true', help='print stats')
parser.add_argument('--arch', '-a', default='vgg16_bn')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--sample_length', type=int, default=3000)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.multiprocessing.set_sharing_strategy('file_system')
#name = splitext(basename(sys.argv[0]))[0]
name = args.arch

train_transformations = [
    transformations.Crop(args.sample_length),
    transformations.RandomMultiplier(-1),
]

test_transformations = [

]


in_channels = 1
if args.spectrogram is not None:
    spectral_transformations = [
        transformations.Spectrogram(args.spectrogram),
        transformations.Logarithm()
    ]
    train_transformations += spectral_transformations
    test_transformations += spectral_transformations
    name += "_freq%d" % args.spectrogram
    in_channels = args.spectrogram // 2 + 1

use_cuda = torch.cuda.is_available()



train_set = datahandler.Dataset('data/', transform=train_transformations)
#train_set.load('data/9600Hz/train.csv')
#train_set.load('data/333Hz/train.csv')
train_set.load('data/train.csv')
test_set = datahandler.Dataset('data/', transform=test_transformations)
#test_set.load('data/9600Hz/test.csv')
#test_set.load('data/333Hz/test.csv')
train_set.load('data/test.csv')

train_producer = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=32, shuffle=True,
        num_workers=32, collate_fn=datahandler.batchify)
test_producer = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=datahandler.batchify)
print("=> Building model %30s"%(args.arch))
sample = next(iter(train_producer))
print(sample['x'].shape)
net = models.__dict__[args.arch](in_channels=in_channels, num_classes=train_set.num_classes)
#
# net = torch.nn.Sequential(
#     torch.nn.AvgPool1d(9600//600),
#     net
# )

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    raise RuntimeWarning('GPU could not be initialized')

class_weight = [1.] * train_set.num_classes
trainer = T.Trainer('ckpt/'+name, class_weight=class_weight, dryrun=args.debug)
if args.debug:
    print(net)
trainer(net, train_producer, test_producer, useAdam=True, epochs=1000)
