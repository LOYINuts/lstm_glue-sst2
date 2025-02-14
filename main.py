# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import Datasetutils
import mymodel
import train

# 在命令行的参数设置
parser = argparse.ArgumentParser(
    description="PyTorch RNN/LSTM/GRU/Transformer Language Model"
)
parser.add_argument(
    "--data", type=str, default="./data/glue-sst2", help="location of the data corpus"
)
parser.add_argument("--emsize", type=int, default=400, help="size of word embeddings")
parser.add_argument(
    "--nhid", type=int, default=200, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,  # 你可能需要调整它
    help="initial learning rate",
)
parser.add_argument("--clip", type=float, default=0.15, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=10, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
parser.add_argument("--bptt", type=int, default=35, help="sequence length")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.05,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument(
    "--tied", action="store_true", help="tie the word embedding and softmax weights"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=50, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
parser.add_argument(
    "--nhead",
    type=int,
    default=2,
    help="the number of heads in the encoder/decoder of the transformer model",
)
parser.add_argument(
    "--dry-run", action="store_true", help="verify the code and the model"
)
args = parser.parse_args()

# 设置相同的种子
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda."
        )

device = torch.device("cuda" if args.cuda else "cpu")

# 数据集设置
train_dataset = Datasetutils.Mydataset(args.data, args.bptt, "train")
test_dataset = Datasetutils.Mydataset(args.data, args.bptt, "test")
valid_dataset = Datasetutils.Mydataset(args.data, args.bptt, "valid")
# dataloader设置
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
# 词表大小
ntokens = len(Datasetutils.ALL_DICT)
# 网络模型构建
net = mymodel.MyNet(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, nlayers=args.nlayers)
# 转到显卡训练
net = net.to(device)
# 损失函数和优化函数
lossF = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)

train.TrainNet(
    epochs=args.epochs,
    batch_size=args.batch_size,
    net=net,
    trainDataLoader=train_dataloader,
    validDataLoader=valid_dataloader,
    device=device,
    lossF=lossF,
    optimizer=optimizer,
)