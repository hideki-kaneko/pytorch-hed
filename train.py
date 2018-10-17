from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import os
import argparse

from model import HED, get_vgg_weights
from dataset import HEDDataset
from utils import train, test, TrainLogger

'''
    A script for training.

'''


def main(args):
    done_epoch = 0
    if args.resume and os.path.exists(args.model_path):
        print("resume training...")
        model = HED()
        model.load_state_dict(torch.load(args.model_path))
        with open("{0}-history.csv".format(args.expname), 'r') as f:
            for i, l in enumerate(f):
                pass
            done_epoch = i
    else:
        print("initialize training...")
        model = HED()
        model_dict = model.state_dict()
        vgg_weights = get_vgg_weights()
        model_dict.update(vgg_weights)
        model.load_state_dict(model_dict)
        nn.init.constant_(model.fuse.weight_sum.weight, 0.2)
        nn.init.constant_(model.side1.conv.weight, 1.0)
        nn.init.constant_(model.side2.conv.weight, 1.0)
        nn.init.constant_(model.side3.conv.weight, 1.0)
        nn.init.constant_(model.side4.conv.weight, 1.0)
        nn.init.constant_(model.side5.conv.weight, 1.0)
        nn.init.constant_(model.side1.conv.bias, 1.0)
        nn.init.constant_(model.side2.conv.bias, 1.0)
        nn.init.constant_(model.side3.conv.bias, 1.0)
        nn.init.constant_(model.side4.conv.bias, 1.0)
        nn.init.constant_(model.side5.conv.bias, 1.0)
        logger = TrainLogger("{0}-history.csv".format(args.expname), overwrite=True)
        del(logger)
    
    dataset_train = HEDDataset(csv_path=args.train_list_path, root_dir=args.train_dir, enableBatch=True)
    dataset_test = HEDDataset(csv_path=args.test_list_path, root_dir=args.test_dir, enableBatch=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device("cpu" if args.no_cuda else "cuda:0")
    model = model.to(device)
    sgd = opt.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #sc_lambda = lambda epoch: 0.1 if epoch <5000 else 0.01
    #scheduler = LambdaLR(sgd, lr_lambda=sc_lambda)

    train(model=model, device=device, train_loader=train_loader, test_loader=test_loader,
                                  optimizer=sgd, n_epochs=args.n_epochs, prefix=args.expname, done_epoch=done_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", required=True, help="prefix for auto-saving params and loss history")
    parser.add_argument("--train_list_path", required=True, help="location of trainig files list")
    parser.add_argument("--train_dir", required=True, help="location of trainig files directory")
    parser.add_argument("--test_list_path", required=True, help="location of test files list")
    parser.add_argument("--test_dir", required=True, help="location of test files directory")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum factor")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs")

    parser.add_argument("--model_path", default="hed.model", help="parameters updated every epoch")
    parser.add_argument("--resume", action="store_true", help="resume the training")
    parser.add_argument("--no-cuda", action="store_true", help="disable GPU")
    args = parser.parse_args()

    main(args)
