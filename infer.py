from utils import Predictor
from model import HED
from dataset import HEDDataset
from torch.utils.data import DataLoader
import argparse
import torch

'''
    Script for prediction.
    Call this script after training.
'''

def main(args):
    dataset_test = HEDDataset(csv_path=args.list, root_dir=args.dir, enableBatch=True, enableInferMode=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

    device = torch.device("cpu" if args.no_cuda else "cuda:0")
    p = Predictor(HED, args.model, device, args.dst, test_loader)
    p.infer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="location of trained parameter")
    parser.add_argument("--list", required=True, help="location of the file list")
    parser.add_argument("--dir", required=True, help="location of the directory of images")
    parser.add_argument("--dst", required=True, help="output directory")
    parser.add_argument("--no-cuda", action="store_true", help="disable GPU")
    args = parser.parse_args()

    main(args)
