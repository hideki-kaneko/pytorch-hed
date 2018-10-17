import numpy as np
import os
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from PIL import Image


class TrainLogger():
    '''
        Log the trainig history.
        Usage: logger = TrainLogger("path_to_log.csv")
               logger.log(loss_train, loss_test)
        This class automatically append the csv file unless overwrite is set True.
    '''
    def __init__(self, dst_path, overwrite=False):
        self.path = dst_path
        if overwrite:
            with open(self.path, 'w') as f:
                writer = csv.writer(f)
                header = ["loss_train", "loss_test"]
                writer.writerow(header)
    
    def log(self, loss_train, loss_test):
        with open(self.path, 'a') as f:
       	    writer = csv.writer(f)
            row = [loss_train, loss_test]
            writer.writerow(row)

class Predictor():
    '''
        Generate segmentation images.
        Usage: predictor = Predictor(model, weigth_path, device, dst_dir, dataloader)
               predictor.infer()

    '''
    def __init__(self, model, weight_path, device, dst_dir, dataloader):
        self.model = model()
        self.model.load_state_dict(torch.load(weight_path))
        self.model = self.model.to(device)
        self.dst_dir = dst_dir
        self.dataloader = dataloader 
        self.device = device
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    
    def infer(self):
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                x, name, size = data 
                size = (size[0].item(), size[1].item())
                path = os.path.join(self.dst_dir, name[0])
                x = x.to(self.device)
                fuse, s1, s2, s3, s4, s5 = self.model(x)
                maxv = fuse.max().item()
                fuse = fuse * (255/maxv) if maxv!=0 else fuse 
                img = fuse.squeeze().cpu().numpy()
                img = Image.fromarray(img).convert("L")
                img = img.resize(size)
                dirname = os.path.dirname(path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                img.save(path)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def test(model, device, test_loader):
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            fuse, s1, s2, s3, s4, s5 = model(x)
            loss = model.loss(fuse, s1, s2, s3, s4, s5, y)
            total_loss += loss.item()

    loss = total_loss/len(test_loader.dataset)
    return loss

def train(model, device, train_loader, test_loader, optimizer, n_epochs,
                scheduler=None, done_epoch=0, prefix="", path_checkpoint="./checkpoint"):
    '''
        Method for training process.
        Args:
            done_epoch (int): if you resume the training, set the number of epochs you have done before.
            prefix (str): name the prefix for auto-saving files
            path_checkpoint (str): this method automatically saves parameters to 
                                   this location for every 10 epochs.
                                   
        For every epoch, parameters will be saved as "hed.model"
    '''
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    if done_epoch >= n_epochs:
        print("epochs exceeded{0}".format(n_epochs))
        return
    logger = TrainLogger("{0}-history.csv".format(prefix))

    for epoch in range(done_epoch+1, n_epochs):
        train_total_loss = 0
        for data in tqdm(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            fuse, s1, s2, s3, s4, s5 = model(x)
            loss = model.loss(fuse, s1, s2, s3, s4, s5, y)
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss = train_total_loss / len(train_loader.dataset) 
        test_loss = test(model, device, test_loader)
        logger.log(train_loss, test_loss)
        save_model(model, "hed.model")
        if epoch % 10 == 0:
            name = "{0}-ep{1}.model".format(prefix, epoch)
            save_model(model, os.path.join(path_checkpoint, name))
    
    
def plot_loss(train_loss, test_loss):
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss)
    plt.plot(x, test_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"])
