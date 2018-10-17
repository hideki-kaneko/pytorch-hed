import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class HEDDataset(Dataset):
    '''
        Args:
        csv_path (string): Specify the location of csv file including pairs of relative paths.
        root_dir (string): Specify the root directory used in csv_path.
        enableBatch(bool, optional): Set True if you want to enable batch training.
            This flag reshapes all images to 400x400px. default: False
        enableInferMode(bool, optional): Set True to return (image, filename, originalsize) 
        inverse(bool, optional): Swap 0 and 1 of input images.
    '''
    def __init__(self, csv_path, root_dir, transform=None, enableBatch=False, enableInferMode=False, inverse=False):
        self.file_list = pd.read_csv(csv_path, delimiter=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.enableBatch = enableBatch
        self.enableInferMode = enableInferMode
        self.inverse = inverse

        # pytorch default preprocessing for VGG
        self.rgb_means = [0.485, 0.456, 0.406]
        self.rgb_stds = [0.229, 0.224, 0.225]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        x_img_name = os.path.join(self.root_dir, self.file_list.iloc[idx,0])
        x_img = Image.open(x_img_name)
        size = x_img.size
        if self.enableBatch:
            x_img = x_img.resize((400, 400))
        x_img = np.asarray(x_img, dtype=np.float32)
        x_img = x_img.transpose((2,0,1))
        x_img /= 255.0
        x_img[0,:,:] = (x_img[0,:,:] - self.rgb_means[0]) / self.rgb_stds[0]
        x_img[1,:,:] = (x_img[0,:,:] - self.rgb_means[1]) / self.rgb_stds[1]
        x_img[2,:,:] = (x_img[0,:,:] - self.rgb_means[2]) / self.rgb_stds[2]

        if self.enableInferMode:
            return (x_img, self.file_list.iloc[idx,0], size)
        else:
            y_img_name = os.path.join(self.root_dir, self.file_list.iloc[idx,1])
            y_img = Image.open(y_img_name).convert("L")
            if self.enableBatch:
                y_img = y_img.resize((400, 400))
            y_img = np.asarray(y_img)
            y_img = (y_img > 0).astype(np.float32)
            if self.inverse:
                y_img = 1-y_img
            y_img = y_img[np.newaxis, :, :]
            
            return (x_img, y_img)

