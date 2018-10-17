import numpy as np
import pandas as pd
import os
import csv
import argparse
from tqdm import tqdm
from PIL import Image

'''
    Script for evaluation of output
'''

def evaluate_images(dst_path, img_pred, img_true, step=0.01):
    '''
        experimental purpose (this method is not used) 
    '''

    thres = 0.0
    best_f = 0.0
    best_thres = 0.0
    while thres < 1.0:
        img_thred = (img_pred > thres).astype(np.int32)
        try:
            precision = np.sum(np.logical_and(img_thred, img_true)) / np.sum(img_thred)
            recall = np.sum(np.logical_and(img_thred, img_true)) / np.sum(img_true)
            f = 2*precision*recall / (recall + precision)
            
        except ZeroDivisionError:
            f = 0.0
        if f > best_f:
            best_f = f
            best_thres = thres
        thres += step
    return best_f, best_thres

def write_evaluation(dst_dir, img_pred, img_true, step=0.01):
    '''
        make the temporary file for get_ods_ap().
        This method automatically change the threshold to calculate the ODS and AP.

        args:
            dst_dir (str): location to store temporary files
            img_pred (ndarray): a predicted image 
            img_true (ndarray): a ground truth image 
            step (float): step of threshold
    '''

    pred_csv = open(os.path.join(dst_dir, "num_pred.csv"), 'a')
    true_csv = open(os.path.join(dst_dir, "num_true.csv"), 'a')
    and_csv = open(os.path.join(dst_dir, "num_and.csv"), 'a')
    writer_pred = csv.writer(pred_csv)
    writer_true = csv.writer(true_csv)
    writer_and = csv.writer(and_csv)
    thres = 0.0
    list_pred = []
    list_and = []
    num_true = [np.sum(img_true)]
    while thres < 1.0:
        img_thred = (img_pred > thres).astype(np.int32)
        num_pred = np.sum(img_thred)        
        num_and = np.sum(np.logical_and(img_thred, img_true))
        list_pred.append(num_pred)
        list_and.append(num_and)
        thres += step
    writer_pred.writerow(list_pred)
    writer_and.writerow(list_and)
    writer_true.writerow(num_true)
    pred_csv.close()
    true_csv.close()
    and_csv.close()
    
def get_ods_ap(src_dir):
    '''
        Calculate the ODS(optimal dataset scale) and AP(average precision).
        You must run write_evaluation before this function.

        args:
            src_dir (str): location of temporary files
    '''
    df_pred = pd.read_csv(os.path.join(src_dir, "num_pred.csv"), header=None)
    df_true = pd.read_csv(os.path.join(src_dir, "num_true.csv"), header=None)
    df_and = pd.read_csv(os.path.join(src_dir, "num_and.csv"), header=None)
    if not len(df_pred) == len(df_true) == len(df_and):
        print("Different rows")
        return 
    n_cols = len(df_pred.columns)
    f_best = 0.0
    sum_precision = 0.0
    for i in range(n_cols):
        try:
            precision = np.sum(df_and.iloc[:,i]) / np.sum(df_pred.iloc[:,i])
            recall = np.sum(df_and.iloc[:,i]) / np.sum(df_true.iloc[:])
            f = float(2*precision*recall / (recall + precision))

            sum_precision += precision
        except ZeroDivisionError:
            f = 0.0
        if f > f_best:
            f_best = f
        
    ois = f_best
    ap = sum_precision / n_cols
    return ois, ap

def main(args):
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    with open(os.path.join(dst_dir, "num_pred.csv"), 'w') as f:
        pass
    with open(os.path.join(dst_dir, "num_true.csv"), 'w') as f:
        pass
    with open(os.path.join(dst_dir, "num_and.csv"), 'w') as f:
        pass

    files = os.listdir(args.preddir)
    for f in tqdm(files):
        pred_path = os.path.join(args.preddir, f) 
        true_path = os.path.join(args.truedir, f)
        img_pred = np.asarray(Image.open(pred_path)) / 255.0
        img_true = np.asarray(Image.open(true_path)) / 255.0

        write_evaluation(args.dst, img_pred, img_true)
        
    ods, ap = get_ods_ap(args.dst)
    with open(os.path.join(args.dst, "result.txt"), 'w') as f:
        text = "ods:{0}, ap:{1}".format(ods, ap)
        f.write(text)
        print(text)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--preddir", required=True, help="specify the directory of predicted images")
    parser.add_argument("--truedir", required=True, help="specify the directory of true images") 
    parser.add_argument("--dst", required=True, help="specifiy destination directory path")
    args = parser.parse_args()

    main(args)
