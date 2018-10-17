import os
import csv
import argparse

'''
    A script for create file list which is required in training process.

'''

def make_train_data_list(root_dir, dst_path, suffix, delimiter=" "):
    with open(dst_path, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for cur_dir, dirs, files in os.walk(root_dir):
            print(cur_dir)
            for file in files:
                if file.endswith(suffix):
                    x_name = file[:len(suffix)] + suffix[suffix.rfind("."):]
                    x_path = os.path.join(cur_dir, x_name)
                    x_path_short = x_path[len(root_dir):]
                    if x_path_short[0] == "/":
                         x_path_short = x_path_short[1:]
                    y_path = os.path.join(cur_dir, file)
                    y_path_short = y_path[len(root_dir):]
                    if y_path_short[0] == "/":
                         y_path_short = y_path_short[1:]
                    if os.path.exists(x_path):
                        paths = [x_path_short, y_path_short]
                        writer.writerow(paths) 
                        
def main(args):
    make_train_data_list(args.rootdir, args.dst, args.suffix, args.delimiter)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", required=True, help="specify the root directory")
    parser.add_argument("--dst", required=True, help="specify destination file path")
    parser.add_argument("--delimiter", help="delimiter in the output csv", default = " ")
    parser.add_argument("--suffix", help="suffix of file names for target", default=".png")
                       
    args = parser.parse_args()

    main(args)

