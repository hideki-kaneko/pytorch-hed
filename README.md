# Pytorch-HED
Program author: Hideki Kaneko

Original paper: [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375)


## 1. preprocess
Generate file lists before starting training process.

```
python make_train_data_list.py \
    --rootdir ./path/to/dir \
    --dst ./train.lst \
```

## 2. train
Run command below to start training. Please set correct paths for your environment.

```
python train.py \
    --expname test1\
    --train_list_path ./train.lst \
    --train_dir ./path/to/dir \
    --test_list_path ./test.lst \
    --test_dir ./path/to/dir \
    --lr 0.01\
    --momentum 0.9\
    --batch_size 10\
    --n_epochs 10000\
    --model_path hed.model\
    --resume
```
For every epoch, this script update and save parameters as "hed.model".

For every 10 epochs, parameters will be saved under ./checkpoint .

## 3. predict
Generate segmention images with trained model.

```
python infer.py \
    --model hed.model \
    --list test_pair.lst \
    --dir ./path/to/dir \
    --dst pred 
```

## 4. evaluate
Evaluate the performance of generated images.
ODS(optimal dataset scale) and AP(average precision) will be calculated.

```
python ecaluate.py \
    --preddir pred \
    --truedir ./path/to/dir \
    --dst results
```