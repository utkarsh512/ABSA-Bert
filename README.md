# Aspect-based Sentiment Analysis using BERT
The README of original repo is available [here](https://github.com/HSLCY/ABSA-BERT-pair/blob/master/README.md). In this project, our approach is to perform aspect-based sentiment analysis via constructing auxilary sentences using BERT.
## How to use this repo
Four different models are used for this task. They are __QA_M__, __QA_B__, __NLI_M__ and __NLI_B__. Their description can be found in the paper. Training and testing using this repo can be done by following these steps:
### Step 1: Preparing environment for BERT fine-tuning
Run this command to install necessary file required for training the models (for the _first_ time only)
```
$ bash make_env.sh
```
### Step 2: Shuffling Dataset and preparing train, dev and test sets
Dataset is shuffled and partitioned into train, dev and test sets. For partitioning, arguments are passed denoting the train and dev ratio. It is guarantted that sentence pairs generated from a particular sentence from the original dataset (which are provided as `JSON` files) occurs in only one of the partition.
```
$ python partition.py --train 0.7 --dev 0.1
```
### Step 3: Training the model
If you want to train __QA_M__ model on the dataset with maximum allowed sequence length of 100 characters and batch size of 32, this can be done as
```
$ bash train.sh QA_M 100 32
```
### Step 4: Evaluating performance of the model
Suppose, we have trained __QA_M__ model. To evaluate its performance, we need the path to directory where results of last epoch was saved while training. Generally, this directory is available at `result/sentihood/{model}/test_ep_{T}.txt` where `T` is the number of epoch used while training and `model` is the name of the model.
```
$ python evaluation.py --task_name sentihood_QA_M --pred_dir result/sentihood/QA_M/test_ep_6.txt
```
