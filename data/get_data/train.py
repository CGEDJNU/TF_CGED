import os
import time
import pickle
#from tqdm import tqdm
import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from utils.visualize import *

torch.manual_seed(1)

CUDA_VALID = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_VALID = 0


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def seq_to_ix(seq, ix_dict):
	"""
		word_to_ix
		tag_to_ix
	"""
	return [ix_dict[char] for char in seq]

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def load_word_tag_ix(word_to_ix_path, tag_to_ix_path):
    with open('../'+word_to_ix_path, 'rb') as wordf:
        word_to_ix = pickle.load(wordf)
    with open('../'+tag_to_ix_path, 'rb') as tagf:
        tag_to_ix = pickle.load(tagf)
    return word_to_ix, tag_to_ix

def get_training_data(train_data_path, seq_len = 80):
    """
        input: 
            ---------------
            抱 B-v O
            怨 B-v O    -> sentence 1
            的 b-u O

            科 B-n O
            技 B-n O    -> sentence 2
            ---------------
    """
    data = open(train_data_path).readlines()
    training_data = []

    list1 = []
    list2 = []
    for i in range(len(data)):
        if data[i] != '\n':
            char, pos, err = data[i].split()
            if len(list1) < seq_len:
            	list1.append(char)
            	list2.append(err)
        else:
            if len(list1) < seq_len:
                for i in range(seq_len - len(list1)):
                       list1.append(' ')
                       list2.append('O')
            training_data.append((list1, list2))
            list1 = []
            list2 = []
    return training_data
def get_training_data__(train_data_path):
    """
        input: 
            ---------------
            抱 B-v O
            怨 B-v O    -> sentence 1
            的 b-u O

            科 B-n O
            技 B-n O    -> sentence 2
            ---------------
    """
    data = open(train_data_path).readlines()
    training_data = []

    list1 = []
    list2 = []
    for i in range(len(data)):
        if data[i] != '\n':
            char, pos, err = data[i].split()
            list1.append(char)
            list2.append(err)
        else:
            training_data.append((list1, list2))
            list1 = []
            list2 = []
    return training_data

def get_training_data_(train_data_path):
    """
        input: 
            ---------------
            我 们 在 回 去 加 拿 大 之 前
            O  O  O  O  O  B-R O O  O  O    -> sentence 1
            我 认 为 所 有 的 人 类
            O  O  O  O  O  O  O  O          -> sentence 2
            ---------------
    """
    training_data=[]
    data= open(train_data_path).readlines()
    for i in range(len(data)):
        data[i] = data[i].replace('\n', ' ');
        if i%2 ==0:
            tup1=(data[i].split(),)
        else :
            tup2=(data[i].split(),)
            tup3 = tup1 + tup2
            training_data.append(tup3)
    return training_data

def get_dict_word_and_tag(train_data_path, test_data_path, word_to_ix_path, tag_to_ix_path):
    training_data = get_training_data(train_data_path)
    test_data = get_training_data(test_data_path) 
    all_data = training_data + test_data
    word_to_ix = {}
    for sentence, tags in all_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)+1
    word_to_ix[' '] = 0
    tag_to_ix = {"O": 0, "B-R": 1, "I-R": 2,"B-M": 3, "I-M": 4, "B-S": 5, "I-S": 6,"B-W": 7, "I-W": 8}

    with open('data/'+word_to_ix_path, 'wb') as wordf:
        pickle.dump(word_to_ix, wordf)
    with open('data/'+tag_to_ix_path, 'wb') as tagf:
        pickle.dump(tag_to_ix, tagf)

#####################################################################

if __name__ == '__main__':
        experiment_num  =   1
        # Run training
        EMBEDDING_DIM   =   10
        HIDDEN_DIM      =   20
        RESUME          =   0
        CHECKPOINT      =   80
        
        # Make up some training data
        epoch_num = 300
        model_save_interval = 5
        # Ratio of data to use
        train_ratio = 0.8
        val_ratio = 0.3
        train_data_path = 'train_CGED2016_2018.txt'
        test_data_path = 'test_CGED2016.txt'
        
        word_to_ix_path = 'word_to_ix.pkl'
        tag_to_ix_path = 'tag_to_ix.pkl'
        
        get_dict_word_and_tag(train_data_path, test_data_path, word_to_ix_path, tag_to_ix_path)
        word_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, tag_to_ix_path)
        def get_train_test(data, word_to_ix, tag_to_ix):
                char_seqs = [data[i][0] for i in range(len(data))]
                char_seqs_ix = [seq_to_ix(seq, word_to_ix) for seq in char_seqs]
                tag_seqs = [data[i][1] for i in range(len(data))]
                tag_seqs_ix = [seq_to_ix(seq, tag_to_ix) for seq in tag_seqs]
                X = np.array(char_seqs_ix)
                Y  = np.array(tag_seqs_ix)
                return X, Y
        train_data = get_training_data(train_data_path)
        test_data = get_training_data(test_data_path)
        X_train, Y_train = get_train_test(train_data, word_to_ix, tag_to_ix)	
        X_test, Y_test = get_train_test(test_data, word_to_ix, tag_to_ix)
        data = [X_train, Y_train, X_test, Y_test]
        name = ['X_train.npy', 'Y_train.npy', 'X_test.npy', 'Y_test.npy']
        for i in range(len(data)):
            np.save('../'+name[i], data[i])
