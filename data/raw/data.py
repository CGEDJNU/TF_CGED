import os
import time
import pickle
#from tqdm import tqdm
import argparse
import word2vec

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

def load_word_tag_ix(word_to_ix_path, bigram_to_ix_path, pos_to_ix_path, tag_to_ix_path):
    with open('../input/'+word_to_ix_path, 'rb') as wordf:
        word_to_ix = pickle.load(wordf)
    with open('../input/'+bigram_to_ix_path, 'rb') as bigramf:
        bigram_to_ix = pickle.load(bigramf)
    with open('../input/'+pos_to_ix_path, 'rb') as posf:
        pos_to_ix = pickle.load(posf)
    with open('../input/'+tag_to_ix_path, 'rb') as tagf:
        tag_to_ix = pickle.load(tagf)
    return word_to_ix, bigram_to_ix, pos_to_ix, tag_to_ix

def get_training_data(data_path, seq_len = 80):
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
    data = open(data_path).readlines()
    training_data = []

    list1 = []
    list2 = []
    list3 = []
    
    for i in range(len(data)):
        if data[i] != '\n':
            char, pos, err = data[i].split()
            if len(list1) < seq_len:
                list1.append(char)
                list2.append(pos)
                list3.append(err)
        else:
            if len(list1) < seq_len:
                for i in range(seq_len - len(list1)):
                    list1.append(' ')
                    list2.append('B-wp')
                    list3.append('O')
            training_data.append((list1, list2, list3))
            list1 = []
            list2 = []
            list3 = []
    # Add bigram feature
    full_training_data = []
    for i in range(len(training_data)):
        list1, list3, list4 = training_data[i]
        list2 = []
        for j in range(len(list1)):
                #if j == 0:
                #    list2.append('<SOS>'+list1[j])
                #    list2.append(list1[j]+list1[j+1])
                #elif j == seq_len - 1:
                #    list2.append(list1[j]+'<EOS>')
                #else:
                #    list2.append(list1[j]+list1[j+1])
                
                if j == seq_len - 1:
                    list2.append(list1[j]+'<EOS>')
                else:
                    list2.append(list1[j]+list1[j+1])
        
        full_training_data.append((list1, list2, list3, list4))
    return full_training_data

def get_dict_word_and_tag(train_data_path, test_data_path, word_to_ix_path, bigram_to_ix_path, pos_to_ix_path, tag_to_ix_path):
    training_data = get_training_data(train_data_path)
    test_data = get_training_data(test_data_path) 
    all_data = training_data + test_data
    word_to_ix = {}
    bigram_to_ix = {}
    for sentence, bigram, pos, tags in all_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)+1
        for gram in bigram:
            if gram not in bigram_to_ix:
                bigram_to_ix[gram] = len(bigram_to_ix)+1
    
    bigram_to_ix[' '] = 0
    word_to_ix[' '] = 0
    tag_to_ix = {"O": 0, "B-R": 1, "I-R": 2,"B-M": 3, "I-M": 4, "B-S": 5, "I-S": 6,"B-W": 7, "I-W": 8}
    pos_list = ['B-wp','B-o', 'B-a', 'I-m', 'B-ws', 'I-n', 'B-nl', 'I-nt', 'I-nl', 'I-nh', 'I-a', 'I-c', 'B-h', 'I-ni', 'B-p', 'B-n', 'B-z', 'I-i', 'B-v', 'B-m', 'I-q', 'I-r', 'I-p', 'I-u', 'B-nt', 'I-b', 'B-b', 'I-d', 'B-ns', 'B-ni', 'I-ns', 'B-q', 'B-nd', 'I-v', 'B-d', 'B-i', 'B-c', 'B-nh', 'I-ws', 'I-j', 'B-u', 'I-wp', 'B-j', 'B-r', 'I-z', 'I-nz', 'I-o', 'B-k', 'B-e', 'I-nd', 'B-nz']
    pos_to_ix = {pos_list[i]:i for i in range(len(pos_list))}
    with open('../input/'+word_to_ix_path, 'wb') as wordf:
        pickle.dump(word_to_ix, wordf)
    with open('../input/'+pos_to_ix_path, 'wb') as posf:
        pickle.dump(pos_to_ix, posf)
    with open('../input/'+tag_to_ix_path, 'wb') as tagf:
        pickle.dump(tag_to_ix, tagf)
    with open('../input/'+bigram_to_ix_path, 'wb') as bigramf:
        pickle.dump(bigram_to_ix, bigramf)

def get_train_test(data, word_to_ix, bigram_to_ix, pos_to_ix, tag_to_ix):
        char_seqs = [data[i][0] for i in range(len(data))]
        char_seqs_ix = [seq_to_ix(seq, word_to_ix) for seq in char_seqs]
        
        bigram_seqs = [data[i][1] for i in range(len(data))]
        bigram_seqs_ix = [seq_to_ix(seq, bigram_to_ix) for seq in bigram_seqs]
        
        pos_seqs = [data[i][2] for i in range(len(data))]
        pos_seqs_ix = [seq_to_ix(seq, pos_to_ix) for seq in pos_seqs]
        
        tag_seqs = [data[i][3] for i in range(len(data))]
        tag_seqs_ix = [seq_to_ix(seq, tag_to_ix) for seq in tag_seqs]

        X_char = np.array(char_seqs_ix)
        X_bigram = np.array(bigram_seqs_ix)
        X_pos = np.array(pos_seqs_ix)
        
        Y  = np.array(tag_seqs_ix)
        return X_char, X_bigram, X_pos, Y

def get_word2vec_model(corpus_path, model_path, embedding_dim):
    word2vec.word2vec(corpus_path, model_path, embedding_dim, verbose=True)
    exit()

def get_data_by_bigram(train_data, test_data, corpus_path):
    # Merge data
    data = []
    data.extend(train_data)
    data.extend(test_data)
    
    for i in range(len(data)):
        sample = data[i][0]
        sample_split = []
        for j in range(len(sample)-1):
                sample_split.append(sample[j])
                sample_split.append(sample[j+1])
                sample_split.append(' ')
        sample_str = ''.join(sample_split)
        with open(corpus_path, 'a+') as f:
            f.write(sample_str+'\n')
#####################################################################

if __name__ == '__main__':
        train_data_path = 'train_CGED2016_2018.txt'
        test_data_path = 'test_CGED2016.txt'
        
        word_to_ix_path = '../input/word_to_ix.pkl'
        pos_to_ix_path = '../input/pos_to_ix.pkl'        
        tag_to_ix_path = '../input/tag_to_ix.pkl'
        bigram_to_ix_path = '../input/bigram_to_ix.pkl'
        
        corpus_path = '../input/bigram_corpus_train_test.txt'
        word2vec_model_path = '../input/word2vec_model.bin'
        embedding_dim = 128
        
        #get_word2vec_model(corpus_path, word2vec_model_path, embedding_dim)
        
        #get_dict_word_and_tag(train_data_path, test_data_path, word_to_ix_path, bigram_to_ix_path, pos_to_ix_path, tag_to_ix_path)
        word_to_ix, bigram_to_ix, pos_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, bigram_to_ix_path,pos_to_ix_path, tag_to_ix_path)
        train_data = get_training_data(train_data_path)
        test_data = get_training_data(test_data_path)
        
        #get_data_by_bigram(train_data, test_data, corpus_path, word2vec_model_path, embedding_dim)
        
        X_train_char, X_train_bigram, X_train_pos,Y_train =    get_train_test(train_data, word_to_ix, bigram_to_ix, pos_to_ix, tag_to_ix)	
        X_test_char, X_test_bigram, X_test_pos, Y_test    =    get_train_test(test_data, word_to_ix, bigram_to_ix, pos_to_ix, tag_to_ix)
        data = [X_train_char, X_train_bigram, X_train_pos, Y_train, X_test_char, X_test_bigram, X_test_pos, Y_test]
        name = ['X_train_char.npy', 'X_train_bigram.npy','X_train_pos.npy','Y_train.npy', 'X_test_char.npy', 'X_test_bigram.npy','X_test_pos.npy', 'Y_test.npy']
        for i in range(len(data)):
            np.save('../input/'+name[i], data[i])
