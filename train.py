import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.set_random_seed(123)

def pipeline_train(X, y, sess, params, isBigram=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(params['batch_size'])
    #dataset = dataset.shuffle(len(X)).batch(params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    #if not isBigram:
    #    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    #else:
    #    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']+1])
    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    y_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    init_dict = {X_ph: X, y_ph: y}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict

def pipeline_test(X, sess, params, isBigram=False):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    #if not isBigram:
    #    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    #else:
    #    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']+1])
    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    init_dict = {X_ph: X}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict

def load_word_tag_ix(word_to_ix_path, bigram_to_ix_path, pos_to_ix, tag_to_ix_path):
    with open(word_to_ix_path, 'rb') as wordf:
            word_to_ix = pickle.load(wordf)
    
    with open(bigram_to_ix_path, 'rb') as bigramf:
            bigram_to_ix = pickle.load(bigramf)
    
    with open(pos_to_ix_path, 'rb') as posf:
            pos_to_ix = pickle.load(posf)
    
    with open(tag_to_ix_path, 'rb') as tagf:
            tag_to_ix = pickle.load(tagf)
    
    return word_to_ix, bigram_to_ix, pos_to_ix,tag_to_ix

def rnn_cell(params):
    #return tf.nn.rnn_cell.GRUCell(params['hidden_dim'],kernel_initializer=tf.orthogonal_initializer())
    return tf.nn.rnn_cell.BasicLSTMCell(params['hidden_dim'] * 4, forget_bias=1.0)

def clip_grads(loss, params):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, params['clip_norm'])
    return zip(clipped_grads, variables)

def forward(x_char, x_bigram, x_pos, reuse, is_training, params):
    with tf.variable_scope('model', reuse=reuse):
        # Input feature[character, pos, bigram]
        x_char = tf.contrib.layers.embed_sequence(x_char, params['char_vocab_size'], params['hidden_dim'])
        x_bigram = tf.contrib.layers.embed_sequence(x_bigram, params['bigram_vocab_size'], params['hidden_dim'])
        x_pos = tf.contrib.layers.embed_sequence(x_pos, params['pos_vocab_size'], params['hidden_dim'])
        
        # Concat bigram
        start_bigram = tf.zeros([params['batch_size'], 1, params['hidden_dim']])
        x_bigram_except_start = x_bigram[:, 1:, :]
        x_bigram_add = tf.concat([start_bigram, x_bigram_except_start], axis=1)

        # Concat dim according to [None, seq_len, hidden_dim]
        x = tf.concat([x_char, x_bigram, x_bigram_add,x_pos],axis=2)
        x = tf.nn.relu(x)
        bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell(params), rnn_cell(params), x, dtype=tf.float32,time_major=False)
       
        x = tf.concat(bi_outputs, -1)
        x = tf.nn.relu(x)
        logits = tf.layers.dense(x, params['n_class'])
    return logits    

def eval(Y_test, Y_pred):
    # Eval(confusion matrix)
    labels = [i for i in range(params['n_class'])]
    target_names = ['O', 'B-R', 'I-R', 'B-M', 'I-M', 'B-S', 'I-S','B-W','I-W']
    print(classification_report(Y_test.ravel(), Y_pred.ravel(), labels=labels, target_names=target_names)) 

if __name__ == '__main__':
    
    params = {
        'seq_len': 80,
        'batch_size': 128,
        'n_class': 9,
        'hidden_dim': 128,
        'clip_norm': 5.0,
        'lr': {'start': 1e-1, 'end': 0.9e-1},
        'n_epoch': 100,
        'display_step': 10,
    }
    word_to_ix_path = 'data/input/word_to_ix.pkl'
    bigram_to_ix_path = 'data/input/bigram_to_ix.pkl'
    pos_to_ix_path = 'data/input/pos_to_ix.pkl'
    tag_to_ix_path = 'data/input/tag_to_ix.pkl'
        
    X_train_char_path = 'data/input/X_train_char.npy'
    X_train_bigram_path = 'data/input/X_train_bigram.npy'
    X_train_pos_path = 'data/input/X_train_pos.npy'
    Y_train_path = 'data/input/Y_train.npy'

    X_test_char_path = 'data/input/X_test_char.npy'
    X_test_bigram_path = 'data/input/X_test_bigram.npy'
    X_test_pos_path = 'data/input/X_test_pos.npy'
    Y_test_path = 'data/input/Y_test.npy'
    
    word_to_ix, bigram_to_ix, pos_to_ix,tag_to_ix = load_word_tag_ix(word_to_ix_path, bigram_to_ix_path,pos_to_ix_path, tag_to_ix_path)
    params['char_vocab_size'] = len(word_to_ix)
    params['bigram_vocab_size'] = len(bigram_to_ix)
    params['pos_vocab_size'] = len(pos_to_ix)
    ix_to_tag = {v:k for k, v in tag_to_ix.items()}
    
    
    # Load data
    X_train_char, X_train_bigram, X_train_pos = np.load(X_train_char_path), np.load(X_train_bigram_path),np.load(X_train_pos_path)
    X_test_char, X_test_bigram,X_test_pos = np.load(X_test_char_path), np.load(X_test_bigram_path), np.load(X_test_pos_path)
    
    train_end_index = ( X_train_char.shape[0] //  params['batch_size'] ) * params['batch_size'] 
    test_end_index = ( X_test_char.shape[0] // params['batch_size'] ) * params['batch_size'] 
    
    Y_train, Y_test = np.load(Y_train_path), np.load(Y_test_path)
    
    Y_test = Y_test[:test_end_index]
    
    sess = tf.Session(config=config) 
    
    iter_train_char, init_dict_train_char = pipeline_train(X_train_char[:train_end_index], Y_train[:train_end_index], sess, params)
    iter_test_char, init_dict_test_char = pipeline_test(X_test_char[:test_end_index], sess, params)
    
    iter_train_bigram, init_dict_train_bigram = pipeline_train(X_train_bigram[:train_end_index], Y_train[:train_end_index], sess, params)
    iter_test_bigram, init_dict_test_bigram = pipeline_test(X_test_bigram[:test_end_index], sess, params)
    
    iter_train_pos, init_dict_train_pos = pipeline_train(X_train_pos[:train_end_index], Y_train[:train_end_index], sess, params)
    iter_test_pos, init_dict_test_pos = pipeline_test(X_test_pos[:test_end_index], sess, params)

    ops = {}
     
    X_train_char_batch, y_train_batch = iter_train_char.get_next()
    X_test_char_batch = iter_test_char.get_next()
    
    X_train_bigram_batch, y_train_batch = iter_train_char.get_next()
    X_test_bigram_batch = iter_test_bigram.get_next()
    
    X_train_pos_batch, y_train_batch = iter_train_pos.get_next()
    X_test_pos_batch = iter_test_pos.get_next()

    logits_tr = forward(X_train_char_batch, X_train_bigram_batch,X_train_pos_batch,False, True, params)
    logits_te = forward(X_test_char_batch, X_test_bigram_batch,X_test_pos_batch, True, False, params)
    
    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        logits_tr, y_train_batch, tf.count_nonzero(X_train_char_batch, 1))

    ops['global_step'] = tf.Variable(0, trainable=False)
    
    #decay_steps = 100
    decay_steps = X_train_char.shape[0] // params['batch_size']
    
    #decay_rate = 0.96
    decay_rate = params['lr']['end'] / params['lr']['start']
    ops['lr'] = tf.train.exponential_decay(params['lr']['start'], ops['global_step'], decay_steps,decay_rate, staircase=False)
    
    ops['train_loss'] = tf.reduce_mean(-log_likelihood)
    
    ops['train'] = tf.train.AdamOptimizer(ops['lr']).apply_gradients(
        clip_grads(ops['train_loss'], params), global_step=ops['global_step'])

    ops['crf_decode'] = tf.contrib.crf.crf_decode(
        logits_te, trans_params, tf.count_nonzero(X_test_char_batch, 1))[0]
    
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, params['n_epoch']+1):
        while True:
            try:
                _, step, train_loss, lr = sess.run([ops['train'],
                                              ops['global_step'],
                                              ops['train_loss'],
                                              ops['lr']])
            except tf.errors.OutOfRangeError:
                break
            else:
                if step % params['display_step'] == 0 or step == 1:
                    print("Epoch %d | Step %d | Train_Loss %.3f | LR: %.4f" % (epoch, step, train_loss, lr))
        Y_pred = []
        while True:
            try:
                Y_pred.append(sess.run(ops['crf_decode']))
            except tf.errors.OutOfRangeError:
                break
        Y_pred = np.concatenate(Y_pred)
        eval(Y_test, Y_pred)
        if epoch != params['n_epoch']:
            sess.run(iter_train_char.initializer, init_dict_train_char)
            sess.run(iter_train_pos.initializer, init_dict_train_pos)
            sess.run(iter_train_bigram.initializer, init_dict_train_bigram)
            sess.run(iter_test_char.initializer, init_dict_test_char)
            sess.run(iter_test_pos.initializer, init_dict_test_pos)
            sess.run(iter_test_bigram.initializer, init_dict_test_bigram)
    
    # Close session
    sess.close() 
