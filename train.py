import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def pipeline_train(X, y, sess, params):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    y_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    init_dict = {X_ph: X, y_ph: y}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict

def pipeline_test(X, sess, params):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    init_dict = {X_ph: X}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict

def load_word_tag_ix(word_to_ix_path, tag_to_ix_path):
    with open(word_to_ix_path, 'rb') as wordf:
            word_to_ix = pickle.load(wordf)
    with open(tag_to_ix_path, 'rb') as tagf:
            tag_to_ix = pickle.load(tagf)
    return word_to_ix, tag_to_ix


def rnn_cell(params):
    return tf.nn.rnn_cell.GRUCell(params['hidden_dim'],
        kernel_initializer=tf.orthogonal_initializer())

def clip_grads(loss, params):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, params['clip_norm'])
    return zip(clipped_grads, variables)

def forward(x, reuse, is_training, params):
    with tf.variable_scope('model', reuse=reuse):
        x = tf.contrib.layers.embed_sequence(x, params['vocab_size'], params['hidden_dim'])
        x = tf.layers.dropout(x, 0.1, training=is_training)
        
        bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            rnn_cell(params), rnn_cell(params), x, dtype=tf.float32)
        x = tf.concat(bi_outputs, -1)
        
        logits = tf.layers.dense(x, params['n_class'])
    return logits

if __name__ == '__main__':
    
    params = {
        'seq_len': 80,
        'batch_size': 128,
        'n_class': 9,
        'hidden_dim': 64,
        'clip_norm': 5.0,
        'lr': {'start': 1e-1, 'end': 5e-4},
        'n_epoch': 4,
        'display_step': 10,
    }
    word_to_ix_path = 'data/word_to_ix.pkl'
    tag_to_ix_path = 'data/tag_to_ix.pkl'
    X_train_path = 'data/X_train.npy'
    Y_train_path = 'data/Y_train.npy'
    X_test_path = 'data/X_test.npy'
    Y_test_path = 'data/Y_test.npy'
    checkpoint_dir = 'model'
    word_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, tag_to_ix_path)
    
    params['vocab_size'] = len(word_to_ix)
    ix_to_tag = {v:k for k, v in tag_to_ix.items()}
    
    # Load data
    X_train, Y_train = np.load(X_train_path), np.load(Y_train_path)
    X_test, Y_test = np.load(X_test_path), np.load(Y_test_path)

    
    params['lr']['steps'] = len(X_train) // params['batch_size']
    sess = tf.Session()
    iter_train, init_dict_train = pipeline_train(X_train, Y_train, sess, params)
    iter_test, init_dict_test = pipeline_test(X_test, sess, params)


    ops = {}

    X_train_batch, y_train_batch = iter_train.get_next()
    X_test_batch = iter_test.get_next()

    logits_tr = forward(X_train_batch, False, True, params)
    logits_te = forward(X_test_batch, True, False, params)

    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        logits_tr, y_train_batch, tf.count_nonzero(X_train_batch, 1))

    ops['loss'] = tf.reduce_mean(-log_likelihood)

    ops['global_step'] = tf.Variable(0, trainable=False)

    ops['lr'] = tf.train.exponential_decay(
        params['lr']['start'], ops['global_step'], params['lr']['steps'],
        params['lr']['end']/params['lr']['start'], staircase=False)

    ops['train'] = tf.train.AdamOptimizer(ops['lr']).apply_gradients(
        clip_grads(ops['loss'], params), global_step=ops['global_step'])

    ops['crf_decode'] = tf.contrib.crf.crf_decode(
        logits_te, trans_params, tf.count_nonzero(X_test_batch, 1))[0]

    sess.run(tf.global_variables_initializer())
    for epoch in range(1, params['n_epoch']+1):
        while True:
            try:
                _, step, loss, lr = sess.run([ops['train'],
                                              ops['global_step'],
                                              ops['loss'],
                                              ops['lr']])
            except tf.errors.OutOfRangeError:
                break
            else:
                if step % params['display_step'] == 0 or step == 1:
                    print("Epoch %d | Step %d | Loss %.3f | LR: %.4f" % (epoch, step, loss, lr))
        
        Y_pred = []
        while True:
            try:
                Y_pred.append(sess.run(ops['crf_decode']))
            except tf.errors.OutOfRangeError:
                break
        Y_pred = np.concatenate(Y_pred)
        
        if epoch != params['n_epoch']:
            sess.run(iter_train.initializer, init_dict_train)
            sess.run(iter_test.initializer, init_dict_test)
    
    # Eval(confusion matrix)
    labels = [i for i in range(params['n_class'])]
    target_names = ['O', 'B-R', 'I-R', 'B-M', 'I-M', 'B-S', 'I-S','B-W','I-W']
    print(classification_report(Y_test.ravel(), Y_pred.ravel(), labels=labels, target_names=target_names))
     
