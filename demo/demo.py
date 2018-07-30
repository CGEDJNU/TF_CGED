import chseg
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
params = {
    'seq_len': 50,
    'batch_size': 128,
    'n_class': 4,
    'hidden_dim': 128,
    'clip_norm': 5.0,
    'text_iter_step': 10,
    'lr': {'start': 5e-3, 'end': 5e-4},
    'n_epoch': 1,
    'display_step': 50,
}
def to_test_seq(*args):
    return [np.reshape(x[:(len(x)-len(x)%params['seq_len'])],
        [-1,params['seq_len']]) for x in args]

def iter_seq(x):
    return np.array([x[i: i+params['seq_len']] for i in range(
        0, len(x)-params['seq_len'], params['text_iter_step'])])

def to_train_seq(*args):
    return [iter_seq(x) for x in args]
def pipeline_train(X, y, sess):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    y_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    init_dict = {X_ph: X, y_ph: y}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict

def pipeline_test(X, sess):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(params['batch_size'])
    iterator = dataset.make_initializable_iterator()
    X_ph = tf.placeholder(tf.int32, [None, params['seq_len']])
    init_dict = {X_ph: X}
    sess.run(iterator.initializer, init_dict)
    return iterator, init_dict


x_train, y_train, x_test, y_test, params['vocab_size'], word2idx, idx2word = chseg.load_data()
import pdb
pdb.set_trace()
X_train, Y_train = to_train_seq(x_train, y_train)
X_test, Y_test = to_test_seq(x_test, y_test)

sess = tf.Session()
params['lr']['steps'] = len(X_train) // params['batch_size']

iter_train, init_dict_train = pipeline_train(X_train, Y_train, sess)
iter_test, init_dict_test = pipeline_test(X_test, sess)

def rnn_cell():
    return tf.nn.rnn_cell.GRUCell(params['hidden_dim'],
        kernel_initializer=tf.orthogonal_initializer())

def clip_grads(loss):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, params['clip_norm'])
    return zip(clipped_grads, variables)

def forward(x, reuse, is_training):
    with tf.variable_scope('model', reuse=reuse):
        x = tf.contrib.layers.embed_sequence(x, params['vocab_size'], params['hidden_dim'])
        x = tf.layers.dropout(x, 0.1, training=is_training)
        
        bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            rnn_cell(), rnn_cell(), x, dtype=tf.float32)
        x = tf.concat(bi_outputs, -1)
        
        logits = tf.layers.dense(x, params['n_class'])
    return logits
ops = {}

X_train_batch, y_train_batch = iter_train.get_next()
X_test_batch = iter_test.get_next()

logits_tr = forward(X_train_batch, reuse=False, is_training=True)
logits_te = forward(X_test_batch, reuse=True, is_training=False)

log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
    logits_tr, y_train_batch, tf.count_nonzero(X_train_batch, 1))

ops['loss'] = tf.reduce_mean(-log_likelihood)

ops['global_step'] = tf.Variable(0, trainable=False)

ops['lr'] = tf.train.exponential_decay(
    params['lr']['start'], ops['global_step'], params['lr']['steps'],
    params['lr']['end']/params['lr']['start'])

ops['train'] = tf.train.AdamOptimizer(ops['lr']).apply_gradients(
    clip_grads(ops['loss']), global_step=ops['global_step'])

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
print(classification_report(Y_test.ravel(), Y_pred.ravel(), target_names=['B','M','E','S']))

sample = '我来到大学读书，希望学到知识'
x = np.atleast_2d([word2idx[w] for w in sample] + [0]*(params['seq_len']-len(sample)))

ph = tf.placeholder(tf.int32, [None, params['seq_len']])
logits = forward(ph, reuse=True, is_training=False)
inference = tf.contrib.crf.crf_decode(logits, trans_params, tf.count_nonzero(ph, 1))[0]

x = sess.run(inference, {ph: x})[0][:len(sample)]
res = ''
for i, l in enumerate(x):
    c = sample[i]
    if l == 2 or l == 3:
        c += ' '
    res += c
print(res)
