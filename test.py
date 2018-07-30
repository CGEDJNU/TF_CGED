import numpy as np
import tensorflow as tf
from train import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    params = {
            'seq_len': 80,
            'hidden_dim': 128,
            'n_class': 9
            } 
    sample1 = '据我所知，随着经济发展而富裕起来的娱乐环境的繁华，很多青少年的吸烟率逐渐增长。这些变化会引起的结果将能够影响一个国家未来。'
    sample2 = '据我所知，随着经济发展而富裕起来的娱乐环境的繁华，很多青少年的吸烟率逐渐增长。'
    
    model_path = 'model/model.ckpt'
    word_to_ix_path = 'data/word_to_ix.pkl'
    tag_to_ix_path = 'data/tag_to_ix.pkl'
    word_to_ix, tag_to_ix = load_word_tag_ix(word_to_ix_path, tag_to_ix_path)
    ix_to_tag = {v:k for k, v in tag_to_ix.items()}
    params['vocab_size'] = len(word_to_ix)
    
    # Load model
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model/model.ckpt'))
        x = np.atleast_2d([word_to_ix[w] for w in sample2] + [0]*(params['seq_len']-len(sample2)))

        ph = tf.placeholder(tf.int32, [None, params['seq_len']])
        logits = forward(ph, True, False, params)
        inference = tf.contrib.crf.crf_decode(logits, trans_params, tf.count_nonzero(ph, 1))[0]
        

        x = sess.run(inference, {ph: x})[0][:len(sample2)]
        res = ''
        for i, l in enumerate(x):
            res += sample2[i]+' '+ix_to_tag[l]
        print(res)
