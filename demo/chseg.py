import re


def load_data():
    # B: 0, M: 1, E: 2, S: 3
    char2idx = {'<pad>':0}
    idx2char = {0:'<pad>'}
    char_idx = 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    def preprocess(path):
        text = open(path).read()
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        return text

    def build_y(chars, ys):
        if len(chars) == 1:
            ys.append(3)
        else:
            if i == 0:
                ys.append(0)
            elif i == len(chars) - 1:
                ys.append(2)
            else:
                ys.append(1)
    
    text = preprocess('pku_training.utf8')
    cutoff = int(0.8 * len(text))
    segs_train = text[:cutoff].split()
    segs_test = text[cutoff:].split()

    for seg in segs_train:
        chars = list(seg)
        for i, char in enumerate(chars):
            # handle x
            if char not in char2idx:
                char2idx[char] = char_idx
                idx2char[char_idx] = char
                char_idx += 1
            x_train.append(char2idx[char])
            # handle y
            build_y(chars, y_train)

    char2idx['<unknown>'] = char_idx

    for seg in segs_test:
        chars = list(seg)
        for i, char in enumerate(chars):
            # handle x
            if char in char2idx:
                x_test.append(char2idx[char])
            else:
                x_test.append(char_idx)
            # handle y
            build_y(chars, y_test)
    
    return x_train, y_train, x_test, y_test, len(char2idx), char2idx, idx2char
