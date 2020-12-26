import numpy as np
import textwrap
import time


def split_data(data_all, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=123):
    data_size = len(data_all['abstract'])
    # data_size = 50
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    test_size = int(data_size * test_ratio)

    np.random.seed(seed)
    idxs = np.random.permutation(data_size)

    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:-test_size]
    test_idxs = idxs[-test_size:]

    data_train = {
        'abstract': [data_all['abstract'][i] for i in train_idxs],
        'title': [data_all['title'][i] for i in train_idxs],
        'title_pos': [data_all['title_pos'][i] for i in train_idxs],
    }
    data_val = {
        'abstract': [data_all['abstract'][i] for i in val_idxs],
        'title': [data_all['title'][i] for i in val_idxs],
        'title_pos': [data_all['title_pos'][i] for i in val_idxs],
    }
    data_test = {
        'abstract': [data_all['abstract'][i] for i in test_idxs],
        'title': [data_all['title'][i] for i in test_idxs],
        'title_pos': [data_all['title_pos'][i] for i in test_idxs],
    }

    return data_train, data_val, data_test


def replace_special_tokens(sentence):
    sentence = sentence.replace('\\', '//')
    sentence = sentence.replace('{', '(')
    sentence = sentence.replace('}', ')')
    sentence = sentence.replace('^', '')
    return sentence


class arXivDataLoader():
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.size = len(data['abstract'])
        self.num_batch = (self.size + batch_size - 1) // batch_size
        self.reset()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def reset(self):
        self.idxs = np.random.permutation(self.size)
        self.position = 0

    def __next__(self):
        if self.position == self.size:
            self.reset()
            raise StopIteration()

        next_position = min(self.position + self.batch_size, self.size)
        idxs = self.idxs[self.position:next_position]
        batch = {
            'abstract': [self.data['abstract'][i] for i in idxs],
            'title': [self.data['title'][i] for i in idxs],
            'title_pos': [self.data['title_pos'][i] for i in idxs],
        }
        self.position = next_position

        return batch


def wrap_text(text, width):
    return '\n'.join(textwrap.wrap(text, width))


class TimeKeeper():
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = time.time()

    def get_eta(self, epoch):
        eta = (time.time() - self.start_time) * ((self.num_epochs - epoch) / epoch)
        eta_hour = int(eta / 3600)
        eta_min = int((eta - eta_hour * 3600) / 60)
        eta_sec = int(eta - eta_hour * 3600 - eta_min * 60)
        return eta_hour, eta_min, eta_sec
