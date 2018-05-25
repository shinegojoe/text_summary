import pickle

class DataSet():
    def __init__(self):
        train_x = None
        train_y = None
        val_x = None
        val_y = None
        test_x = None
        test_y = None

class Vocab_dict():
    def __init__(self):
        self.vocab_to_int = None
        self.int_to_vocab = None


def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_data_set(config):
    data_set = DataSet()
    data_set.train_x = load_pkl(config.train_texts_path)
    data_set.train_y = load_pkl(config.train_summaries_path)
    data_set.test_x = load_pkl(config.test_texts_path)
    data_set.test_y = load_pkl(config.test_summaries_path)

    return data_set

def load_vocab_dict(config):
    vocab_dict = Vocab_dict()
    vocab_dict.int_to_vocab = load_pkl(config.int2vocab_path)
    vocab_dict.vocab_to_int = load_pkl(config.vocab2int_path)

    return vocab_dict

