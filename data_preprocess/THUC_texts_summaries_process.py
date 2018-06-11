import pickle
import os
import re
import jieba
import numpy as np
import quick_sort
from data_model import DataSet

def save_pkl(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def load_pkl(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
        return data


def clean_text(sentence):
    sentence = re.sub('[a-zA-z：？，。0-9《》！:(),.“”□、（）■]', '', sentence)
    sentence = re.sub('\u3000', '', sentence)
    sentence = re.sub(' ', '', sentence)
    sentence = re.sub('-', '', sentence)
    sentence = re.sub('\n', '', sentence)
    return sentence

def clean_summary(sentence):
    sentence = re.sub('[a-zA-z：？，。《》！:(),.]', '', sentence)
    sentence = re.sub('\u3000', '', sentence)
    sentence = re.sub(' ', '', sentence)
    sentence = re.sub('-', '', sentence)
    sentence = re.sub('\n', '', sentence)
    return sentence




def split_texts_and_summaries(folder_path, amount_of_data, max_length, min_length):
    summaries = []
    texts = []
    vocab_dict = {}


    for root_path in folder_path:
        file_list = os.listdir(root_path)
        label = root_path[-2:]
        print(label)
        # for i in range(len(file_list)):
        for i in range(amount_of_data):
            # if i % 1000 == 0:
            #     print(i)
            file = file_list[i]
            path = os.path.join(root_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                data = f.readlines()
                text = ''
                for i in range(1, len(data)):
                    text += data[i]
                summary = data[0]
                text = clean_text(text)
                summary = clean_summary(summary)

                text_iter = jieba.cut(text)
                summary_iter = jieba.cut(summary)
                cutted_text = []
                cutted_summary = []
                """
                if the word is in vocab dict, count+1
                """
                for word in text_iter:
                    cutted_text.append(word)
                    if word not in vocab_dict:
                        vocab_dict[word] = 1
                        # int_to_vocab[word_count] = word
                    else:
                        vocab_dict[word] += 1


                for word in summary_iter:
                    cutted_summary.append(word)
                    if word not in vocab_dict:
                        vocab_dict[word] = 1
                        # int_to_vocab[word_count] = word
                    else:
                        vocab_dict[word] += 1

                """
                remove too long and too short texts
                """
                if len(cutted_text) < max_length and len(cutted_text) > min_length:
                    texts.append(cutted_text)
                    summaries.append(cutted_summary)

    # for t in texts:
    #     print(t)
    # for s in summaries:
    #     print(s)
    # print(len(vocab_dict))

    return texts, summaries, vocab_dict

def trans_to_int(sentences, vocab_to_int):
    all_int_sentence = []
    for sentence in sentences:
        int_sentence = []
        for word in sentence:
            if word not in vocab_to_int:
                int_word = vocab_to_int[Tokens.UNK]
                int_sentence.append(int_word)
            else:
                int_word = vocab_to_int[word]
                int_sentence.append(int_word)
        # all_int_sentence.append(np.array(int_sentence))
        all_int_sentence.append(int_sentence)
    return all_int_sentence


def split_data(all_sentence, labels, train_fraction):
    # train_fraction = 0.9
    val_fraction = (1 - train_fraction) / 2
    test_fraction = (1 - train_fraction) / 2
    num_data = len(all_sentence)

    train_end_index = int(train_fraction * num_data)
    val_start_index = train_end_index
    val_end_index = val_start_index + int(val_fraction * num_data)
    test_start_index = val_end_index
    # test_end_index = val_end_index + int(test_fraction * num_data)

    train_x = all_sentence[:train_end_index]
    val_x = all_sentence[val_start_index : val_end_index ]
    test_x = all_sentence[test_start_index : ]

    train_y = labels[:train_end_index]
    val_y = labels[val_start_index: val_end_index]
    test_y = labels[test_start_index:]

    # print(len(train))
    # print(len(val))
    # print(len(test))
    # print(val_start_index)
    # print(val_end_index)
    # print(test_end_index)
    data_set = DataSet()
    data_set.train_x = train_x
    data_set.train_y = train_y
    data_set.val_x = val_x
    data_set.val_y = val_y
    data_set.test_x = test_x
    data_set.test_y = test_y
    return data_set


def vocab_filter(vocab_dict):

    vocab_to_int = {}
    int_to_vocab = {}
    word_id = 0
    for vocab, count in vocab_dict.items():
        # print(vocab, count)
        if count >= 3:
            if vocab not in vocab_to_int:
                vocab_to_int[vocab] = word_id
                int_to_vocab[word_id] = vocab
                word_id += 1
    return vocab_to_int, int_to_vocab




def add_tokens(vocab_to_int, int_to_vocab):
    codes = [Tokens.GO, Tokens.EOS, Tokens.UNK, Tokens.PAD]
    id = len(vocab_to_int)
    for code in codes:
        vocab_to_int[code] = id
        int_to_vocab[id] = code
        id += 1
    return vocab_to_int, int_to_vocab

def add_eos_token(summaries):
    for i in range(len(summaries)):
        summaries[i].append(Tokens.EOS)

def build_vocab_dict(texts, summaries):
    vocab_dict = {}
    for text in texts:
        for word in text:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1

    for sum in summaries:
        for word in sum:
            if word not in vocab_dict:
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1

    return vocab_dict



class Tokens():
    PAD = "<PAD>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    GO = "<GO>"



class Path():
    base = 'THUC_texts_summaries_data/'
    texts = base + 'texts.pkl'
    summaries = base + 'summaries.pkl'
    vocab_dict = base + 'vocab_dice.pkl'
    vocab_to_int = base + 'vocab_to_int.pkl'
    int_to_vocab = base + 'int_to_vocab.pkl'
    int_texts = base + 'int_texts.pkl'
    int_summaries = base + 'int_summaries.pkl'
    data_set = base + 'data_set.pkl'

def main():
    data_path = 'THUC_texts_summaries_data/'
    amount_of_data = 50000
    max_length = 200
    min_length = 10
    # folder_path = load_pkl('folder_path')
    # texts, summaries, vocab_dict = split_texts_and_summaries(folder_path, amount_of_data, max_length, min_length)
    # print('texts', len(texts))
    # print('summaries', len(summaries))
    # #
    # # save_pkl(Path.vocab_dict, vocab_dict)
    # save_pkl(Path.texts, texts)
    # save_pkl(Path.summaries, summaries)

    texts = load_pkl(data_path + "texts.pkl")
    summaries = load_pkl(data_path + "summaries.pkl")
    vocab_dict = build_vocab_dict(texts, summaries)
    print("vocab dict", len(vocab_dict))
    vocab_to_int, int_to_vocab = vocab_filter(vocab_dict)
    print("vocab to int", len(vocab_to_int))
    print("int to vocab", len(int_to_vocab))
    vocab_to_int, int_to_vocab = add_tokens(vocab_to_int, int_to_vocab)
    print(len(vocab_to_int))
    print(len(int_to_vocab))
    add_eos_token(summaries)
    # for i in range(10):
    #     print(summaries[i])

    int_texts = trans_to_int(texts, vocab_to_int)
    # for i in range(10):
    #     print(int_texts[i])
    int_summaries = trans_to_int(summaries, vocab_to_int)

    data_set = split_data(int_texts, int_summaries, train_fraction=0.9)

    # quick_sort.quickSortIterative(data_set.train_x, 0, len(data_set.train_x) - 1, data_set.train_y)
    # quick_sort.quickSortIterative(data_set.val_x, 0, len(data_set.val_x) - 1, data_set.val_y)
    # quick_sort.quickSortIterative(data_set.test_x, 0, len(data_set.test_x) - 1, data_set.test_y)

    save_pkl(Path.data_set, data_set)
    save_pkl(Path.vocab_to_int, vocab_to_int)
    save_pkl(Path.int_to_vocab, int_to_vocab)
    # for x in data_set.train_x:
    #     print(len(x))



    #
    # vocab_dict = load_pkl(Path.vocab_dict)
    # print(len(vocab_dict))
    #
    #
    # print("vocab_dict", len(vocab_dict))

    # vocab_to_int, int_to_vocab = vocab_filter(vocab_dict)
    # print(len(vocab_to_int))
    # vocab_to_int, int_to_vocab = add_tokens(vocab_to_int, int_to_vocab)
    # print(len(vocab_to_int))
    # print(int_to_vocab[2700])
    # print(int_to_vocab[2701])

    # save_pkl(data_path + "vocab_to_int.pkl", vocab_to_int)
    # save_pkl(data_path + "int_to_vocab.pkl", int_to_vocab)

    # import operator

    # # print(vocab_to_int['<GO>'])
    # # print(sorted(dict(vocab_to_int)))
    # sorted_x = sorted(int_to_vocab.items(), key=operator.itemgetter(1))
    # print(sorted_x)

    # vocab_to_int = load_pkl(data_path + "vocab_to_int.pkl")
    # int_to_vocab = load_pkl(data_path + "int_to_vocab.pkl")
    # texts = load_pkl(data_path + "texts.pkl")
    # summaries = load_pkl(data_path + "summaries.pkl")
    #
    # add_eos_token(summaries)
    # for i in range(10):
    #     print(summaries[i])
    # int_texts = trans_to_int(texts, vocab_to_int)
    # for i in range(10):
    #     print(int_texts[i])
    # int_summaries = trans_to_int(summaries, vocab_to_int)
    # print(len(int_texts))
    # print(len(int_summaries))
    # #
    # #
    # data_set = split_data(int_texts, int_summaries, train_fraction=0.9)
    # #
    # save_pkl(data_path + 'data_set', data_set)
    # data_set = load_pkl(data_path + 'data_set')
    # print(len(data_set.train_x))
    # print(len(data_set.train_y))
    # for i in range(3):
    #     print(data_set.train_x[i])
    #     print(data_set.train_y[i])
    # a = [1, 3, 6]
    # print(a)
    # b = []
    # b.append(1)
    # b.append(3)
    # b.append(99)
    # print(b)







main()