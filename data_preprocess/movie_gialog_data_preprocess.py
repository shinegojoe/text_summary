"""
preprocess for the Cornell Movie--Dialogs Corpus
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
reference : https://tutorials.botsfloor.com/how-to-build-your-first-chatbot-c84495d4622d
"""

import pandas as pd
import re
import data_loader
from path import Path
from data_model import DataSet

def clean_text(data):
    cleaned_data = []
    for text in data:
        text = text.lower()

        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        cleaned_data.append(text)
    return cleaned_data

def length_filter(question, ans):
    filtered_question = []
    filtered_ans = []
    min_line_length = 2
    max_line_length = 20
    for i in range(len(question)):
        q = question[i].split()
        a = ans[i].split()
        len_q = len(q)
        len_a = len(a)
        if len_q >= min_line_length and len_q <= max_line_length:
            if len_a >= min_line_length and len_a <= max_line_length:
                filtered_question.append(q)
                filtered_ans.append(a)
    return filtered_question, filtered_ans
    # for line in :
    #     text = line.split()
    #     text_len = len(text)
    #     if text_len >= min_line_length and text_len <= max_line_length:
    #         filtered_data.append(text)


def build_vocab_dict(q, a):
    vocab = {}
    vocab_to_int = {}
    int_to_vocab = {}
    threshold = 10
    codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']
    for question in q:
        for word in question:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for answer in a:
        for word in answer:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    word_id = 0
    for word, count in vocab.items():
        if count >= threshold and word not in vocab_to_int:
            vocab_to_int[word] = word_id
            int_to_vocab[word_id] = word
            word_id += 1

    for token in codes:
        vocab_to_int[token] = word_id
        int_to_vocab[word_id] = token
        word_id += 1

    return vocab_to_int, int_to_vocab


# def remove_rare_words(vocab):
#     threshold = 10
#     count = 0
#     for k, v in vocab.items():
#         if v >= threshold:
#             count += 1
def add_EOS_for_ans(ans):
    for a in ans:
        a.extend(["<EOS>"])

def convert_to_int(data, vocab_to_int):
    int_data = []
    for line in data:
        sentence = []
        for word in line:
            if word not in vocab_to_int:
                sentence.append(vocab_to_int["<UNK>"])
            else:
                sentence.append(vocab_to_int[word])
        int_data.append(sentence)
    return int_data

def split_data(inputs, labels, train_fraction):
    # train_fraction = 0.9
    data_set = DataSet()
    val_fraction = (1 - train_fraction) / 2
    test_fraction = (1 - train_fraction) / 2
    num_data = len(inputs)

    train_end_index = int(train_fraction * num_data)
    val_start_index = train_end_index
    val_end_index = val_start_index + int(val_fraction * num_data)
    test_start_index = val_end_index
    # test_end_index = val_end_index + int(test_fraction * num_data)

    data_set.train_x = inputs[:train_end_index]
    data_set.val_x = inputs[val_start_index : val_end_index ]
    data_set.test_x = inputs[test_start_index : ]

    data_set.train_y = labels[:train_end_index]
    data_set.val_y = labels[val_start_index: val_end_index]
    data_set.test_y = labels[test_start_index:]


    return data_set

def main():

    # lines = open(Path.movie_lines, encoding='utf-8', errors='ignore').read().split('\n')
    # conv_lines = open(Path.movie_conversations, encoding='utf-8', errors='ignore').read().split('\n')
    # print(lines[:10])
    # print(conv_lines[:10])
    #
    # id2line = {}
    # for line in lines:
    #     _line = line.split(' +++$+++ ')
    #     if len(_line) == 5:
    #         id2line[_line[0]] = _line[4]
    #
    # convs = []
    # for line in conv_lines[:-1]:
    #     _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    #     convs.append(_line.split(','))
    #
    # # for x in id2line.items():
    # #     print(x)
    # print(convs[:10])
    # questions = []
    # answers = []
    #
    # for conv in convs:
    #     for i in range(len(conv) - 1):
    #         questions.append(id2line[conv[i]])
    #         answers.append(id2line[conv[i + 1]])
    #
    # cleaned_questions = clean_text(questions)
    # cleaned_answers = clean_text(answers)
    #
    # cleaned_questions, cleaned_answers = length_filter(cleaned_questions, cleaned_answers)
    #
    # data_loader.save_pkl(Path.question, cleaned_questions)
    # data_loader.save_pkl(Path.ans, cleaned_answers)

    cleaned_questions = data_loader.load_pkl(Path.question)
    cleaned_answers = data_loader.load_pkl(Path.ans)
    vocab_to_int, int_to_vocab = build_vocab_dict(cleaned_questions, cleaned_answers)
    data_loader.save_pkl(Path.vocab_to_int, vocab_to_int)
    data_loader.save_pkl(Path.int_to_vocab, int_to_vocab)
    print(len(cleaned_questions))
    print(len(cleaned_questions))
    add_EOS_for_ans(cleaned_answers)
    question_int = convert_to_int(cleaned_questions, vocab_to_int)
    ans_int = convert_to_int(cleaned_answers, vocab_to_int)

    print(len(question_int))
    print(len(ans_int))

    data_set = split_data(question_int, ans_int, 0.8)

    # data_loader.save_pkl(Path.question_int, question_int)
    # data_loader.save_pkl(Path.ans_int, ans_int)
    data_loader.save_pkl(Path.data_set, data_set)


    # a = ["1", "2", "3"]
    # a.extend(["<EOS>"])
    # print(a)
    # go_int = vocab_to_int["<GO>"]
    # print(go_int)
    # print(vocab_to_int["<GO>"])
    # print(vocab_to_int["<PAD>"])
    # print(vocab_to_int["<EOS>"])
    # print(vocab_to_int["<UNK>"])
    # print(len(cleaned_questions))
    # print(len(cleaned_answers))
    # for q in cleaned_questions:
    #     len_q = len(q)
    #     if len_q <2 or len_q >20:
    #         print("q ", len_q)
    #
    # for a in cleaned_answers:
    #     len_a = len(a)
    #     if len_a <2 or len_a >20:
    #         print("a ", len_a)










main()