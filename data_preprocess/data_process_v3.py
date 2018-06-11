"""
preprocess for the amazon reviews
https://www.kaggle.com/snap/amazon-fine-food-reviews/data
reference : https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews
"""


import pandas as pd
from nltk.corpus import stopwords
import re
import pickle
import utils

def load_data():
    path = 'data/Reviews.csv'
    reviews = pd.read_csv(path)
    return reviews

def def_contractions():
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there'd": "there had",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where'd": "where did",
        "where's": "where is",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are"
    }
    return contractions

def clean_text(text, contractions, remove_stopwords = True):
    text = str(text).lower()
    # text = text.lower()
    # print(text)
    text = text.split()
    new_text = []
    # print(text)
    for word in text:
        # print(word)
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    # print(text)

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)


    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

def clean_summay_and_text(summaries, texts, contractions):
    clean_summaries = []
    clean_texts = []
    for sum in summaries:
        cleaned_line = clean_text(text=sum, contractions=contractions, remove_stopwords=False)
        clean_summaries.append(cleaned_line)
    #
    for text in texts:
        cleaned_line = clean_text(text=text, contractions=contractions, remove_stopwords=True)
        clean_texts.append(cleaned_line)

    return clean_summaries, clean_texts

def save_cleaned_data_as_pkl(sum, text):
    with open('data/cleaned_sum', 'wb') as file:
        pickle.dump(obj=sum, file=file)

    with open('data/cleaned_text', 'wb') as file:
        pickle.dump(obj=text, file=file)

def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file=file)
    return data

def count_words(count_dict, text):
    for sentence in text:
        words = sentence.split()
        for word in words:
            if word not in count_dict:
                count_dict[word] =1
            else:
                count_dict[word] += 1



def count_sum_and_text(summaries, texts):
    word_counts = {}
    count_words(count_dict=word_counts, text=summaries)
    count_words(count_dict=word_counts, text=texts)
    return word_counts

def build_word_int_dict(threshold, word_dict):
    spical_token = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
    vocab_to_int = {}
    int_to_vocab = {}

    word_count = 0
    for i, (word, count) in enumerate(word_dict.items()):
        if count > threshold:
            vocab_to_int[word] = word_count
            int_to_vocab[word_count] = word
            word_count += 1
    for code in spical_token:
        index = len(vocab_to_int)
        vocab_to_int[code] = index
        int_to_vocab[index] = code

    usage_ratio = round(len(vocab_to_int) / len(word_dict), 4) * 100

    print("Total number of unique words:", len(word_dict))
    print("Number of words we will use:", len(vocab_to_int))
    print("Percent of words we will use: {}%".format(usage_ratio))

    return int_to_vocab, vocab_to_int


def covert_to_ints(text, vocab_to_int, word_count, unk_count, is_eos = False):
    ints = []
    UNK = "<UNK>"
    EOS = "<EOS>"
    for sentence in text:
        sentence_int = []
        words = sentence.split()
        for word in words:
            word_count += 1
            if word in vocab_to_int:
                int_word = vocab_to_int[word]

                # int_word = vocab_to_int[UNK]
            else:
                int_word = vocab_to_int[UNK]

                # int_word = vocab_to_int[word]
                unk_count += 1
            sentence_int.append(int_word)
            if int_word > len(vocab_to_int):
                print(int_word)
        if is_eos:
            sentence_int.append(vocab_to_int[EOS])

        ints.append(sentence_int)
    return ints , word_count, unk_count




def covert_summaries_and_texts_to_int(summaries, texts, vocab_to_int):
    word_count = 0
    unk_count = 0
    int_summaries, word_count, unk_count = covert_to_ints(text=summaries, vocab_to_int=vocab_to_int, word_count=word_count,
                                                 unk_count= unk_count, is_eos=False)

    int_texts, word_count, unk_count = covert_to_ints(text=texts, vocab_to_int=vocab_to_int, word_count=word_count,
                                                 unk_count=unk_count, is_eos=True)

    unk_percent = round(unk_count / word_count, 4) * 100
    print("Total number of words in headlines:", word_count)
    print("Total number of UNKs in headlines:", unk_count)
    print("Percent of words that are UNK: {}%".format(unk_percent))

    return int_summaries, int_texts

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return lengths
    # return pd.DataFrame(lengths, columns=['counts'])

def get_sentence_length(summaries, texts):
    summaries_length = create_lengths(summaries)
    texts_length = create_lengths(texts)
    return summaries_length, texts_length

def unk_counter(sentence, vocab_to_int):
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count +=1

    return unk_count


#
# def sentence_filter(int_summaries, int_texts, texts_length, vocab_to_int):
#     sorted_summaries = []
#     sorted_texts = []
#     max_text_length = 84
#     max_summary_length = 13
#     min_length = 2
#     unk_text_limit = 1
#     unk_summary_limit = 0
#
#     for length in range(min(texts_length.counts), max_text_length):
#         for count, words in enumerate(int_summaries):
#             if (len(int_summaries[count]) >= min_length and
#                     len(int_summaries[count]) <= max_summary_length and
#                     len(int_texts[count]) >= min_length and
#                     unk_counter(int_summaries[count], vocab_to_int) <= unk_summary_limit and
#                     unk_counter(int_texts[count], vocab_to_int) <= unk_text_limit and
#                     length == len(int_texts[count])
#             ):
#                 sorted_summaries.append(int_summaries[count])
#                 sorted_texts.append(int_texts[count])
#
#     # for sen in sorted_texts:
#     #     print(len(sen))
#     # Compare lengths to ensure they match
#     print(len(sorted_summaries))
#     print(len(sorted_texts))
#     return sorted_summaries, sorted_texts

def length_filter(texts, summaries):
    max_text_length = 80
    min_text_length = 10
    max_summayies_length = 50
    min_summaries_length = 2
    new_texts = []
    new_summaries = []
    for i in range(len(texts)):
        text_length = len(texts[i])
        summary_length = len(summaries[i])
        if text_length >= min_text_length and text_length <= max_text_length:
            if summary_length >= min_summaries_length and summary_length <= max_summayies_length:
                new_texts.append(texts[i])
                new_summaries.append(summaries[i])
    return new_texts, new_summaries



def save_pkl(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(obj=data, file=file)

def split_data(x, y, train_fraction):
    data_set = DataSet()
    num_data = len(x)
    val_fraction = (1 - train_fraction) / 2
    test_fraction = (1 - train_fraction) / 2
    train_end_index = int(train_fraction * num_data)
    val_start_index = train_end_index
    val_end_index = val_start_index + int(val_fraction * num_data)
    test_start_index = val_end_index

    data_set.train_x = x[:train_end_index]
    data_set.val_x = x[val_start_index: val_end_index]
    data_set.test_x = x[test_start_index:]

    data_set.train_y = y[:train_end_index]
    data_set.val_y = y[val_start_index: val_end_index]
    data_set.test_y = y[test_start_index:]

    return data_set



class Path():
    base = "data/"
    cleaned_texts = base + "cleaned_text"
    cleaned_sum = base + "cleaned_sum"
    int_texts = base + "int_texts"
    int_summaries = base + "int_summaries"
    long_length_data_set = base + "long_length_data_set"



class DataSet():
    def __init__(self):
        train_x = None
        train_y = None
        val_x = None
        val_y = None
        test_x = None
        test_y = None

def main():
    # threshold = 2
    # cleaned_texts = load_pkl(Path.cleaned_texts)
    # cleaned_sum = load_pkl(Path.cleaned_sum)
    # word_counts = count_sum_and_text(texts=cleaned_texts, summaries=cleaned_sum)
    # print('size of vocabulary', len(word_counts))
    # int_to_vocab, vocab_to_int = build_word_int_dict(threshold=threshold, word_dict=word_counts)
    # int_summaries, int_texts = covert_summaries_and_texts_to_int(texts=cleaned_texts,
    #                                                              summaries=cleaned_sum,
    #                                                              vocab_to_int=vocab_to_int)
    # save_pkl(Path.int_texts, int_texts)
    # save_pkl(Path.int_summaries, int_summaries)

    int_texts = load_pkl(Path.int_texts)
    int_summaries = load_pkl(Path.int_summaries)

    # for text in int_texts:
    #     print(len(text))
    print(len(int_texts))
    int_texts, int_summaries = length_filter(int_texts, int_summaries)
    print(len(int_texts))
    # for t in int_texts:
    #     print(len(t))
    data_set = split_data(int_texts[:100000], int_summaries[:100000], train_fraction=0.8)
    """
    split data
    sort by length
    """

    utils.quick_sort_iterative(data_set.train_x, 0, len(data_set.train_x) - 1, data_set.train_y)
    utils.quick_sort_iterative(data_set.val_x, 0, len(data_set.val_x) - 1, data_set.val_y)
    utils.quick_sort_iterative(data_set.test_x, 0, len(data_set.test_x) - 1, data_set.test_y)
    for x in data_set.train_x:
        print(len(x))

    save_pkl(Path.long_length_data_set, data_set)


main()
