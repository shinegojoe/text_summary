import pandas as pd
from nltk.corpus import stopwords
import re
import pickle
import operator
from utils import BinaryTreeSort
import utils

class DataSet():
    def __init__(self):
        train_x = None
        train_y = None
        val_x = None
        val_y = None
        test_x = None
        test_y = None



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

def load_pkl_file():
    with open('data/cleaned_sum', 'rb') as file:
        summaries = pickle.load(file=file)

    with open('data/cleaned_text', 'rb') as file:
        texts = pickle.load(file=file)

    return summaries, texts

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

# def sentence_filter(sum_sentence, text_sentence, vocab_to_int,
#         summary_min_length, summary_max_length, summary_unk_limit, text_min_length, text_max_length, text_unk_limit):
#     text_unk_count = unk_counter(sentence=text_sentence, vocab_to_int=vocab_to_int)
#     sum_unk_count = unk_counter(sentence=sum_sentence, vocab_to_int=vocab_to_int)
#
#     text_sentence_length = len(text_sentence)
#     sum_sentence_length = len(sum_sentence)
#     if text_sentence_length > text_min_length and text_sentence_length < text_max_length and text_unk_count < text_unk_limit:
#         if sum_sentence_length > summary_min_length and sum_sentence_length < summary_max_length \
#             and sum_sentence_length < summary_unk_limit:
#             pass




def sentence_filter(int_summaries, int_texts, texts_length, vocab_to_int):
    sorted_summaries = []
    sorted_texts = []
    max_text_length = 84
    max_summary_length = 13
    min_length = 2
    unk_text_limit = 1
    unk_summary_limit = 0

    for length in range(min(texts_length.counts), max_text_length):
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                    len(int_summaries[count]) <= max_summary_length and
                    len(int_texts[count]) >= min_length and
                    unk_counter(int_summaries[count], vocab_to_int) <= unk_summary_limit and
                    unk_counter(int_texts[count], vocab_to_int) <= unk_text_limit and
                    length == len(int_texts[count])
            ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])

    # for sen in sorted_texts:
    #     print(len(sen))
    # Compare lengths to ensure they match
    print(len(sorted_summaries))
    print(len(sorted_texts))
    return sorted_summaries, sorted_texts


def sentence_filter_v2(texts, summaries, vocab_to_int):
    filterd_summaries = []
    filtered_texts = []
    # max_text_length = 84
    max_text_length = 250
    max_summary_length = 13
    min_length = 2
    text_min_length = 50

    unk_text_limit = 1
    unk_summary_limit = 0
    for i in range(len(texts)):
        text = texts[i]
        summary = summaries[i]
        text_length = len(text)
        summary_length = len(summary)
        unk_of_text = unk_counter(sentence=text, vocab_to_int=vocab_to_int)
        unk_of_summary = unk_counter(sentence=summary, vocab_to_int=vocab_to_int)

        if text_length <= max_text_length and text_length >= text_min_length and unk_of_text <= unk_text_limit:
            if summary_length <= max_summary_length and summary_length >= min_length and unk_of_summary <= unk_summary_limit:
                filtered_texts.append(text)
                filterd_summaries.append(summary)

    return filtered_texts, filterd_summaries






def save_file_as_pkl(file_name, data):
    with open('data/' + file_name, 'wb') as file:
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

def sort_data_by_length(x, y):
    data_length = len(x) - 1
    # print(zz_length)
    for j in range(data_length):
        index = j
        # print('index', index)

        for i in range(data_length - index):
            # print(i)
            q1 = len(x[i])
            q2 = len(x[i + 1])
            # print('q1', q1)
            # print('q2', q2)
            if q1 > q2:

                buf_x = x[i]
                x[i] = x[i + 1]
                x[i + 1] = buf_x
                buf_y = y[i]
                y[i] = y[i + 1]
                y[i + 1] = buf_y



    return x, y

def binary_tree_sort(data):
    bts = BinaryTreeSort(data[0])
    for i in range(0, len(data)):
        bts.insert(data[i])

    sorted_data= []
    bts.print_tree(sorted_data)
    return sorted_data


def main():

    # reviews = load_data()
    # print(reviews.shape)
    # # summaries = reviews['Summary']
    # # texts = reviews['Text']
    # summaries = reviews.Summary
    # texts = reviews.Text
    # # #
    # print(summaries.head())
    # print(texts.head())
    # contractions = def_contractions()
    # clean_summaries, clean_texts = clean_summay_and_text(summaries=summaries, texts=texts, contractions=contractions)
    # for i in range(5):
    #     print(clean_summaries[i])
    #     print(clean_texts[i])
    # save_cleaned_data_as_pkl(clean_summaries, clean_texts)


    threshold = 2
    clean_summaries, clean_texts = load_pkl_file()
    for i in range(5):
        print(clean_summaries[i])
        print(clean_texts[i])
    word_counts = count_sum_and_text(summaries=clean_summaries, texts=clean_texts)
    print('size of vocabulary', len(word_counts))
    int_to_vocab, vocab_to_int = build_word_int_dict(threshold=threshold, word_dict=word_counts)
    int_summaries, int_texts = covert_summaries_and_texts_to_int(summaries=clean_summaries, texts=clean_texts,
                                                                 vocab_to_int=vocab_to_int)
    summaries_length, texts_length = get_sentence_length(summaries=int_summaries, texts=int_texts)

    summaries_length = pd.DataFrame(summaries_length, columns=['counts'])
    texts_length = pd.DataFrame(texts_length, columns=['counts'])
    # save_file_as_pkl(file_name="summaries_length.pkl", data=summaries_length)
    # save_file_as_pkl(file_name="texts_length.pkl", data=texts_length)
    save_file_as_pkl(file_name="int_to_vacab.pkl", data=int_to_vocab)
    save_file_as_pkl(file_name="vocab_to_int.pkl", data=vocab_to_int)
    # print(vocab_to_int)


    # print(summaries_length.describe())
    # print(texts_length.describe())

    # summary_min_length = 0
    # summary_max_length = 10
    # summary_unk_limit = 0
    #
    # text_min_length = 0
    # text_max_length = 100
    # text_unk_limit = 0
    # print("length of summaries", len(int_summaries))
    # print("length of texts", len(int_texts))
    # for i in range(len(int_summaries)):
    #     sum_sentence = int_summaries[i]
    #     text_sentence = int_texts[i]
    data_set = split_data(x=int_texts, y=int_summaries, train_fraction=0.9)
    # sorted_summaries, sorted_texts = sentence_filter(int_summaries=int_summaries, int_texts=int_texts, texts_length=texts_length,
    #                                                vocab_to_int=vocab_to_int)

    # print('train_x', len(data_set.train_x))
    # print('train_y', len(data_set.train_y))
    for x in data_set.train_x:
        x_len = len(x)
        if x_len < 50:
            print(x_len)
    filterd_train_x, filterd_train_y = sentence_filter_v2(texts=data_set.train_x, summaries=data_set.train_y, vocab_to_int=vocab_to_int)
    for x in filterd_train_x:
        x_len = len(x)
        if x_len < 50:
            print('filtered', x_len)
    print('filtered_train_x', len(filterd_train_x))
    print('filtered_train_y', len(filterd_train_y))
    filterd_val_x, filterd_val_y = sentence_filter_v2(texts=data_set.val_x, summaries=data_set.val_y,
                                                          vocab_to_int=vocab_to_int)

    filterd_test_x, filterd_test_y = sentence_filter_v2(texts=data_set.test_x, summaries=data_set.test_y,
                                                          vocab_to_int=vocab_to_int)

    save_file_as_pkl(file_name='filterd_test_x', data=filterd_test_x)
    save_file_as_pkl(file_name='filterd_test_y', data=filterd_test_y)


    # for text in filterd_test_x:
    #     print(len(text))
    # sorted_train_x, sorted_train_y = sort_data_by_length(x=filterd_train_x, y=filterd_train_y)
    # sorted_val_x, sorted_val_y = sort_data_by_length(x=filterd_val_x, y=filterd_val_y)
    # sorted_test_x, sorted_test_y = sort_data_by_length(x=filterd_test_x, y=filterd_test_y)
    # print(filterd_test_x[0])
    # sorted_train_x, sorted_train_y = utils.quick_sort(lst=filterd_train_x,lo=0,hi=len(filterd_train_x), y=filterd_train_y)
    # utils.quick_sort(lst=filterd_test_x)

    # for i in range(len(sorted_test_x)):
    #     print(len(sorted_test_x), len(sorted_test_y))



    # sorted_val_x, sorted_val_y = binary_tree_sort(filterd_val_x)
    # sorted_test_x, sorted_test_y = binary_tree_sort(filterd_test_x)
    utils.quick_sort_iterative(filterd_train_x, 0, len(filterd_train_x) - 1, filterd_train_y)
    utils.quick_sort_iterative(filterd_val_x, 0, len(filterd_val_x) - 1, filterd_val_y)
    utils.quick_sort_iterative(filterd_test_x, 0, len(filterd_test_x) - 1, filterd_test_y)


    sorted_data = DataSet()
    sorted_data.train_x = filterd_train_x
    sorted_data.train_y = filterd_train_y
    sorted_data.val_x = filterd_val_x
    sorted_data.val_y = filterd_val_y
    sorted_data.test_x = filterd_test_x
    sorted_data.test_y = filterd_test_y
    save_file_as_pkl(file_name='sorted_data_set', data=data_set)

    # for text in sorted_test_x:
    #     print(len(text))
    #

    # save_sorted_cleaned_data_as_pkl(sum=sorted_summaries, text=sorted_texts)



    # save_file_as_pkl(file_name='sorted_summaries', data=sorted_summaries)
    # save_file_as_pkl(file_name='sorted_texts', data=sorted_texts)


def test():
    with open('data/filterd_test_x', 'rb') as file:
        filterd_test_x = pickle.load(file=file)

    with open('data/filterd_test_y', 'rb') as file:
        filterd_test_y = pickle.load(file=file)

    # x = [3,1,67,33,78,2]
    # utils.quick_sort_iterative(x, 0, len(x)-1)
    # print(x)
    utils.quick_sort_iterative(filterd_test_x, 0, len(filterd_test_x) - 1)
    for x in filterd_test_x:
        print(len(x))


    # print(filterd_test_x[0])
    # print(filterd_test_y[0])
    #
    # print(len(filterd_test_x))
    # print(len(filterd_test_y))

    # x, y = utils.quick_sort(filterd_test_x, 0, len(filterd_test_x), filterd_test_y)

    # bts = BinaryTreeSort(filterd_test_x[0])
    # for i in range(0, len(filterd_test_x)):
    #     bts.insert(filterd_test_x[i])
    #
    # sorted_x = []
    # bts.print_tree(sorted_x)
    # for x in sorted_x:
    #     print(len(x))

    # for i in range(len(x)):
    #     print(len(x[i]), len(y[i]))



















if __name__=="__main__":
    main()
# test()
# import nltk
# nltk.download()