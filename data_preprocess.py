import pandas as pd
from nltk.corpus import stopwords
import re
import pickle

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




def save_file_as_pkl(file_name, data):
    with open('data/' + file_name, 'wb') as file:
        pickle.dump(obj=data, file=file)







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
    save_file_as_pkl(file_name="summaries_length.pkl", data=summaries_length)
    save_file_as_pkl(file_name="texts_length.pkl", data=texts_length)
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
    sorted_summaries, sorted_texts = sentence_filter(int_summaries=int_summaries, int_texts=int_texts, texts_length=texts_length,
                                                   vocab_to_int=vocab_to_int)
    # save_sorted_cleaned_data_as_pkl(sum=sorted_summaries, text=sorted_texts)
    save_file_as_pkl(file_name='sorted_summaries', data=sorted_summaries)
    save_file_as_pkl(file_name='sorted_texts', data=sorted_texts)



















main()
# import nltk
# nltk.download()