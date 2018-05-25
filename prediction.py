import tensorflow as tf
import pickle
from model_v2 import Seq2SeqModel
from rouge_helper import ROUGEHelper
import numpy as np


def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    # max_sentence = max([len(sentence.split()) for sentence in sentence_batch])
    # print(max_sentence)
    #
    # pad_int = vocab_to_int['<PAD>']
    # buf = pad_int * np.ones(shape=(batch_size, max_sentence), dtype=np.int32)
    # for line_index, line in enumerate(sentence_batch):
    #     line_buf = buf[line_index]
    #
    #     for word_index, word in enumerate(line.split()):
    #         print(word)
    #         int_word = vocab_to_int[word]
    #         # line_buf[word_index] = word
    # print(buf)


    # return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size, vocab_to_int):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch, vocab_to_int))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch, vocab_to_int))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_data():
    base_path = 'data/'
    # summaries = load_pkl(base_path + 'sorted_summaries')
    # texts = load_pkl(base_path + 'sorted_texts')
    # summaries = load_pkl(base_path + 'min_summaries')
    # texts = load_pkl(base_path + 'min_texts')
    summaries = load_pkl(base_path + 'min_test_summaries')
    texts = load_pkl(base_path + 'min_test_texts')
    vocab_to_int = load_pkl(base_path + 'vocab_to_int.pkl')

    return summaries, texts, vocab_to_int

def id_to_vocab(data, int_to_vovab):
    sentence = []
    for id in data:
        word = int_to_vovab[id]
        sentence.append(word)

    return sentence

def get_rouge_score():
    summaries, texts, vocab_to_int = load_data()
    int_to_vocab = load_pkl('data/' + 'int_to_vocab.pkl')


    model = Seq2SeqModel(vocab_to_int)
    # model.set_batch_size(1)
    model.build_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        # model.set_batch_size(1)
        helper = ROUGEHelper()
        scores = []
        batch_iter = get_batches(summaries=summaries, texts=texts, batch_size=256, vocab_to_int=vocab_to_int)
        for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:


            # feed = {model.input_data: pad_texts_batch,
            #         model.summary_length: pad_summaries_batch,
            #         model.text_length: pad_texts_lengths}
            feed = {model.input_data: pad_texts_batch,
                    model.targets: pad_summaries_batch,
                    model.summary_length: pad_summaries_lengths,
                    model.text_length: pad_texts_lengths}
            inference_logitis = sess.run(model.inference_logits, feed_dict=feed)
            # print('inference_logitis', inference_logitis)

            batch_score = helper.get_batch_score(inference_logitis, pad_summaries_batch)
            print(batch_score)
            scores.append(batch_score)

        print('score = ', np.mean(scores))


def prediction():
    summaries, texts, vocab_to_int = load_data()
    int_to_vocab = load_pkl('data/' + 'int_to_vocab.pkl')
    print(int_to_vocab)


    model = Seq2SeqModel(vocab_to_int)
    # model.set_batch_size(1)
    model.build_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        # model.set_batch_size(1)
        text_id = texts[0:256]
        sum_id = summaries[0:256]

        # text = int_to_vocab(ids=text_id, int_to_vocab=int_to_vocab)
        # sumXX = int_to_vocab(sum_id, int_to_vocab)
        # print(text_id)
        # print(sum_id)
        batch_size = 256
        length = []
        for text in text_id:
            length.append(len(text))

        # feed = {model.input_data: [text_id] * batch_size,
        #         model.summary_length: [len(text_id)] * batch_size,
        #         model.text_length: [len(text_id)] * batch_size
        #         }
        feed = {model.input_data: text_id,
                model.summary_length: length,
                model.text_length: length}
        inference_logitis = sess.run(model.inference_logits, feed_dict=feed)
        print('inference_logitis', inference_logitis)


        # text = id_to_vocab(text_id, int_to_vocab)
        # summary = id_to_vocab(sum_id, int_to_vocab)
        # predic = id_to_vocab(inference_logitis, int_to_vocab)
        #
        # print(text)
        # print(summary)
        # print(predic)
        #
        helper = ROUGEHelper()
        score = helper.get_batch_score(inference_logitis, sum_id)
        print(score)


# prediction()
get_rouge_score()