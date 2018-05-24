# from model import Seq2SeqModel
from model_v2 import Seq2SeqModel

import pickle
import numpy as np
import tensorflow as tf

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
    summaries = load_pkl(base_path + 'sorted_summaries')
    texts = load_pkl(base_path + 'sorted_texts')
    vocab_to_int = load_pkl(base_path + 'vocab_to_int.pkl')

    return summaries, texts, vocab_to_int

def train():
    epochs = 5
    summaries, texts, vocab_to_int = load_data()
    # print(len(vocab_to_int))
    # print(summaries[:5])
    # print(texts[:3])



    model = Seq2SeqModel(vocab_to_int)
    model.build_model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print('epoch', epoch)
            batch_iter = get_batches(summaries=summaries, texts=texts, batch_size=256, vocab_to_int=vocab_to_int)
            for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:
                # print('pad_summaries_batch', pad_summaries_batch.shape)
                # print('pad_texts_batch', pad_texts_batch.shape)
                # print(pad_summaries_batch)
                feed = {model.input_data:pad_texts_batch,
                        model.targets:pad_summaries_batch,
                        model.summary_length:pad_summaries_lengths,
                        model.text_length:pad_texts_lengths}
                sess.run(model.train_op, feed_dict=feed)

            loss = sess.run(model.cost, feed_dict=feed)
            print('loss', loss)


    # print(pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths)



train()

# summaries, texts, vocab_to_int = load_data()
#
# for xx in texts:
#     print(xx.split())