import numpy as np

class BatchHelper():
    def pad_sentence_batch(self, sentence_batch, vocab_to_int):
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


    def get_batches(self, summaries, texts, batch_size, vocab_to_int):
        """Batch summaries, texts, and the lengths of their sentences together"""
        for batch_i in range(0, len(texts) // batch_size):
            start_i = batch_i * batch_size
            summaries_batch = summaries[start_i:start_i + batch_size]
            texts_batch = texts[start_i:start_i + batch_size]
            pad_summaries_batch = np.array(self.pad_sentence_batch(summaries_batch, vocab_to_int))
            pad_texts_batch = np.array(self.pad_sentence_batch(texts_batch, vocab_to_int))

            # Need the lengths for the _lengths parameters
            pad_summaries_lengths = []
            for summary in pad_summaries_batch:
                pad_summaries_lengths.append(len(summary))

            pad_texts_lengths = []
            for text in pad_texts_batch:
                pad_texts_lengths.append(len(text))

            yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

# def get_batches(summaries, texts, batch_size, vocab_to_int):
#     x = summaries
#     y = texts
#     n_batches = len(x) // batch_size
#     x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
#
#     for ii in range(0, len(x), batch_size):
#         batch_x = x[ii:ii + batch_size]
#         batch_y = y[ii:ii + batch_size]
#         batch_x_max_len = max([len(sentence) for sentence in batch_x])
#         batch_y_max_len = max([len(sentence) for sentence in batch_y])
#         # len_list = [[len(sentence)] for sentence in batch_x]
#         # max_len = max(len_list)
#
#         x_buf = vocab_to_int['<PAD>'] * np.ones(shape=(batch_size, batch_x_max_len), dtype=np.int32)
#         y_buf = vocab_to_int['<PAD>'] * np.ones(shape=(batch_size, batch_y_max_len), dtype=np.int32)
#         for i in range(batch_size):
#             x_row = batch_x[i]
#             x_buf_i = x_buf[i]
#             x_buf_i[:len(x_row)] = x_row[:]
#
#             y_row = batch_y[i]
#             y_buf_i = y_buf[i]
#             y_buf_i[:len(y_row)] = y_row[:]
#         # pad_x_lengths = [[len(sentence)] for sentence in buf]
#         pad_x_lengths = batch_x_max_len * np.ones(shape=(batch_size))
#         pad_y_lengths = batch_y_max_len * np.ones(shape=(batch_size))
#         # pad_y_lengths = [[len(label)] for label in batch_y]
#         yield x_buf, y_buf, np.array(pad_x_lengths), np.array(pad_y_lengths)