import abc
import tensorflow as tf
import numpy as np
from hyper_optimizer.hyper_optimizer import IHyperOptimizer
import batch_helper

class ITrainer(metaclass=abc.ABCMeta):

    # @abc.abstractmethod
    # def train(self):
    #     return NotImplemented

    # @abc.abstractmethod
    # def get_batches(self):
    #     return NotImplemented

    @abc.abstractmethod
    def set_score_helper(self):
        return NotImplemented

    @abc.abstractmethod
    def set_data_set(self):
        return NotImplemented

    @abc.abstractmethod
    def set_vocab_dict(self):
        return NotImplemented

    @abc.abstractmethod
    def set_model(self):
        return NotImplemented

class TrainerDirector():
    def __init__(self, trainer):
        self.trainer = trainer

    def create_trainer(self, data_set, vocab_dict, score_helper, model):
        self.trainer.set_data_set(data_set)
        self.trainer.set_vocab_dict(vocab_dict)
        self.trainer.set_score_helper(score_helper)
        self.trainer.set_model(model)


class Trainer(ITrainer, IHyperOptimizer):
    def __init__(self, config):
        self.config = config
        self.data_set = None
        self.vocab_dict = None
        self.score_helper = None
        self.model = None

    # @property
    # def test(self):
    #     return self._test
    # @test.setter
    # def test(self, val):
    #     self._test = val
    def set_score_helper(self, score_helper):
        self.score_helper = score_helper

    def set_data_set(self, data_set):
        self.data_set = data_set

    def set_vocab_dict(self, vocab_dict):
        self.vocab_dict = vocab_dict

    def set_model(self, model):
        self.model = model

    # def pad_sentence_batch(self, sentence_batch, vocab_to_int):
    #     """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    #     """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    #     max_sentence = max([len(sentence) for sentence in sentence_batch])
    #     return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    #
    # def get_batches(self, summaries, texts, batch_size, vocab_to_int):
    #     """Batch summaries, texts, and the lengths of their sentences together"""
    #     for batch_i in range(0, len(texts) // batch_size):
    #         start_i = batch_i * batch_size
    #         summaries_batch = summaries[start_i:start_i + batch_size]
    #         texts_batch = texts[start_i:start_i + batch_size]
    #         pad_summaries_batch = np.array(self.pad_sentence_batch(summaries_batch, vocab_to_int))
    #         pad_texts_batch = np.array(self.pad_sentence_batch(texts_batch, vocab_to_int))
    #
    #         # Need the lengths for the _lengths parameters
    #         pad_summaries_lengths = []
    #         for summary in pad_summaries_batch:
    #             pad_summaries_lengths.append(len(summary))
    #
    #         pad_texts_lengths = []
    #         for text in pad_texts_batch:
    #             pad_texts_lengths.append(len(text))
    #
    #         yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

    def run(self):
        epochs = self.config.epochs
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                print('epoch', epoch)
                batch_iter = batch_helper.get_batches(summaries=self.data_set.train_y, texts=self.data_set.train_x,
                                         batch_size=256, vocab_to_int=self.vocab_dict.vocab_to_int)
                for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:
                    # print('pad_summaries_batch', pad_summaries_batch.shape)
                    # print('pad_texts_batch', pad_texts_batch.shape)
                    # print(pad_summaries_batch)
                    feed = {self.model.input_data: pad_texts_batch,
                            self.model.targets: pad_summaries_batch,
                            self.model.summary_length: pad_summaries_lengths,
                            self.model.text_length: pad_texts_lengths}
                    sess.run(self.model.train_op, feed_dict=feed)

                loss = sess.run(self.model.cost, feed_dict=feed)
                print('loss', loss)
                inference_logitis = sess.run(self.model.inference_logits, feed_dict=feed)
                print(inference_logitis[0])

            save_path = saver.save(sess, "my_net/save_net.ckpt")
            print("Save to path: ", save_path)



if __name__ == "__main__":
    tt = Trainer()



