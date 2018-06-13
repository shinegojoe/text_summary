import abc
import tensorflow as tf
import numpy as np
from hyper_optimizer.hyper_optimizer import IHyperOptimizer



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
    def set_log_helper(self):
        return NotImplemented

    @abc.abstractmethod
    def set_vocab_dict(self):
        return NotImplemented

    @abc.abstractmethod
    def set_model(self):
        return NotImplemented

    @abc.abstractmethod
    def set_batch_helper(self):
        return NotImplemented



class TrainerDirector():
    def __init__(self, trainer):
        self.trainer = trainer

    def create_trainer(self, data_set, vocab_dict, score_helper, model, log_helper, batch_helper):
        self.trainer.set_data_set(data_set)
        self.trainer.set_vocab_dict(vocab_dict)
        self.trainer.set_score_helper(score_helper)
        self.trainer.set_model(model)
        self.trainer.set_log_helper(log_helper)
        self.trainer.set_batch_helper(batch_helper)


class Trainer(ITrainer, IHyperOptimizer):
    def __init__(self, config):
        self.config = config
        self.data_set = None
        self.vocab_dict = None
        self.score_helper = None
        self.log_helper = None
        self.batch_helper = None
        self.model = None

    # @property
    # def test(self):
    #     return self._test
    # @test.setter
    # def test(self, val):
    #     self._test = val
    def set_score_helper(self, score_helper):
        self.score_helper = score_helper
    def set_batch_helper(self, batch_helper):
        self.batch_helper = batch_helper

    def set_data_set(self, data_set):
        self.data_set = data_set

    def set_vocab_dict(self, vocab_dict):
        self.vocab_dict = vocab_dict

    def set_log_helper(self, log_helper):
        self.log_helper = log_helper

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

    def run_train_operation(self, sess, pad_texts_batch, pad_summaries_batch, pad_texts_lengths, pad_summaries_lengths):
        feed = {self.model.input_data: pad_texts_batch,
                self.model.targets: pad_summaries_batch,
                self.model.summary_length: pad_summaries_lengths,
                self.model.text_length: pad_texts_lengths,
                self.model.keep_prob: 0.3,
                self.model.learning_rate: self.config.learning_rate}
        sess.run(self.model.train_op, feed_dict=feed)

    def get_feed(self, pad_texts_batch, pad_summaries_batch, pad_texts_lengths, pad_summaries_lengths):

        feed = {self.model.input_data: pad_texts_batch,
                self.model.targets: pad_summaries_batch,
                self.model.text_length: pad_texts_lengths,
                self.model.summary_length: pad_summaries_lengths,
                self.model.keep_prob: 1.0,
                self.model.learning_rate: self.config.learning_rate}
        return feed

    def get_score(self, sess, x, y):
        scores = []
        batch_iter = self.batch_helper.get_batches(summaries=y, texts=x,
                                              batch_size=self.config.batch_size,
                                              vocab_to_int=self.vocab_dict.vocab_to_int)
        for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:

            feed = self.get_feed(pad_texts_batch, pad_summaries_batch, pad_texts_lengths, pad_summaries_lengths)

            inference_logitis = sess.run(self.model.inference_logits, feed_dict=feed)
            batch_score = self.score_helper.batch_score(inference_logitis, pad_summaries_batch)
            scores.append(batch_score)
        return np.mean(scores)





    def calculate_loss(self, sess, input, target):
        loss = []
        # val_state = sess.run(cell.zero_state(batch_size, tf.float32))
        batch_iter = self.batch_helper.get_batches(summaries=target, texts=input,
                                              batch_size=self.config.batch_size,
                                              vocab_to_int=self.vocab_dict.vocab_to_int)
        for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:

            feed = self.get_feed(pad_texts_batch, pad_summaries_batch, pad_texts_lengths, pad_summaries_lengths)
            batch_loss = sess.run(self.model.cost, feed_dict=feed)
            loss.append(batch_loss)
        # print("Val acc: {:.3f}".format(np.mean(acc)))
        return np.mean(loss)

    def save_model_check(self, saver, sess, val_loss, last_val_loss):
        if val_loss < last_val_loss:
            save_path = saver.save(sess, self.config.save_path_best_loss)
            print("Save to path: ", save_path)
            return val_loss
        else:
            return last_val_loss

    def run(self):
        epochs = self.config.epochs
        saver = tf.train.Saver()
        last_val_loss = self.config.last_val_loss
        with tf.Session() as sess:
            if self.config.is_restore:
                saver.restore(sess, self.config.restore_path)
            else:
                sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "my_net/save_net.ckpt")
            train_loss_log = []
            val_loss_log = []
            train_score_log = []
            val_score_log = []
            for epoch in range(epochs):
                print('epoch', epoch)
                # batch_iter = batch_helper.get_batches(summaries=self.data_set.train_y, texts=self.data_set.train_x,
                #                          batch_size=self.config.batch_size, vocab_to_int=self.vocab_dict.vocab_to_int)
                batch_iter = self.batch_helper.get_batches(summaries=self.data_set.train_y, texts=self.data_set.train_x,
                                                      batch_size=self.config.batch_size,
                                                      vocab_to_int=self.vocab_dict.vocab_to_int)
                for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:
                    # print('pad_summaries_batch', pad_summaries_batch.shape)
                    # print('pad_texts_batch', pad_texts_batch.shape)
                    # print(pad_summaries_batch)
                    # feed = {self.model.input_data: pad_texts_batch,
                    #         self.model.targets: pad_summaries_batch,
                    #         self.model.summary_length: pad_summaries_lengths,
                    #         self.model.text_length: pad_texts_lengths,
                    #         self.model.keep_prob:0.3,
                    #         self.model.learning_rate:self.config.learning_rate}
                    # sess.run(self.model.train_op, feed_dict=feed)
                    self.run_train_operation(sess, pad_texts_batch, pad_summaries_batch, pad_texts_lengths, pad_summaries_lengths)
                    # loss = sess.run(self.model.cost, feed_dict=feed)
                    # print(loss)
                # inference_logitis = sess.run(self.model.inference_logits, feed_dict=feed)
                # for i in range(10):
                #     print(inference_logitis[i])
                #
                # score = self.score_helper.batch_score(inference_logitis, pad_summaries_batch)
                # print(score)
                train_score = self.get_score(sess, self.data_set.train_x, self.data_set.train_y)
                val_score = self.get_score(sess, self.data_set.val_x, self.data_set.val_y)
                print('train_score', train_score)
                print('val_score', val_score)

                # loss = sess.run(self.model.cost, feed_dict=feed)
                # test_amount = self.config.batch_size * 20
                train_loss = self.calculate_loss(sess=sess, input=self.data_set.train_x, target=self.data_set.train_y)
                print('train_loss', train_loss)
                val_loss = self.calculate_loss(sess=sess, input=self.data_set.val_x, target=self.data_set.val_y)
                print("val_loss", val_loss)

                train_score_log.append(train_score)
                val_score_log.append(val_score)
                train_loss_log.append(train_loss)
                val_loss_log.append(val_loss)

                last_val_loss = self.save_model_check(saver=saver, sess=sess, val_loss=val_loss, last_val_loss=last_val_loss)
                # print('last loss ', last_val_loss)
                # if val_loss < last_val_loss:
                #     save_path = saver.save(sess, self.config.save_path_best_loss)
                #     print("Save to path: ", save_path)
                #     last_val_loss = val_loss
                # inference_logitis = sess.run(self.model.inference_logits, feed_dict=feed)

                # print(inference_logitis[0])

            # save_path = saver.save(sess, "my_net/save_net_no_attention.ckpt")
            self.log_helper.save_plt(x1=train_loss_log, x2=val_loss_log, file_name=self.config.log_file_name, x1_label="train_loss",
                                     x2_label="val_loss",y_label='loss')

            self.log_helper.save_plt(x1=train_score_log, x2=val_score_log, file_name=self.config.log_file_name,
                                     x1_label="train_score",
                                     x2_label="val_score", y_label='bleu_score')
            save_path = saver.save(sess, self.config.save_path)
            # save_path = saver.save(sess, 'my_net/test.ckpt')
            print("Save to path: ", save_path)
            # print("Save to path: ", save_path)



if __name__ == "__main__":
    tt = Trainer()



