import abc
import tensorflow as tf
import batch_helper
import numpy as np

class IPredictor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_score(self):
        return NotImplemented

    @abc.abstractmethod
    def one_text_prediction(self):
        return NotImplemented

    # @abc.abstractmethod
    # def set_model(self):
    #     return NotImplemented
    #
    # @abc.abstractmethod
    # def set_data_set(self):
    #     return NotImplemented
    #
    # @abc.abstractmethod
    # def set_score_helper(self):
    #     return NotImplemented

# class PredictorDirector():
#     def __init__(self, predictor):
#         self.predictor = predictor
#
#     def create_predictor(self, data_set, model, score_helper):
#         self.predictor.set_model(model)
#         self.predictor.set_data_set(data_set)
#         self.predictor.set_score_helper(score_helper)


class Predictor(IPredictor):
    def __init__(self, data_set, model, score_helper, vocab_dict):
        self.data_set = data_set
        self.model = model
        self.score_helper = score_helper
        self.vocab_dict = vocab_dict
        self.sess = self.predic_init()


    # def set_model(self, model):
    #     self.model = model
    #
    # def set_data_set(self, data_set):
    #     self.data_set = data_set
    #
    # def set_score_helper(self, score_helper):
    #     self.score_helper = score_helper
    def predic_init(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, "my_net/save_net.ckpt")
        return sess

    def id_to_vocab(self, data, int_to_vovab):
        sentence = []
        for id in data:
            word = int_to_vovab[id]
            sentence.append(word)

        return sentence

    def get_score(self, input_data, target, vocab_to_int):
        # summaries, texts, vocab_to_int = load_data()
        # int_to_vocab = load_pkl('data/' + 'int_to_vocab.pkl')


        scores = []
        batch_iter = batch_helper.get_batches(texts=input_data, summaries=target, batch_size=256, vocab_to_int=vocab_to_int)
        for pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths in batch_iter:
            # feed = {model.input_data: pad_texts_batch,
            #         model.summary_length: pad_summaries_batch,
            #         model.text_length: pad_texts_lengths}
            feed = {self.model.input_data: pad_texts_batch,
                    self.model.targets: pad_summaries_batch,
                    self.model.summary_length: pad_summaries_lengths,
                    self.model.text_length: pad_texts_lengths}
            inference_logitis = self.sess.run(self.model.inference_logits, feed_dict=feed)
            # print('inference_logitis', inference_logitis)

            batch_score = self.score_helper.get_batch_score(inference_logitis, pad_summaries_batch)
            print(batch_score)
            scores.append(batch_score)

        print('score = ', np.mean(scores))

    def one_text_prediction(self, sentence, batch_size):
        sentence_length = len(sentence)
        feed = {self.model.input_data: [sentence] * batch_size,
                self.model.summary_length: [sentence_length] * batch_size,
                self.model.text_length: [sentence_length] * batch_size
                }
        inference_logitis = self.sess.run(self.model.inference_logits, feed_dict=feed)[0]
        print('inference_logitis', inference_logitis)
        sentence = self.id_to_vocab(data=inference_logitis, int_to_vovab=self.vocab_dict.int_to_vocab)
        print(sentence)



if __name__ == "__main__":
    p = Predictor()
