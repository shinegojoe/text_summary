from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

class BLEUHelper():
    def __init__(self, vocab_dict):
        self.smooth = SmoothingFunction()
        self.vocab_dict = vocab_dict
        self.pad_int = vocab_dict.vocab_to_int['<PAD>']

    def get_score(self, predict, ref):
        score = sentence_bleu([ref], predict, smoothing_function=self.smooth.method3)
        return score

    def remove_pad(self, sentence):
        index = 0
        for word in sentence:
            if word == self.pad_int:
                break
            else:
                index +=1
        return sentence[:index]


    def batch_score(self, batch_predict, batch_y):
        batch_score = []
        for i in range(len(batch_predict)):
            predict = self.remove_pad(batch_predict[i])
            label = self.remove_pad(batch_y[i])
            # print('predict', predict)
            # print('label', label)
            bleu_score = self.get_score(predict, label)
            batch_score.append(bleu_score)

        return np.mean(batch_score)


