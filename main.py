import data_loader as dl
from data_preprocess_v2 import DataSet
from trainer import Trainer, TrainerDirector
from predictor import Predictor
# from model_v2 import Seq2SeqModel
# from model import Seq2SeqModel
from model_v4 import Seq2SeqModel
from hyper_optimizer.hyper_optimizer import Random_search
from rouge_helper import ROUGEHelper
from nltk.translate.bleu_score import sentence_bleu
from log_helper import LogHelper
from data_model import DataSet


class Config():
    # base_path = 'data/'
    # base_path = 'movie_data/'
    base_path = 'THUC_texts_summaries_data/'
    # train_texts_path = base_path + 'min_texts'
    # train_summaries_path = base_path+ 'min_summaries'
    # val_texts_path = ''
    # val_summaries_path = ''
    # test_texts_path = base_path + 'min_test_texts'
    # test_summaries_path = base_path + 'min_test_summaries'

    # vocab2int_path = base_path + 'vocab_to_int.pkl'
    # int2vocab_path = base_path + 'int_to_vocab.pkl'
    # data_set_path = base_path + 'long_length_data_set'

    vocab2int_path = base_path + 'vocab_to_int.pkl'
    int2vocab_path = base_path + 'int_to_vocab.pkl'
    data_set_path = base_path + 'data_set.pkl'




class TrainerConfig():
    # save_path_best_loss = 'my_net/thuc_model.ckpt'
    # save_path_final = 'my_net/thuc_model_final.ckpt'
    save_path_best_loss = 'my_net/thuc_model_attention.ckpt'
    save_path_final = 'my_net/thuc_model_final_attention.ckpt'
    predict_model_path = save_path_final
    # save_path = 'my_net/save_net_without_attention.ckpt'
    # finan_save_path = 'my_net/save_net_without_attention_final.ckpt'
    # save_path = 'my_net/test.ckpt'
    is_restore = False
    restore_path = save_path_final
    save_path = save_path_final
    batch_size = 64
    epochs = 40
    learning_rate_decay = 0.9
    learning_rate = 0.003
    min_learning_rate = 0.0005



def split_by_length(texts, summaries):
    g1_x = []
    g2_x = []
    g3_x = []
    g4_x = []
    g5_x = []
    g6_x = []
    g7_x = []

    g1_y = []
    g2_y = []
    g3_y = []
    g4_y = []
    g5_y = []
    g6_y = []
    g7_y = []
    data_x = [g1_x, g2_x, g3_x, g4_x, g5_x, g6_x, g7_x]
    data_y = [g1_y, g2_y, g3_y, g4_y, g5_y, g6_y, g7_y]
    for i in range(len(texts)):
        x = texts[i]
        y = summaries[i]
        x_len = len(x)
        if x_len >=10 and x_len < 30:
            g1_x.append(x)
            g1_y.append(y)
        elif x_len >= 30 and x_len < 50:
            g2_x.append(x)
            g2_y.append(y)
        elif x_len >=50 and x_len < 70:
            g3_x.append(x)
            g3_y.append(y)
        elif x_len >=70 and x_len < 90:
            g4_x.append(x)
            g4_y.append(y)
        elif x_len >=90 and x_len < 110:
            g5_x.append(x)
            g5_y.append(y)
        elif x_len >=110 and x_len < 130:
            g6_x.append(x)
            g6_y.append(y)
        elif x_len >=130 and x_len < 150:
            g7_x.append(x)
            g7_y.append(y)

    return data_x, data_y


import pickle
def main():
    cf = Config()
    log_helper = LogHelper()
    trainer_config = TrainerConfig()
    data_set = dl.load_data_set(cf)
    vocab_dict = dl.load_vocab_dict(cf)
    # model = None



    model = Seq2SeqModel(vocab_dict.vocab_to_int, trainer_config.batch_size)
    model.build_model()
    score_helper = ROUGEHelper()
    #
    trainer = Trainer(trainer_config)
    trainer_director = TrainerDirector(trainer)
    trainer_director.create_trainer(data_set=data_set, vocab_dict=vocab_dict, score_helper=None, model=model, log_helper=log_helper)

    # trainer = Random_search(component=trainer, hp_generator=None)
    trainer.run()



    # test_sentence = data_set.train_x[0]
    # targrt_sentence = data_set.train_y[0]
    # print(test_sentence)
    # print(targrt_sentence)
    # predictor.one_text_prediction(sentence=test_sentence, batch_size=256)


    # predictor_director = PredictorDirector(predictor)
    # predictor_director.create_predictor(model=model, data_set=data_set, score_helper=None)


    # predictor.get_score(input_data=data_set.test_x, target=data_set.test_y, vocab_to_int=vocab_dict.vocab_to_int,
    #                     batch_size=trainer_config.batch_size)

    # data_x, data_y = split_by_length(data_set.train_x, data_set.train_y)
    # # #
    # for x in data_set.train_x:
    #     print(len(x))
    # for x in data_x[0]:
    #     print(len(x))
    #
    # predictor = Predictor(data_set=data_set, model=model, score_helper=score_helper, vocab_dict=vocab_dict,
    #                       path=trainer_config.predict_model_path)

    # predictor.get_score(input_data=data_x[0], target=data_y[0], vocab_to_int=vocab_dict.vocab_to_int,
    #                  batch_size=trainer_config.batch_size)
    # predictor.get_score(input_data=data_set.train_x, target=data_set.train_y, vocab_to_int=vocab_dict.vocab_to_int,
    #                     batch_size=trainer_config.batch_size)
    # d1 = data_x[2]
    # dy = data_y[2]
    #
    # for i in range(10):
    #     predictor.one_text_prediction(d1[i+200], TrainerConfig.batch_size, dy[i+200])
    #     print()












main()