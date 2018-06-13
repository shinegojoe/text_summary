import data_loader as dl
from data_preprocess_v2 import DataSet
from trainer import Trainer, TrainerDirector
from predictor import Predictor
# from model_v2 import Seq2SeqModel
# from model import Seq2SeqModel
# from model_v3 import Seq2SeqModel
from models.model_v3 import Seq2SeqModel
from hyper_optimizer.hyper_optimizer import Random_search
from nltk.translate.bleu_score import sentence_bleu
from data_model import DataSet
import pickle
import data_factory

from helpers.log_helper import LogHelper
from helpers.rouge_helper import ROUGEHelper
from helpers.bleu_helper import BLEUHelper
from helpers.batch_helper import BatchHelper


class Config():
    base_path = 'data/'
    # base_path = 'movie_data/'
    # base_path = 'THUC_texts_summaries_data/'
    # train_texts_path = base_path + 'min_texts'
    # train_summaries_path = base_path+ 'min_summaries'
    # val_texts_path = ''
    # val_summaries_path = ''
    # test_texts_path = base_path + 'min_test_texts'
    # test_summaries_path = base_path + 'min_test_summaries'

    vocab2int_path = base_path + 'vocab_to_int.pkl'
    int2vocab_path = base_path + 'int_to_vocab.pkl'
    data_set_path = base_path + 'long_length_data_set'

    # vocab2int_path = base_path + 'vocab_to_int.pkl'
    # int2vocab_path = base_path + 'int_to_vocab.pkl'
    # data_set_path = base_path + 'data_set.pkl'


class Hyperparameters():
    learning_rate = 0.005
    l2_lambda = 0.001
    keep_prob = 0.2


class TrainerConfig():
    # save_path_best_loss = 'my_net/thuc_model.ckpt'
    # save_path_final = 'my_net/thuc_model_final.ckpt'
    # save_path_best_loss = 'my_net/thuc_model_attention.ckpt'
    # save_path_final = 'my_net/thuc_model_final_attention.ckpt'
    # save_path_best_loss = 'movie_dialog_net/model_best_loss.ckpt'
    # save_path_final = 'movie_dialog_net/model_final.ckpt'

    save_path_best_loss = 'amazon_reviews_net/model_best_loss_v2.ckpt'
    save_path_final = 'amazon_reviews_net/model_final_v2.ckpt'

    predict_model_path = save_path_final
    # save_path = 'my_net/save_net_without_attention.ckpt'
    # finan_save_path = 'my_net/save_net_without_attention_final.ckpt'
    # save_path = 'my_net/test.ckpt'
    is_restore = False
    last_val_loss = 1.95

    restore_path = save_path_final
    save_path = save_path_final
    # log_file_name = 'movie_dialog'
    log_file_name = 'amazon_reviews'
    batch_size = 128
    epochs = 2

    # learning_rate_decay = 0.9
    learning_rate = 0.005
    # min_learning_rate = 0.0005
    # mode = 'train'
    mode = 'predict'
    # mode = 'none'



def main():
    cf = Config()
    log_helper = LogHelper()
    batch_helper = BatchHelper()
    trainer_config = TrainerConfig()
    data_set = dl.load_data_set(cf)
    vocab_dict = dl.load_vocab_dict(cf)
    # model = None
    print(len(data_set.train_x))
    # data_set.train_x = data_set.train_x[:5000]
    # data_set.train_y = data_set.train_y[:5000]
    # data_set.val_x = data_set.val_x[:500]
    # data_set.val_x = data_set.val_y[:500]
    # data_set = data_factory.random_select(10000, data_set)
    # print(len(data_set.train_x))
    # for x in data_set.train_x:
    #     print(len(x))
    # with open('small_data_set.pkl', 'wb') as f:
    #     pickle.dump(data_set, f)

    with open('small_data_set.pkl', 'rb') as f:
        data_set = pickle.load(f)




    model = Seq2SeqModel(vocab_dict.vocab_to_int, trainer_config.batch_size)
    model.build_model()
    # score_helper = ROUGEHelper()
    score_helper = BLEUHelper(vocab_dict)

    if trainer_config.mode == 'train':
        trainer = Trainer(trainer_config)
        trainer_director = TrainerDirector(trainer)
        trainer_director.create_trainer(data_set=data_set, vocab_dict=vocab_dict, score_helper=score_helper, model=model,
                                        log_helper=log_helper, batch_helper=batch_helper)

        # trainer = Random_search(component=trainer, hp_generator=None)
        trainer.run()
    elif trainer_config.mode == 'predict':
        predictor = Predictor(data_set=data_set, model=model, score_helper=score_helper, vocab_dict=vocab_dict,
                              path=trainer_config.predict_model_path, batch_helper = batch_helper)

        # data_x, data_y = data_factory.split_by_length(data_set.train_x, data_set.train_y)

        # for x in data_x[6]:
        #     print(len(x))
        #
        # print(len(data_x[6]))
        predictor.get_score(input_data=data_set.train_x, target=data_set.train_y, vocab_to_int=vocab_dict.vocab_to_int,
                            batch_size=trainer_config.batch_size)
        # predictor.get_score(input_data=data_x[0], target=data_y[0], vocab_to_int=vocab_dict.vocab_to_int,
        #                     batch_size=trainer_config.batch_size)

        # predictor.get_score(input_data=data_x[5], target=data_y[5], vocab_to_int=vocab_dict.vocab_to_int,
        #                     batch_size=32)

        # for i in range(10):
        #     predictor.one_text_prediction(data_set.train_x[i+200], TrainerConfig.batch_size, data_set.train_y[i+200])
        #     print()

main()