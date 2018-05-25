import data_loader as dl
from trainer import Trainer, TrainerDirector
from predictor import Predictor
from model_v2 import Seq2SeqModel
from hyper_optimizer.hyper_optimizer import Random_search
from rouge_helper import ROUGEHelper

class Config():
    base_path = 'data/'
    train_texts_path = base_path + 'min_texts'
    train_summaries_path = base_path+ 'min_summaries'
    val_texts_path = ''
    val_summaries_path = ''
    test_texts_path = base_path + 'min_test_texts'
    test_summaries_path = base_path + 'min_test_summaries'
    vocab2int_path = base_path + 'vocab_to_int.pkl'
    int2vocab_path = base_path + 'int_to_vocab.pkl'


class TrainerConfig():
    save_path = 'my_net/save_net.ckpt'
    epochs = 60


def main():
    cf = Config()
    trainer_config = TrainerConfig()
    data_set = dl.load_data_set(cf)
    vocab_dict = dl.load_vocab_dict(cf)
    # model = None

    model = Seq2SeqModel(vocab_dict.vocab_to_int)
    model.build_model()
    score_helper = ROUGEHelper()

    trainer = Trainer(trainer_config)
    trainer_director = TrainerDirector(trainer)
    trainer_director.create_trainer(data_set=data_set, vocab_dict=vocab_dict, score_helper=None, model=model)

    # trainer = Random_search(component=trainer, hp_generator=None)
    # trainer.run()

    predictor = Predictor(data_set=data_set, model=model, score_helper=score_helper, vocab_dict=vocab_dict)
    # test_sentence = data_set.train_x[0]
    # targrt_sentence = data_set.train_y[0]
    # print(test_sentence)
    # print(targrt_sentence)
    # predictor.one_text_prediction(sentence=test_sentence, batch_size=256)


    # predictor_director = PredictorDirector(predictor)
    # predictor_director.create_predictor(model=model, data_set=data_set, score_helper=None)


    predictor.get_score(input_data=data_set.test_x, target=data_set.test_y, vocab_to_int=vocab_dict.vocab_to_int)











main()