

import tensorflow as tf
import pickle
from tensorflow.python.layers.core import Dense

class Seq2SeqModel:
    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def summary_length(self):
        return self._summary_length

    @property
    def text_length(self):
        return self._text_length

    @property
    def learning_rate(self):
        return self._learning_rate

    def __init__(self, vocab_to_int, batch_size):
        # self.learning_rate = 0.001
        self.num_vocab = len(vocab_to_int)
        self.embedding_size = 300
        # self.embedding_size = 200
        # self.num_encoder_symbols = len(vocab_to_int)
        # self.num_decoder_symbols = len(vocab_to_int)
        # self.num_layers = 2
        self.num_layers = 1
        self.hidden_size = 128
        # self.hidden_size = 16
        self.vocab_to_int = vocab_to_int
        self.batch_size = batch_size



    def placeholder_init(self):
        self._input_data = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self._targets = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self._keep_prob = tf.placeholder(dtype=tf.float32)
        self._summary_length = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.max_summary_length = tf.reduce_max(self._summary_length)
        self._text_length = tf.placeholder(dtype=tf.int32, shape=(None,))
        self._learning_rate = tf.placeholder(dtype=tf.float32)

    def process_encoding_input(self):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

        ending = tf.strided_slice(self._targets, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def make_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        return cell

    def enc_embedding_layer_init(self):
        enc_embeddings = tf.Variable(tf.random_uniform([self.num_vocab, self.embedding_size], 0, 1))
        enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, self._input_data)
        # enc_embed_input = tf.nn.dropout(enc_embed_input, self._keep_prob)
        return enc_embed_input
    def encoder_layer_init(self, enc_embed_input):
        cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
        enc_output, enc_state = tf.nn.dynamic_rnn(cell, enc_embed_input,
                                                  sequence_length=self.text_length, dtype=tf.float32)

        # enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell, cell, enc_embed_input,
        #                                                            self.text_length, dtype=tf.float32)
        return enc_state

    def dec_embedding_layer_init(self, dec_input):
        dec_embeddings = tf.Variable(tf.random_uniform([self.num_vocab, self.embedding_size], 0, 1))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        # dec_embed_input = tf.nn.dropout(dec_embed_input, self._keep_prob)
        return dec_embed_input, dec_embeddings

    # def decoder_layer_init(self, enc_embed_input):
    #     dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
    #     # enc_output, enc_state = tf.nn.dynamic_rnn(cell, enc_embed_input,
    #     #                                           sequence_length=self.text_length, dtype=tf.float32)
    #     return dec_cell
    def output_layer_init(self):
        output_layer = Dense(self.num_vocab, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        return output_layer

    def training_help_init(self, enc_state, output_layer, dec_embed_input, dec_cell):

        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=self._summary_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, enc_state, output_layer)

        # training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True)[0]
        # training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
        #                                                        output_time_major=False,
        #                                                        impute_finished=True,
        #                                                        maximum_iterations=self.max_summary_length)
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                                maximum_iterations=self.max_summary_length)[0]
        # training_decoder_output = tf.nn.dropout(training_decoder_output, self._keep_prob)
        return training_decoder_output

    def inference_helper_init(self, dec_embeddings, output_layer, enc_state, dec_cell):
        start_token = self.vocab_to_int["<GO>"]
        end_token = self.vocab_to_int["<EOS>"]
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size],
                               name='start_tokens')
        # initial_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=enc_state)
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_token)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, enc_state, output_layer)
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                     output_time_major=False,
                                                                     impute_finished=True,
                                                                     maximum_iterations=self.max_summary_length)[0]
        return inference_decoder_output

    def train_operation_init(self, training_decoder_output, inference_decoder_output):
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        self.inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')
        masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32,
                                 name='masks')

        # l2_loss = 0.05 * (tf.nn.l2_loss(dec_embeddings) + tf.nn.l2_loss(self.enc_embed_input))
        # regularization_cost = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.0001), tf.trainable_variables())

        # self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)

        # tv = tf.trainable_variables()
        # regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

        # self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets, logits=training_logits)
        self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)
        self.cost = tf.reduce_mean(self.cost)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        # self.train_op = optimizer.minimize(self.cost)
        # # Gradient Clipping
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)






    def build_model(self):
        self.placeholder_init()
        enc_embed_input = self.enc_embedding_layer_init()
        enc_state = self.encoder_layer_init(enc_embed_input)
        dec_input = self.process_encoding_input()
        dec_embed_input, dec_embeddings = self.dec_embedding_layer_init(dec_input)
        # dec_cell = self.dec_embedding_layer_init(dec_input)
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])

        output_layer = self.output_layer_init()
        training_decoder_output = self.training_help_init(enc_state=enc_state, output_layer=output_layer,
                                                  dec_embed_input=dec_embed_input, dec_cell=dec_cell)
        inference_decoder_output = self.inference_helper_init(dec_embeddings, output_layer, enc_state, dec_cell)
        self.train_operation_init(training_decoder_output, inference_decoder_output)






