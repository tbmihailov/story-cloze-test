import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper

from utils.tf.tf_helpers import tf_helpers
import logging


def tf_matmul_broadcast(tensor1_ijk, tensor2_kl):
    i = tf.shape(tensor1_ijk)[0]
    j = tf.shape(tensor1_ijk)[1]
    k = tf.shape(tensor1_ijk)[2]

    l = tf.shape(tensor2_kl)[1]

    a_flat = tf.reshape(tensor1_ijk, [-1, k])
    res_flat = tf.matmul(a_flat, tensor2_kl)

    res_desired = tf.reshape(res_flat, tf.pack([i, j, l]))

    return res_desired


def tf_gather_on_axis1(tensor, indices, axis=1):
    tensor_flat = tf.reshape(tensor, tf.concat(0,
                                               [tf.expand_dims(tf.reduce_prod(tf.shape(tensor)[:axis + 1]), 0),
                                                tf.shape(tensor)[axis + 1:]]))
    index_offset = tf.range(0, tf.reduce_prod(tf.shape(tensor)[:axis + 1]),
                            tf.shape(tensor)[axis])  # shape=[50,1], tensor [[55],[110],165....]
    flattened_indices = indices + index_offset
    res = tf.gather(tensor_flat, flattened_indices)
    return res


def gaussian_noise_layer(input_layer, std, name=None):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32, name=name)
    return input_layer + noise

class StoryClozeLSTMAttentionReader(object):
    def __init__(self,
                 n_classes,
                 embeddings,
                 embeddings_size,
                 embeddings_number,
                 hidden_size,
                 learn_rate=0.001,
                 loss_l2_beta=0.01,
                 layer_out_noise=0.00,
                 embeddings_trainable=True,
                 bilstm_layers=1,
                 dropout_prob_keep=0.00,
                 out_embeddings_mean=False,
                 out_attention_repr=False,
                 lstm_out_repr=True,
                 cond_answ_on_story=True,
                 story_reads_num=1,
                 cond_answ2_on_answ1=False,
                 bidirectional=True,
                 use_mlp=False
                 ):
        init_xavier = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        init_random_uniform = tf.random_uniform_initializer(-0.05, 0.05, seed=42)
        init_constant_zero = tf.constant_initializer(value=0)

        self.l2_beta = loss_l2_beta
        self.learn_rate = learn_rate

        self.embeddings_size = embeddings_size if embeddings is None else len(embeddings[0])
        self.embeddings_number = embeddings_number if embeddings is None else len(embeddings)
        self.embeddings_trainable = embeddings_trainable

        self.n_classes = n_classes
        self.hidden_size = hidden_size

        with tf.name_scope("input_params"):
            # input params
            self.input_story_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_story_tokens")
            self.input_answer1_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_answer1_tokens")
            self.input_answer2_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_answer2_tokens")

            # sequence lengths
            self.input_story_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                             name="input_story_tokens_seq_len")  # length of every sequence in the batch
            self.input_answer1_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                               name="input_answer1_tokens_seq_len")  # length of every sequence in the batch
            self.input_answer2_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                               name="input_answer2_tokens_seq_len")  # length of every sequence in the batch

            self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name="input_y")  # this is onehot!

        # mask
        self.input_story_tokens_mask = tf.expand_dims(
            tf.cast(tf.sequence_mask(self.input_story_tokens_seq_len), dtype=tf.float32,
                    name="input_story_tokens_mask"), -1)
        self.input_answer1_tokens_mask = tf.expand_dims(
            tf.cast(tf.sequence_mask(self.input_answer1_tokens_seq_len), dtype=tf.float32,
                    name="input_answer1_tokens_mask"), -1)
        self.input_answer2_tokens_mask = tf.expand_dims(
            tf.cast(tf.sequence_mask(self.input_answer2_tokens_seq_len), dtype=tf.float32,
                    name="input_answer2_tokens_mask"), -1)

        self.input_y_onehot = tf.one_hot(self.input_y, n_classes)

        # model settings
        self.weights = {}
        self.biases = {}

        with tf.name_scope("embeddings"):
            # embeddings, loaded, tuned
            if not embeddings is None:
                self.embeddings_tuned = tf.get_variable("embeddings",
                                                        shape=embeddings.shape,
                                                        initializer=tf.constant_initializer(embeddings),
                                                        trainable=self.embeddings_trainable,
                                                        dtype=tf.float32)
            else:
                self.embeddings_tuned = tf.get_variable("embeddings",
                                        initializer=tf.truncated_normal([embeddings.shape], stddev=0.1, dtype=tf.float32),
                                        trainable=self.embeddings_trainable,
                                        dtype=tf.float32)

            self.embedded_chars_story = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_story_tokens)
            self.embedded_chars_answer1 = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_answer1_tokens)
            self.embedded_chars_answer2 = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_answer2_tokens)

        self.mean_story_embeddings = tf.reduce_sum(self.embedded_chars_story * self.input_story_tokens_mask,
                                                   reduction_indices=-2) / tf.expand_dims(tf.cast(self.input_story_tokens_seq_len, dtype=tf.float32), -1)
        self.mean_answer1_embeddings = (self.mean_story_embeddings - tf.reduce_sum(self.embedded_chars_answer1 * self.input_answer1_tokens_mask,
                                                     reduction_indices=-2) / tf.expand_dims(tf.cast(self.input_answer1_tokens_seq_len, dtype=tf.float32), -1))
        self.mean_answer2_embeddings = (self.mean_story_embeddings - tf.reduce_sum(self.embedded_chars_answer2 * self.input_answer2_tokens_mask,
                                                     reduction_indices=-2) / tf.expand_dims(tf.cast(self.input_answer2_tokens_seq_len, dtype=tf.float32), -1))

        lstm_layers_cnt = bilstm_layers
        shared_fw_bw = False
        with tf.variable_scope('lstm-encode') as lstm_encode_scope:
            def encode_sequence(x_embedd, input_seq_len, lstm_cell_fw, lstm_cell_bw, initial_state=None,
                                bidirectional=False):
                output_layer_story = None
                output_states_story = None
                if bidirectional:
                    outputs_story, states_story = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=lstm_cell_fw,
                        cell_bw=lstm_cell_bw,
                        dtype=tf.float32,
                        sequence_length=input_seq_len,
                        inputs=x_embedd,
                        initial_state_fw=initial_state[0] if initial_state is not None else None,
                        initial_state_bw=initial_state[1] if initial_state is not None else None)

                    output_layer_story = tf.concat(2, outputs_story, name="output_layer")  # concat fw and bw layers
                    output_states_story = states_story
                else:
                    outputs_story, states_story = tf.nn.dynamic_rnn(
                        cell=lstm_cell_fw,
                        dtype=tf.float32,
                        sequence_length=input_seq_len,
                        inputs=x_embedd,
                        initial_state=initial_state)

                    output_layer_story = outputs_story  # concat fw and bw layers
                    output_states_story = states_story

                return output_layer_story, output_states_story

            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                   use_peepholes=False,
                                                   cell_clip=None,
                                                   num_proj=None,
                                                   proj_clip=None,
                                                   num_unit_shards=1, num_proj_shards=1,
                                                   forget_bias=1.0,
                                                   activation=tf.tanh,
                                                   state_is_tuple=True,
                                                   initializer=init_random_uniform)


            lstm_cell_fw = DropoutWrapper(lstm_cell_fw, output_keep_prob=dropout_prob_keep)

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * lstm_layers_cnt, state_is_tuple=True)

            # if not shared_fw_bw:
            #     lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
            #                                            use_peepholes=False,
            #                                            cell_clip=None,
            #                                            num_proj=None,
            #                                            proj_clip=None,
            #                                            num_unit_shards=1, num_proj_shards=1,
            #                                            forget_bias=1.0,
            #                                            activation=tf.tanh,
            #                                            state_is_tuple=True,
            #                                            initializer=init_random_uniform)
            # else:
            #     lstm_cell_bw = lstm_cell_fw
            #
            #     multicell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * lstm_layers_cnt, state_is_tuple=True)

            multicell_bw = None

            output_layer = None
            output_layer_size = 0

            lstm_out_layer_size = 2 * hidden_size if bidirectional else hidden_size

            encoded_story_states = None
            # encode representations
            for read_id in range(story_reads_num):
                encoded_story, encoded_story_states = encode_sequence(self.embedded_chars_story,
                                                                  self.input_story_tokens_seq_len,
                                                                  lstm_cell_fw=lstm_cell_fw, lstm_cell_bw=lstm_cell_fw,
                                                                  initial_state=encoded_story_states, bidirectional=bidirectional)
                lstm_encode_scope.reuse_variables()

            encoded_story_last_output = tf_gather_on_axis1(encoded_story,
                                                           self.input_story_tokens_seq_len - tf.ones_like(
                                                               self.input_story_tokens_seq_len))
            logging.info("encoded_story_last_output shape: %s" % (encoded_story_last_output.get_shape()))

            lstm_encode_scope.reuse_variables()

            # Encode Answer 1
            encoded_answer1, encoded_answer1_states = encode_sequence(self.embedded_chars_answer1, self.input_answer1_tokens_seq_len,
                                                 lstm_cell_fw=lstm_cell_fw, lstm_cell_bw=lstm_cell_fw,
                                                 initial_state=encoded_story_states if cond_answ_on_story else None,
                                                 bidirectional=bidirectional)
            encoded_answer1_last_output = tf_gather_on_axis1(encoded_answer1,
                                                             self.input_answer1_tokens_seq_len - tf.ones_like(
                                                                 self.input_answer1_tokens_seq_len))

            # Encode Answer 2
            encoded_answer2, encoded_answer2_states = encode_sequence(self.embedded_chars_answer2, self.input_answer2_tokens_seq_len,
                                                                      lstm_cell_fw=lstm_cell_fw, lstm_cell_bw=lstm_cell_fw,
                                                                      initial_state=encoded_answer1_states if cond_answ2_on_answ1 else (encoded_story_states if cond_answ_on_story else None),
                                                                      bidirectional=bidirectional)
            encoded_answer2_last_output = tf_gather_on_axis1(encoded_answer2,
                                                             self.input_answer2_tokens_seq_len - tf.ones_like(
                                                                 self.input_answer2_tokens_seq_len))

        with tf.variable_scope('attention') as attention_scope:
            def tf_compute_attention_base(prem_seq, prem_seq_len, hypo_seq, hypo_seq_len, hidden_size):
                """
                Computes attention over words in prem_seq by hypo_seq. Implemented as in  Reasoning about Entailment with Neural Attention http://arxiv.org/abs/1509.06664
                :param prem_seq: Batch of premise words representations (The output layer of an LSTM)
                :param hypo_seq:
                :param prem_seq_len:
                :param hypo_seq_len:
                :return: r : The attention weighted representation with size - the input vector,
                attention:Attention weights:`Tensor` with the size [batch_size, length] with attention weights per word,
                h_star: Story and Answer attention weighted representation
                """

                prem_seq_masked = prem_seq * tf.expand_dims(
                    tf.cast(tf.sequence_mask(prem_seq_len), dtype=tf.float32), -1)  # [b, l, k]
                b = tf.shape(prem_seq)[0]
                l = tf.shape(prem_seq)[1]
                k = tf.shape(prem_seq)[2]

                hypo_hN = tf_gather_on_axis1(hypo_seq, hypo_seq_len - tf.ones_like(hypo_seq_len))  # [b, k]
                # with tf.variable_scope("attention-1") as scope_attention:
                Wy = tf.get_variable("att_Wy",
                                     shape=[hidden_size, hidden_size],
                                     initializer=init_random_uniform,
                                     dtype=tf.float32)
                Wh = tf.get_variable("att_Wh",
                                     shape=[hidden_size, hidden_size],
                                     initializer=init_random_uniform,
                                     dtype=tf.float32)
                L = l  # length of the premise (max length) [l]
                e = tf.ones([1, L])  # shape = [l, 1]

                hypo_hN_repeat = tf.expand_dims(hypo_hN, 1)
                pattern = tf.pack([1, L, 1])
                hypo_hN_repeat = tf.tile(hypo_hN_repeat, pattern)

                # M = tf.tanh(tf_matmul_broadcast(prem_seq_masked, Wy) + tf.transpose(tf_matmul_broadcast(tf.expand_dims(tf_matmul_broadcast(hypo_hN, Wh), -1),e),[0,2,1]), name="adsadsadsad")  # shape = [b, k, L]

                M = tf.tanh(tf_matmul_broadcast(prem_seq_masked, Wy) + tf_matmul_broadcast(hypo_hN_repeat, Wh),
                            name="M")  # shape = [b, k, L]
                #  [b,l,k], [k,k]->blk + bk *kk 0>bk1*1l _> T -> blk
                att_w = tf.get_variable("att_w", shape=[hidden_size, 1], initializer=init_random_uniform, dtype=tf.float32)  # [k]

                tmp3 = tf.matmul(tf.reshape(M, shape=[b * l, k]), att_w)
                # need 1 here so that later can do multiplication with h x L
                attention = tf.nn.softmax(tf.reshape(tmp3, shape=[b, 1, l], name="att"))  # b x 1 x l

                # attention = tf.nn.softmax(tf_matmul_broadcast(M, tf.transpose(att_w))) # [b,l,1]

                r = tf.batch_matmul(attention, prem_seq_masked)  # [b, l, k] matmul b 1 l -> b 1 k
                r = tf.reshape(r, [b, k])  # bk

                Wp = tf.get_variable("Wp", shape=[hidden_size, hidden_size],initializer=init_random_uniform, dtype=tf.float32)  # [k, k]
                bp = tf.get_variable("bp", shape=[hidden_size], initializer=init_random_uniform, dtype=tf.float32)

                Wx = tf.get_variable("Wx", shape=[hidden_size, hidden_size], initializer=init_random_uniform, dtype=tf.float32)  # [k, k]
                bx = tf.get_variable("bx", shape=[hidden_size], initializer=init_random_uniform, dtype=tf.float32)

                h_star = tf.tanh(tf.matmul(r, Wp) + bp + tf.matmul(hypo_hN, Wx) + bx)  # b k 1

                attention_reshaped = tf.reshape(attention, [b, l])
                return r, attention_reshaped, h_star

            self.story_answ1_att_r, self.story_answ1_att_attention, self.story_answ1_rel_repr = tf_compute_attention_base(
                encoded_story,
                self.input_story_tokens_seq_len,
                encoded_answer1,
                self.input_answer1_tokens_seq_len,
                lstm_out_layer_size)

            attention_scope.reuse_variables()
            self.story_answ2_att_r, self.story_answ2_att_attention, self.story_answ2_rel_repr = tf_compute_attention_base(
                encoded_story,
                self.input_story_tokens_seq_len,
                encoded_answer2,
                self.input_answer2_tokens_seq_len,
                lstm_out_layer_size)

        # default features
        # all_feats = tf.concat(1, [encoded_story_last_output,encoded_answer1_last_output, encoded_answer2_last_output])
        # output_layer_size = 3 * output_layer_size #+ 3 * self.embeddings_size

        # attention based representations features
        # all_feats = tf.concat(1, [self.story_answ1_rel_repr, self.story_answ2_rel_repr])
        output_layer_size = 0
        self.all_feats = None
        if lstm_out_repr:
            self.all_feats = tf.concat(1, [encoded_answer1_last_output, encoded_answer2_last_output])
            output_layer_size += 2 * lstm_out_layer_size

            # self.all_feats = tf.concat(1, [encoded_story_last_output, encoded_answer1_last_output,
            #                                encoded_answer2_last_output])
            # output_layer_size += 3 * lstm_out_layer_size

        if out_attention_repr:
            if self.all_feats is not None:
                self.all_feats = tf.concat(1, [self.all_feats, self.story_answ1_rel_repr, self.story_answ2_rel_repr])
            else:
                self.all_feats = tf.concat(1, [self.story_answ1_rel_repr, self.story_answ2_rel_repr])
            output_layer_size += 2 * lstm_out_layer_size

        if out_embeddings_mean:
            if self.all_feats is not None:
                self.all_feats = tf.concat(1, [self.all_feats,  self.mean_answer1_embeddings, self.mean_answer2_embeddings])
            else:
                self.all_feats = tf.concat(1, [self.mean_answer1_embeddings, self.mean_answer2_embeddings])
            output_layer_size += 2 * embeddings_size


        logging.info("all_feats:%s " % self.all_feats.get_shape())

        # all_feats = tf.nn.dropout(all_feats, keep_prob=dropout_prob_keep)
        if layer_out_noise > 0.01:
            self.all_feats = gaussian_noise_layer(self.all_feats, std=layer_out_noise)

        self.all_feats = tf.identity(self.all_feats, name="all_feats")

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        # bi_output_concat_flat = tf.reshape(output_layer, [-1, output_layer_size])
        # bi_output_concat_flat_clipped = tf.clip_by_value(bi_output_concat_flat, -1., 1.)  # use clipping if needed. Currently not

        # use_mlp = True #this is input parameter now
        if not use_mlp:
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            self.weights['out'] = tf.get_variable("out_w", shape=[output_layer_size, n_classes], initializer=init_xavier)
            self.biases['out'] = tf.get_variable("out_b", shape=[n_classes], initializer=init_constant_zero, dtype=tf.float32)

            self.logits = tf.matmul(self.all_feats, self.weights["out"]) + self.biases["out"]

            # self.logits_flat = tf.batch_matmul(bi_output_concat_flat, self.weights["out"]) + self.biases["out"]
            # logits_flat = tf.clip_by_value(logits_flat, -1., 1.)  # use clipping if needed. Currently not
            self.probs = tf.nn.softmax(self.logits)

            self.losses = tf_helpers.tf_nan_to_zeros_float32(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y_onehot))
        else:
            n_input = output_layer_size
            n_hidden_1 = 200  # 1st layer number of features
            n_hidden_2 = 200  # 2nd layer number of features

            with tf.variable_scope('multilayer_perceptron') as attention_scope:
                self.weights = {
                    'mlp_h1': tf.get_variable("mlp_h1", shape=[n_input, n_hidden_1],  initializer=init_xavier),
                    'mlp_h2': tf.get_variable("mlp_h2", shape=[n_hidden_1, n_hidden_2],  initializer=init_xavier),
                    'mlp_out': tf.get_variable("mlp_out", shape=[n_hidden_2, n_classes],  initializer=init_xavier),
                }

                self.biases = {
                    'mlp_b1': tf.get_variable("mlp_b1", shape=[n_hidden_1], initializer=init_constant_zero, dtype=tf.float32),
                    'mlp_b2': tf.get_variable("mlp_b2", shape=[n_hidden_2], initializer=init_constant_zero, dtype=tf.float32),
                    'mlp_b_out': tf.get_variable("mlp_b_out", shape=[n_classes], initializer=init_constant_zero, dtype=tf.float32),
                }

                def multilayer_perceptron(x, weights, biases):
                    # Hidden layer with RELU activation
                    layer_1 = tf.add(tf.matmul(x, weights['mlp_h1']), biases['mlp_b1'])
                    layer_1 = tf.nn.relu(layer_1)

                    # Hidden layer with RELU activation
                    layer_2 = tf.add(tf.matmul(layer_1, weights['mlp_h2']), biases['mlp_b2'])
                    layer_2 = tf.nn.relu(layer_2)

                    # Output layer with linear activation
                    out_layer = tf.matmul(layer_2, weights['mlp_out']) + biases['mlp_b_out']
                    return out_layer

            self.probs = multilayer_perceptron(self.all_feats, self.weights, self.biases)
            # sparse_softmax_cross_entropy_with_logits - calculates on non-onehot y!
            # this should also not calculate the cross entropy for 0 labels (the padded labels)
            self.losses = tf_helpers.tf_nan_to_zeros_float32(
                tf.nn.softmax_cross_entropy_with_logits(self.probs, self.input_y_onehot))

        self.predictions = tf.argmax(self.probs, 1, name="predictions")

        # print "outputs[-1].shape:%s" % outputs[-1]  #
        # logging.info("weights[\"out\"]].shape:%s" % self.weights["out"])  # .shape
        # logging.info("biases[\"out\"].shape:%s" % self.biases["out"])  # .shape

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # calculate L2 loss
        vars = tf.trainable_variables()
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_beta

        self.mean_loss = tf.add(tf.reduce_mean(self.losses), loss_l2, name="mean_loss")

        # Compute and apply gradients
        if self.learn_rate > 0:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        else:
            optimizer = tf.train.AdamOptimizer()

        gvs = optimizer.compute_gradients(self.mean_loss)

        logging.info("gradients:")
        for grad, var in gvs:
            logging.info("%s - %s" % (grad, var))
        capped_gvs = [(tf.clip_by_value(tf_helpers.tf_nan_to_zeros_float32(grad), -6.,
                                        6.) if grad is not None else grad, var) for grad, var in
                      gvs]  # cap to prevent NaNs

        self.apply_grads_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            # Calculate the accuracy
            # Used during training
            # Mask the losses - padded values are zeros

            # dnn_model.losses
            # dnn_model.predictions
            correct_pred = tf.equal(tf.cast(self.predictions, dtype=tf.int32), self.input_y)

            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        logging.info("The model might be okay :D")
        pass

    def train_step(self, sess, batch_x_zipped, batch_y):
        batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen, batch_endings2_padded, batch_endings2_seqlen = zip(
            *batch_x_zipped)

        feed_dict = {
            self.input_story_tokens: batch_stories_padded,
            self.input_story_tokens_seq_len: batch_stories_seqlen,
            self.input_answer1_tokens: batch_endings1_padded,
            self.input_answer1_tokens_seq_len: batch_endings1_seqlen,
            self.input_answer2_tokens: batch_endings2_padded,
            self.input_answer2_tokens_seq_len: batch_endings2_seqlen,
            self.input_y: batch_y
        }

        _, \
        step, \
        res_cost, \
        res_acc \
            = sess.run([
            # graph_ops["learn_rate"],
            self.apply_grads_op,
            self.global_step,
            self.mean_loss,
            self.accuracy
        ],
            feed_dict=feed_dict)

        return res_cost, res_acc  # , res_learn_rate

    def dev_step(self, sess, batch_x_zipped, batch_y):
        batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen, batch_endings2_padded, batch_endings2_seqlen = zip(
            *batch_x_zipped)

        feed_dict = {
            self.input_story_tokens: batch_stories_padded,
            self.input_story_tokens_seq_len: batch_stories_seqlen,
            self.input_answer1_tokens: batch_endings1_padded,
            self.input_answer1_tokens_seq_len: batch_endings1_seqlen,
            self.input_answer2_tokens: batch_endings2_padded,
            self.input_answer2_tokens_seq_len: batch_endings2_seqlen,
            self.input_y: batch_y
        }

        # _, \
        step, \
        feats, \
        res_cost, \
        res_acc, \
        res_pred_y = sess.run([
            # graph_ops["learn_rate"],
            # graph_ops["apply_grads_op"],
            self.global_step,
            self.all_feats,
            self.mean_loss,
            self.accuracy,
            self.predictions
        ],
            feed_dict=feed_dict)

        return res_cost, res_acc, res_pred_y, feats  # , res_learn_rate

    @staticmethod
    def eval_step(graph, sess, batch_x_zipped, batch_y):
        batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen, batch_endings2_padded, batch_endings2_seqlen = zip(
            *batch_x_zipped)

        input_story_tokens = graph.get_operation_by_name("input_params/input_story_tokens").outputs[0]
        input_story_tokens_seq_len = graph.get_operation_by_name("input_params/input_story_tokens_seq_len").outputs[0]
        input_answer1_tokens = graph.get_operation_by_name("input_params/input_answer1_tokens").outputs[0]
        input_answer1_tokens_seq_len = graph.get_operation_by_name("input_params/input_answer1_tokens_seq_len").outputs[0]
        input_answer2_tokens = graph.get_operation_by_name("input_params/input_answer2_tokens").outputs[0]
        input_answer2_tokens_seq_len = graph.get_operation_by_name("input_params/input_answer2_tokens_seq_len").outputs[0]
        input_y = graph.get_operation_by_name("input_x").outputs[0]

        all_feats = graph.get_operation_by_name("all_feats").outputs[0]
        mean_loss = graph.get_operation_by_name("mean_loss").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        predictions = graph.get_operation_by_name("predictions").outputs[0]

        feed_dict = {
            input_story_tokens: batch_stories_padded,
            input_story_tokens_seq_len: batch_stories_seqlen,
            input_answer1_tokens: batch_endings1_padded,
            input_answer1_tokens_seq_len: batch_endings1_seqlen,
            input_answer2_tokens: batch_endings2_padded,
            input_answer2_tokens_seq_len: batch_endings2_seqlen,
            input_y: batch_y
        }

        # _, \
        feats, \
        res_cost, \
        res_acc, \
        res_pred_y = sess.run([
            # graph_ops["learn_rate"],
            # graph_ops["apply_grads_op"],
            all_feats,
            mean_loss,
            accuracy,
            predictions
        ],
            feed_dict=feed_dict)

        return res_cost, res_acc, res_pred_y, feats  # , res_learn_rate
