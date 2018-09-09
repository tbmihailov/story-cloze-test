import tensorflow as tf

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

class StoryClozeMemoryNetworkReader(object):
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
                 bidirectional=False,
                 sent_encoder="BoW",  # "PE", "BoW", "LSTM"
                 memory_size=4
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
            self.input_story_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="input_story_tokens")
            self.input_answer1_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_answer1_tokens")
            self.input_answer2_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_answer2_tokens")

            # sequence lengths
            self.input_story_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                             name="input_story_tokens_seq_len")  # length of every sequence in the batch
            self.input_answer1_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                               name="input_answer1_tokens_seq_len")  # length of every sequence in the batch
            self.input_answer2_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                               name="input_answer2_tokens_seq_len")  # length of every sequence in the batch

            self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name="input_y")  # this is onehot!

        # mask
        self.input_story_tokens_mask = tf.expand_dims(
            tf.reshape(tf.cast(tf.sequence_mask(tf.reshape(self.input_story_tokens_seq_len, [-1])), dtype=tf.float32),[tf.shape(self.input_story_tokens_seq_len)[0],tf.shape(self.input_story_tokens_seq_len)[1],-1],
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
                self.embeddings_A = tf.get_variable("embeddings_A",
                                                        shape=embeddings.shape,
                                                        initializer=tf.constant_initializer(embeddings),
                                                        trainable=self.embeddings_trainable,
                                                        dtype=tf.float32)

                self.embeddings_B = tf.get_variable("embeddings_B",
                                                        shape=embeddings.shape,
                                                        initializer=tf.constant_initializer(embeddings),
                                                        trainable=self.embeddings_trainable,
                                                        dtype=tf.float32)

                self.embeddings_C = tf.get_variable("embeddings_C",
                                                        shape=embeddings.shape,
                                                        initializer=tf.constant_initializer(embeddings),
                                                        trainable=self.embeddings_trainable,
                                                        dtype=tf.float32)
            else:
                self.embeddings_A = tf.get_variable("embeddings_A",
                                        initializer=tf.truncated_normal([embeddings_number, embeddings_size], stddev=0.1, dtype=tf.float32),
                                        trainable=self.embeddings_trainable,
                                        dtype=tf.float32)

                self.embeddings_B = tf.get_variable("embeddings_B",
                                        initializer=tf.truncated_normal([embeddings_number, embeddings_size], stddev=0.1, dtype=tf.float32),
                                        trainable=self.embeddings_trainable,
                                        dtype=tf.float32)

                self.embeddings_C = tf.get_variable("embeddings_C",
                                                    initializer=tf.truncated_normal([embeddings_number, embeddings_size], stddev=0.1,
                                                                                    dtype=tf.float32),
                                                    trainable=self.embeddings_trainable,
                                                    dtype=tf.float32)

            # self.embedded_chars_story = tf.nn.embedding_lookup(self.embeddings_A, self.input_story_tokens)
            # self.embedded_chars_answer1 = tf.nn.embedding_lookup(self.embeddings_A, self.input_answer1_tokens)
            # self.embedded_chars_answer2 = tf.nn.embedding_lookup(self.embeddings_A, self.input_answer2_tokens)


        def encode_bow(enc_embeddings, data_tokens, data_seq_lens, data_seq_mask, name):
            embedded_data = tf.nn.embedding_lookup(enc_embeddings, data_tokens)
            encoded = tf.cast(tf.reduce_sum(embedded_data * data_seq_mask,
                                                   reduction_indices=-2), tf.float32) / tf.expand_dims(tf.cast(data_seq_lens, dtype=tf.float32), -1)
            return tf.identity(encoded, name=name)


        def tf_compute_attention_base(prem_mem, hypo_mem, hidden_size):
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
            with tf.variable_scope('attention') as attention_scope:
                prem_seq_masked = prem_mem
                b = tf.shape(prem_mem)[0]
                l = tf.shape(prem_mem)[1]
                k = tf.shape(prem_mem)[2]

                hypo_hN = hypo_mem
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

        def tf_compute_attention_mem_n2n(prem_mem, hypo_mem, mem_size, hidden_size):
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
            with tf.name_scope('attention') as attention_scope:

                b = tf.shape(prem_mem)[0]  # batch
                l = tf.shape(prem_mem)[1]  # memory
                k = tf.shape(prem_mem)[2]  # dimentions

                logging.info("memory size: %s" % l)

                hypo_hN = hypo_mem,
                logging.info(hypo_hN)
                hypo_hN_expand = tf.expand_dims(hypo_hN, -2)  # make the same rank as the desired output
                logging.info(hypo_hN_expand)

                pattern = tf.pack([1, 1, l, 1])  # pattern for tile - repeat the second column
                hypo_hN_repeat = tf.tile(hypo_hN_expand, pattern)
                logging.info("hypo_hN_repeat:%s" % str(hypo_hN_repeat))
                hypo_hN_repeat = tf.reshape(hypo_hN_repeat, tf.shape(prem_mem))

                logging.info(hypo_hN_repeat)

                logging.info("prem_mem: %s" % str(prem_mem))

                prod = tf.reduce_sum(tf.mul(hypo_hN_repeat, prem_mem), axis=-1)
                logging.info("prod: %s" % str(prod))
                attention = tf.nn.softmax(prod)

                attention_reshape = tf.expand_dims(attention, -1)


                return attention_reshape # attention

        hops = bilstm_layers

        encoded_answer1 = encode_bow(self.embeddings_B, self.input_answer1_tokens,
                                      self.input_answer1_tokens_seq_len,
                                      self.input_answer1_tokens_mask, name="encoded_answer1_B")

        encoded_answer2 = encode_bow(self.embeddings_B, self.input_answer2_tokens,
                                       self.input_answer2_tokens_seq_len,
                                       self.input_answer2_tokens_mask, name="encoded_answer2_B")

        logging.info("encoded_answer1: %s" % str(encoded_answer1))
        logging.info("encoded_answer2: %s" % str(encoded_answer2))
        # H = tf.get_variable("H", shape=[hidden_size, hidden_size], initializer=init_random_uniform,
        #                      dtype=tf.float32)  # [k, k]

        for i in range(hops):
            with tf.variable_scope("hop_%s" % i) as vs_hop:
                logging.info("---------------------")
                # encode
                encoded_story_memory_A = encode_bow(self.embeddings_A, self.input_story_tokens, self.input_story_tokens_seq_len,
                                                self.input_story_tokens_mask, name="encoded_story_A")
                logging.info(encoded_story_memory_A)

                encoded_story_memory_C = encode_bow(self.embeddings_C, self.input_story_tokens,
                                                         self.input_story_tokens_seq_len,
                                                         self.input_story_tokens_mask, name="encoded_story_C")

                # attention
                att_answer1_story = tf_compute_attention_mem_n2n(encoded_story_memory_A, encoded_answer1, memory_size,
                                                                hidden_size=embeddings_size)

                logging.info("att_answer1_story: %s" % str(att_answer1_story))
                att_answer2_story = tf_compute_attention_mem_n2n(encoded_story_memory_A, encoded_answer2, memory_size,
                                                                hidden_size=embeddings_size)

                # out representations
                repr_out_answer1_story = tf.reduce_sum(tf.mul(att_answer1_story, encoded_story_memory_C), 1)

                repr_out_answer2_story = tf.reduce_sum(tf.mul(att_answer2_story, encoded_story_memory_C), 1)

                # next output
                encoded_answer1 = encoded_answer1 + repr_out_answer1_story
                encoded_answer2 = encoded_answer2 + repr_out_answer2_story

        logging.info("encoded_answer1:%s" % str(encoded_answer1))
        logging.info("encoded_answer2:%s" % str(encoded_answer2))

        self.all_feats = tf.concat(1, [encoded_answer1, encoded_answer2])

        output_layer_size = embeddings_size*2

        logging.info("all_feats:%s " % self.all_feats.get_shape())

        # all_feats = tf.nn.dropout(all_feats, keep_prob=dropout_prob_keep)
        if layer_out_noise > 0.01:
            self.all_feats = gaussian_noise_layer(self.all_feats, std=layer_out_noise)

        self.all_feats = tf.identity(self.all_feats, name="all_feats")

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        # bi_output_concat_flat = tf.reshape(output_layer, [-1, output_layer_size])
        # bi_output_concat_flat_clipped = tf.clip_by_value(bi_output_concat_flat, -1., 1.)  # use clipping if needed. Currently not

        use_mlp = False
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
        vals = zip(*batch_x_zipped)
        # logging.info(vals)
        batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen, batch_endings2_padded, batch_endings2_seqlen = vals

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
