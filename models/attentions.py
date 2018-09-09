import tensorflow as tf
import logging
from utils.tf.tf_helpers import tf_helpers



def tf_compute_attention_base(prem_seq, prem_seq_len, hypo_seq, hypo_seq_len, hidden_size, init_random_uniform):
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

    hypo_hN = tf_helpers.tf_gather_on_axis1(hypo_seq, hypo_seq_len - tf.ones_like(hypo_seq_len))  # [b, k]
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

    M = tf.tanh(
        tf_helpers.tf_matmul_broadcast(prem_seq_masked, Wy) + tf_helpers.tf_matmul_broadcast(hypo_hN_repeat, Wh),
        name="M")  # shape = [b, k, L]

    #  [b,l,k], [k,k]->blk + bk *kk 0>bk1*1l _> T -> blk
    att_w = tf.get_variable("att_w", shape=[hidden_size, 1], initializer=init_random_uniform,
                            dtype=tf.float32)  # [k]

    tmp3 = tf.matmul(tf.reshape(M, shape=[b * l, k]), att_w)

    # need 1 here so that later can do multiplication with h x L
    attention = tf.nn.softmax(tf.reshape(tmp3, shape=[b, 1, l], name="att"))  # b x 1 x l

    # attention = tf.nn.softmax(tf_matmul_broadcast(M, tf.transpose(att_w))) # [b,l,1]

    r = tf.batch_matmul(attention, prem_seq_masked)  # [b, l, k] matmul b 1 l -> b 1 k
    r = tf.reshape(r, [b, k])  # bk

    Wp = tf.get_variable("Wp", shape=[hidden_size, hidden_size], initializer=init_random_uniform,
                         dtype=tf.float32)  # [k, k]
    bp = tf.get_variable("bp", shape=[hidden_size], initializer=init_random_uniform, dtype=tf.float32)

    Wx = tf.get_variable("Wx", shape=[hidden_size, hidden_size], initializer=init_random_uniform,
                         dtype=tf.float32)  # [k, k]
    bx = tf.get_variable("bx", shape=[hidden_size], initializer=init_random_uniform, dtype=tf.float32)

    h_star = tf.tanh(tf.matmul(r, Wp) + bp + tf.matmul(hypo_hN, Wx) + bx)  # b k 1

    attention_reshaped = tf.reshape(attention, [b, l])
    return r, attention_reshaped, h_star


# # rc task
# self.story_question_att_r, self.story_question_att_attention, self.story_question_att_weight_sum = tf_compute_attention_base(
#     encoded_story,
#     self.input_story_tokens_seq_len,
#     encoded_question,
#     self.input_question_tokens_seq_len,
#         lstm_out_layer_size)

def tf_compute_attention_dot(premise_sequence_memory, hypothesis_repr):
    """
    Computes attention over words in prem_seq by hypo_seq. Simple dot product.
    :param prem_seq: Batch of premise words representations (The output layer of an LSTM)
    :param hypo_representation: Batch of hypothesis representaitons (last state)

    :return: attention weights for the memory of premise_sequence_memory
    """
    with tf.name_scope('attention_dot') as attention_scope:
        # b = tf.shape(premise_sequence_memory)[0]  # batch


        l = tf.shape(premise_sequence_memory)[1]  # memory
        # k = tf.shape(premise_sequence_memory)[2]  # dimentions

        hypo_hN = hypothesis_repr,
        logging.info(hypo_hN)
        hypo_hN_expand = tf.expand_dims(hypo_hN, -2)  # make the same rank as the desired output
        logging.info(hypo_hN_expand)

        pattern = tf.pack([1, 1, l, 1])  # pattern for tile - repeat the second column
        hypo_hN_repeat = tf.tile(hypo_hN_expand, pattern)
        logging.info("hypo_hN_repeat:%s" % str(hypo_hN_repeat))
        hypo_hN_repeat = tf.reshape(hypo_hN_repeat, tf.shape(premise_sequence_memory))

        logging.info(hypo_hN_repeat)

        logging.info("premise_sequence_memory: %s" % str(premise_sequence_memory))

        prod = tf.reduce_sum(tf.mul(hypo_hN_repeat, premise_sequence_memory), axis=-1)
        logging.info("prod: %s" % str(prod))
        attention = tf.nn.softmax(prod)

        attention_reshape = tf.expand_dims(attention, -1)

        return attention_reshape  # attention


def tf_compute_attention_bilinear(premise_sequence_memory, hypothesis_repr, init_random_uniform, name_sufix):
    """
    Computes attention over words in prem_seq by hypothesis_repr. Simple dot product between every item from premise and the single vector hypothesis repr.
    This is the implementation from https://arxiv.org/abs/1606.02858
    :param prem_seq: Batch of premise words representations (The output layer of an LSTM)
    :param hypo_representation: Batch of hypothesis representaitons (last state)

    :return: attention weights for the memory of premise_sequence_memory
    """
    with tf.name_scope('attention_bilinear_' + name_sufix) as attention_scope:
        b = tf.shape(premise_sequence_memory)[0]  # batch
        l = tf.shape(premise_sequence_memory)[1]  # memory
        k = tf.shape(premise_sequence_memory)[2]  # dimentions

        hypo_hN = tf.reshape(hypothesis_repr, [b, k])

        logging.info(hypo_hN)

        hidden_size = k
        Wy = tf.get_variable("att_Wy",
                             shape=[hidden_size, hidden_size],
                             initializer=init_random_uniform,
                             dtype=tf.float32)

        By = tf.get_variable("att_By",
                             shape=[hidden_size, hidden_size],
                             initializer=init_random_uniform,
                             dtype=tf.float32)

        hypo_hN = tf.matmul(hypo_hN, Wy, name="repr_bilinear") + By

        logging.info(hypo_hN)
        hypo_hN_expand = tf.expand_dims(hypo_hN, -2)  # make the same rank as the desired output
        logging.info(hypo_hN_expand)

        pattern = tf.pack([1, l, 1])  # pattern for tile - repeat the second column
        hypo_hN_repeat = tf.tile(hypo_hN_expand, pattern)
        logging.info("hypo_hN_repeat:%s" % str(hypo_hN_repeat))
        hypo_hN_repeat = tf.reshape(hypo_hN_repeat, tf.shape(premise_sequence_memory))

        logging.info(hypo_hN_repeat)

        logging.info("premise_sequence_memory: %s" % str(premise_sequence_memory))

        prod = tf.reduce_sum(tf.mul(hypo_hN_repeat, premise_sequence_memory), axis=-1)
        logging.info("prod: %s" % str(prod))
        attention = tf.nn.softmax(prod)

        attention_reshape = tf.expand_dims(attention, -1)

        return attention_reshape  # attention


def tf_compute_attention_over_attention(premise_sequence_memory, hypothesis_sequence_memory):
    """
    Computes Attentio Over Attenion https://arxiv.org/abs/1607.04423 
    :param premise_sequence_memory: Context Representaiton 
    :param hypothesis_sequence_memory:  Question representaiton
    :return: Returns the attention over attention 
    """
    with tf.name_scope('attention_dot_seq') as attention_scope:
        # b = tf.shape(premise_sequence_memory)[0]  # batch
        # l1 = tf.shape(premise_sequence_memory)[1]  # memory
        # k = tf.shape(premise_sequence_memory)[2]  # dimentions

        t_hypothesis_sequence_memory = tf.transpose(hypothesis_sequence_memory, [0, 2, 1])

        prod = tf.batch_matmul(premise_sequence_memory, t_hypothesis_sequence_memory)  # attention matrix b, l1, l2

        prod_softmax_row_wise = tf.nn.softmax(prod)  # b,l1,l2
        logging.info("prod_softmax_row_wise:%s" % str(prod_softmax_row_wise))
        prod_softmax_row_wise_col_wise_sum = tf.expand_dims \
            (tf.reduce_sum(tf.transpose(prod_softmax_row_wise, [0, 2, 1]), axis=-1), -1)  # b, l2, 1
        logging.info("prod_softmax_row_wise_col_wise_sum:%s" % str(prod_softmax_row_wise_col_wise_sum))
        prod_softmax_col_wise = tf.transpose(tf.nn.softmax(tf.transpose(prod, [0, 2, 1])), [0, 2
            , 1])  # b,l1, l2 - transposed softmax on the last axis is columnwise attention?

        att_over_att = tf.squeeze(tf.batch_matmul(prod_softmax_col_wise, prod_softmax_row_wise_col_wise_sum))  # b, l1

        att_over_att = tf.expand_dims(att_over_att, axis=-1, name="att_over_att")
        return att_over_att  # attention


def tf_compute_attention_elemwise(premise_sequence_memory, hypothesis_sequence_memory):
    """
    Computes Attention between the story and question
    :param premise_sequence_memory: Context Representaiton 
    :param hypothesis_sequence_memory:  Question representaiton
    :return: Returns the attention over attention 
    """
    with tf.name_scope('attention_elemwise') as attention_scope:
        # b = tf.shape(premise_sequence_memory)[0]  # batch
        # l1 = tf.shape(premise_sequence_memory)[1]  # memory
        # k = tf.shape(premise_sequence_memory)[2]  # dimentions

        t_hypothesis_sequence_memory = tf.transpose(hypothesis_sequence_memory, [0, 2, 1])

        attention = tf.batch_matmul(premise_sequence_memory, t_hypothesis_sequence_memory)  # attention matrix b, l1, l2

        attention_soft = tf.nn.softmax(attention)  # b,l1,l2

        return attention_soft  # attention

def tf_compute_attention_elemwise_hypothesis_mask(premise_sequence_memory, hypothesis_sequence_memory, hypothesis_attention_mask):
    """
    Computes Attention between the story and question
    :param premise_sequence_memory: Context Representaiton 
    :param hypothesis_sequence_memory:  Question representaiton
    :return: Returns the attention over attention 
    """
    with tf.name_scope('attention_elemwise') as attention_scope:
        # b = tf.shape(premise_sequence_memory)[0]  # batch
        l1 = tf.shape(premise_sequence_memory)[1]  # memory
        # k = tf.shape(premise_sequence_memory)[2]  # dimentions

        t_hypothesis_sequence_memory = tf.transpose(hypothesis_sequence_memory, [0, 2, 1])

        attention = tf.batch_matmul(premise_sequence_memory, t_hypothesis_sequence_memory)  # attention matrix b, l1, l2

        hypothesis_attention_mask_tiled = tf.tile(tf.expand_dims(hypothesis_attention_mask, 1), tf.pack([1, l1, 1]))

        softmax_mask = tf_helpers.tf_mask_to_softmaxmask(hypothesis_attention_mask_tiled)  # [1,1,1, -10000000, -10000000]
        softmax_mask = softmax_mask - hypothesis_attention_mask_tiled

        attention = attention * hypothesis_attention_mask_tiled

        attention_soft = tf.nn.softmax(attention + softmax_mask) * hypothesis_attention_mask_tiled  # b,l1,l2

        return attention_soft  # attention


def tf_compute_attention_elemwise_masked(premise_sequence_memory, hypothesis_sequence_memory, hypothesis_sequence_mask):
    """
    Computes Attention between the story and question
    :param premise_sequence_memory: Context Representaiton 
    :param hypothesis_sequence_memory:  Question representaiton
    :return: Returns the attention over attention 
    """
    with tf.name_scope('attention_elemwise') as attention_scope:
        # b = tf.shape(premise_sequence_memory)[0]  # batch
        # l1 = tf.shape(premise_sequence_memory)[1]  # memory
        # k = tf.shape(premise_sequence_memory)[2]  # dimentions

        t_hypothesis_sequence_memory = tf.transpose(hypothesis_sequence_memory, [0, 2, 1])

        attention = tf.batch_matmul(premise_sequence_memory, t_hypothesis_sequence_memory)  # attention matrix b, l1, l2

        attention_soft = tf.nn.softmax(attention)  # b,l1,l2

        return attention_soft  # attention

def tf_compute_attention_elemwise_bilinear(premise_sequence_memory, hypothesis_sequence_memory, w, b):
    """
    Computes Attention between the story and question
    :param premise_sequence_memory: Context Representaiton 
    :param hypothesis_sequence_memory:  Question representaiton
    :return: Returns the attention over attention 
    """
    with tf.name_scope('attention_elemwise') as attention_scope:
        # b = tf.shape(premise_sequence_memory)[0]  # batch
        # l1 = tf.shape(premise_sequence_memory)[1]  # memory
        # k = tf.shape(premise_sequence_memory)[2]  # dimentions

        t_hypothesis_sequence_memory = tf.transpose(hypothesis_sequence_memory, [0, 2, 1])

        attention = tf.batch_matmul(premise_sequence_memory, t_hypothesis_sequence_memory)  # attention matrix b, l1, l2

        attention_soft = tf.nn.softmax(attention)  # b,l1,l2

        return attention_soft  # attention


def tf_tokens_to_key_value_knowledge_attention(encoded_doc_tokens, knowledge_keys, knowledge_values, top_facts, name, out_attention=False):
    """
    For each token in encoded_doc_tokens, select top [top_facts] closest keys and get the sum of their objects.
    The result is the same type as the [encoded_doc_tokens]
    :param encoded_doc_tokens: Sequence of encoded tokens
    :param knowledge_keys: Memory of encoded knowledge keys
    :param knowledge_values: Memory of encoded knowledge values
    :param top_facts: Top number of closes knowledge facts to the encoded token
    :return: The sum of the corresponding values for the closest keys to each token.
    """
    story_to_knowledge_subj_attention = tf_compute_attention_elemwise(encoded_doc_tokens,  # b, l1, 2*hidden_size
                                                                      knowledge_keys  # b, items_cnt, 2*hidden_size
                                                                      )  # b, l1, items_cnt

    logging.info("story_to_knowledge_subj_attention:%s" % str(story_to_knowledge_subj_attention))

    # get only top 3 facts!
    story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
        story_to_knowledge_subj_attention, top_facts, sorted=True)
    story_to_knowledge_subj_att_top_values_norm = tf.nn.softmax(
        story_to_knowledge_subj_att_top_values)  # normalize  b, l1, 3
    logging.info("story_to_knowledge_subj_att_top_values_norm:%s" % str(story_to_knowledge_subj_att_top_values_norm))

    story_to_knowledge_subj_att_top_idx_batch_flat = tf.reshape(story_to_knowledge_subj_att_top_idx,  # b, l1 * 3
                                                                shape=[tf.shape(story_to_knowledge_subj_att_top_idx)[0],
                                                                       -1])  # b, l1 * 3

    encoded_knowledge_obj_top_batch_flat = tf_helpers.tf_gather_batch_items(knowledge_values,  # b, items_cnt, hs
                                                                            story_to_knowledge_subj_att_top_idx_batch_flat
                                                                            # b, l1 * 3
                                                                            )  # b, l1 * 3, hs

    encoded_knowledge_obj_top = tf.reshape(encoded_knowledge_obj_top_batch_flat,
                                           shape=[tf.shape(story_to_knowledge_subj_att_top_idx)[0],  # b,
                                                  tf.shape(story_to_knowledge_subj_att_top_idx)[1],  # l1,
                                                  tf.shape(story_to_knowledge_subj_att_top_idx)[2],  # 3
                                                  tf.shape(knowledge_values)[-1]]  # hs
                                           )
    logging.info(
        "encoded_knowledge_obj_top:%s" % str(encoded_knowledge_obj_top))

    story_to_knowledge_obj_repr = tf.reduce_mean(
        tf.expand_dims(story_to_knowledge_subj_att_top_values_norm, -1)  # b, l1, 3
        * encoded_knowledge_obj_top,  # b, l1, 3, hs
        axis=-2, name=name)

    if out_attention:
        return story_to_knowledge_obj_repr, story_to_knowledge_subj_attention, story_to_knowledge_subj_att_top_idx, story_to_knowledge_subj_att_top_values_norm
    else:
        return story_to_knowledge_obj_repr

def tf_tokens_to_key_value_knowledge_attention_masked(encoded_doc_tokens, knowledge_keys, knowledge_values, top_facts, name,
                                                      out_attention=False, knowledge_attention_mask=None):
    """
    For each token in encoded_doc_tokens, select top [top_facts] closest keys and get the sum of their objects.
    The result is the same type as the [encoded_doc_tokens]
    :param encoded_doc_tokens: Sequence of encoded tokens
    :param knowledge_keys: Memory of encoded knowledge keys
    :param knowledge_values: Memory of encoded knowledge values
    :param top_facts: Top number of closes knowledge facts to the encoded token
    :return: The sum of the corresponding values for the closest keys to each token.
    """

    story_to_knowledge_subj_attention = tf_compute_attention_elemwise_hypothesis_mask(encoded_doc_tokens,  # b, l1, 2*hidden_size
                                                                      knowledge_keys,  # b, items_cnt, 2*hidden_size
                                                                      knowledge_attention_mask # b, item_cnt
                                                                      )  # b, l1, items_cnt

    logging.info("story_to_knowledge_subj_attention:%s" % str(story_to_knowledge_subj_attention))

    # get only top 3 facts!
    story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
        story_to_knowledge_subj_attention, top_facts, sorted=True)
    story_to_knowledge_subj_att_top_values_norm = tf.nn.softmax(
        story_to_knowledge_subj_att_top_values)  # normalize  b, l1, 3
    logging.info("story_to_knowledge_subj_att_top_values_norm:%s" % str(story_to_knowledge_subj_att_top_values_norm))

    story_to_knowledge_subj_att_top_idx_batch_flat = tf.reshape(story_to_knowledge_subj_att_top_idx,  # b, l1 * 3
                                                                shape=[tf.shape(story_to_knowledge_subj_att_top_idx)[0],
                                                                       -1])  # b, l1 * 3

    encoded_knowledge_obj_top_batch_flat = tf_helpers.tf_gather_batch_items(knowledge_values,  # b, items_cnt, hs
                                                                            story_to_knowledge_subj_att_top_idx_batch_flat
                                                                            # b, l1 * 3
                                                                            )  # b, l1 * 3, hs

    encoded_knowledge_obj_top = tf.reshape(encoded_knowledge_obj_top_batch_flat,
                                           shape=[tf.shape(story_to_knowledge_subj_att_top_idx)[0],  # b,
                                                  tf.shape(story_to_knowledge_subj_att_top_idx)[1],  # l1,
                                                  tf.shape(story_to_knowledge_subj_att_top_idx)[2],  # 3
                                                  tf.shape(knowledge_values)[-1]]  # hs
                                           )
    logging.info(
        "encoded_knowledge_obj_top:%s" % str(encoded_knowledge_obj_top))

    story_to_knowledge_obj_repr = tf.reduce_mean(
        tf.expand_dims(story_to_knowledge_subj_att_top_values_norm, -1)  # b, l1, 3
        * encoded_knowledge_obj_top,  # b, l1, 3, hs
        axis=-2, name=name)

    if out_attention:
        return story_to_knowledge_obj_repr, story_to_knowledge_subj_attention, story_to_knowledge_subj_att_top_idx, story_to_knowledge_subj_att_top_values_norm
    else:
        return story_to_knowledge_obj_repr

def tf_tokens_to_key_value_knowledge_attention_full(encoded_doc_tokens, knowledge_keys, knowledge_values, name, output_top_facts=0, out_attention=False):
    """
    For each token in encoded_doc_tokens, select top [top_facts] closest keys and get the sum of their objects.
    The result is the same type as the [encoded_doc_tokens]
    :param encoded_doc_tokens: Sequence of encoded tokens
    :param knowledge_keys: Memory of encoded knowledge keys
    :param knowledge_values: Memory of encoded knowledge values
    :param top_facts: Top number of closes knowledge facts to the encoded token
    :return: The sum of the corresponding values for the closest keys to each token.
    """
    story_to_knowledge_subj_attention = tf_compute_attention_elemwise(encoded_doc_tokens,  # b, l1, k
                                                                      knowledge_keys  # b, items_cnt, k
                                                                      )  # b, l1, items_cnt


    attention_sum = tf.batch_matmul(story_to_knowledge_subj_attention,  knowledge_values, name = name)

    if output_top_facts>0:
        story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
            story_to_knowledge_subj_attention, output_top_facts, sorted=True)
    else:
        story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
            story_to_knowledge_subj_attention, tf.shape(story_to_knowledge_subj_attention)[-1], sorted=True)


    if out_attention:
        return attention_sum, story_to_knowledge_subj_attention, story_to_knowledge_subj_att_top_idx, story_to_knowledge_subj_att_top_values
    else:
        return attention_sum

def tf_tokens_to_key_value_knowledge_attention_full_masked(encoded_doc_tokens, knowledge_keys, knowledge_values, name,
                                                           output_top_facts=0, out_attention=False,
                                                           knowledge_attention_mask=None):
    """
    For each token in encoded_doc_tokens, select top [top_facts] closest keys and get the sum of their objects.
    The result is the same type as the [encoded_doc_tokens]
    :param encoded_doc_tokens: Sequence of encoded tokens
    :param knowledge_keys: Memory of encoded knowledge keys
    :param knowledge_values: Memory of encoded knowledge values
    :param top_facts: Top number of closes knowledge facts to the encoded token
    :return: The sum of the corresponding values for the closest keys to each token.
    """
    story_to_knowledge_subj_attention = tf_compute_attention_elemwise_hypothesis_mask(encoded_doc_tokens,  # b, l1, 2*hidden_size
                                                                      knowledge_keys,  # b, items_cnt, 2*hidden_size
                                                                      knowledge_attention_mask # b, item_cnt
                                                                      )  # b, l1, items_cnt


    attention_sum = tf.batch_matmul(story_to_knowledge_subj_attention,  knowledge_values, name = name)

    if output_top_facts>0:
        story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
            story_to_knowledge_subj_attention, output_top_facts, sorted=True)
    else:
        story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
            story_to_knowledge_subj_attention, tf.shape(story_to_knowledge_subj_attention)[-1], sorted=True)


    if out_attention:
        return attention_sum, story_to_knowledge_subj_attention, story_to_knowledge_subj_att_top_idx, story_to_knowledge_subj_att_top_values
    else:
        return attention_sum



def tf_tokens_to_key_value_knowledge_similarity(encoded_doc_tokens, knowledge_keys, knowledge_values, top_facts, name, out_attention=False):
    """
    For each token in encoded_doc_tokens, select top [top_facts] closest keys and get the sum of their objects.
    The result is the same type as the [encoded_doc_tokens]
    :param encoded_doc_tokens: Sequence of encoded tokens
    :param knowledge_keys: Memory of encoded knowledge keys
    :param knowledge_values: Memory of encoded knowledge values
    :param top_facts: Top number of closes knowledge facts to the encoded token
    :return: The sum of the corresponding values for the closest keys to each token.
    """
    story_to_knowledge_subj_attention = tf_compute_attention_elemwise(encoded_doc_tokens,  # b, l1, 2*hidden_size
                                                                      knowledge_keys  # b, items_cnt, 2*hidden_size
                                                                      )  # b, l1, items_cnt

    logging.info("story_to_knowledge_subj_attention:%s" % str(story_to_knowledge_subj_attention))

    # get only top 3 facts!
    story_to_knowledge_subj_att_top_values, story_to_knowledge_subj_att_top_idx = tf.nn.top_k(
        story_to_knowledge_subj_attention, top_facts, sorted=True)
    story_to_knowledge_subj_att_top_values_norm = tf.nn.softmax(
        story_to_knowledge_subj_att_top_values)  # normalize  b, l1, 3
    logging.info("story_to_knowledge_subj_att_top_values_norm:%s" % str(story_to_knowledge_subj_att_top_values_norm))

    story_to_knowledge_subj_att_top_idx_batch_flat = tf.reshape(story_to_knowledge_subj_att_top_idx,  # b, l1 * 3
                                                                shape=[tf.shape(story_to_knowledge_subj_att_top_idx)[0],
                                                                       -1])  # b, l1 * 3

    encoded_knowledge_obj_top_batch_flat = tf_helpers.tf_gather_batch_items(knowledge_values,  # b, items_cnt, hs
                                                                            story_to_knowledge_subj_att_top_idx_batch_flat
                                                                            # b, l1 * 3
                                                                            )  # b, l1 * 3, hs

    encoded_knowledge_obj_top = tf.reshape(encoded_knowledge_obj_top_batch_flat,
                                           shape=[tf.shape(story_to_knowledge_subj_att_top_idx)[0],  # b,
                                                  tf.shape(story_to_knowledge_subj_att_top_idx)[1],  # l1,
                                                  tf.shape(story_to_knowledge_subj_att_top_idx)[2],  # 3
                                                  tf.shape(knowledge_values)[-1]]  # hs
                                           )
    logging.info(
        "encoded_knowledge_obj_top:%s" % str(encoded_knowledge_obj_top))

    story_to_knowledge_obj_repr = tf.reduce_mean(
        tf.expand_dims(story_to_knowledge_subj_att_top_values_norm, -1)  # b, l1, 3
        * encoded_knowledge_obj_top,  # b, l1, 3, hs
        axis=-2, name=name)

    if out_attention:
        return story_to_knowledge_obj_repr, story_to_knowledge_subj_attention, story_to_knowledge_subj_att_top_idx, story_to_knowledge_subj_att_top_values_norm
    else:
        return story_to_knowledge_obj_repr