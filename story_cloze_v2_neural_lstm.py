import codecs
import os
import sys

import logging  # word2vec logging

import numpy as np

from gensim.models.word2vec import Word2Vec  # used for word2vec

import time  # used for performance measuring
from utils.embedding_vector_utilities import AverageVectorsUtilities

import pickle

from data.StoryClozeTest.DataUtilities_ROCStories import DataUtilities_ROCStories
import time as ti

from utils.label_dictionary import LabelDictionary

from utils.embedding_utilities import VocabEmbeddingUtilities
import tensorflow as tf

from models.story_cloze_lstm_reader import StoryClozeLSTMReader

from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def extract_word_frequencies(input_data,
                             data_fields_sents=["sentences", "endings"],
                             tokens_field="tokens",
                             max_nr_sent=0,
                             tokens_lowercase=False):
    vocab_freq = {}
    nr_sent = 0

    # for dir_data in data:
    for data_item in input_data:
        for data_fieldname in data_fields_sents:
            if not data_fieldname in data_item:
                continue

            for sent in data_item[data_fieldname]:
                raw_x = sent[tokens_field]
                for i in range(0, len(raw_x)):
                    word = raw_x[i] if not tokens_lowercase else raw_x[i].lower()

                    if word not in vocab_freq:
                        vocab_freq[word] = 1
                    else:
                        vocab_freq[word] += 1

        nr_sent += 1

        if max_nr_sent > 0 and nr_sent >= max_nr_sent:
            break

    word_counts = [(k, v) for k, v in vocab_freq.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)

    return word_counts


def lowercase_list(lst, lowercase):
    return [x.lower() if lowercase else x for x in lst]


def transofrm_data_word_to_id(input_data,
                              vocab_token_to_id,
                              unknown_token_id=0,
                              tokens_field="tokens",
                              lowercase=True,
                              mutate_data=False):
    input_list_story_ids = []
    input_list_stories = []
    input_list_endings1 = []
    input_list_endings2 = []
    input_list_labels = []

    vocab_set = set(vocab_token_to_id.keys())
    for data_item in input_data:
        story = []

        for sent_data in data_item["sentences"]:
            sent = [vocab_token_to_id[x] if x in vocab_set else unknown_token_id for x in
                    lowercase_list(sent_data[tokens_field], lowercase)]
            story.extend(sent)

        ending1 = [vocab_token_to_id[x] if x in vocab_set else unknown_token_id for x in
                   lowercase_list(data_item["endings"][0][tokens_field], lowercase)]
        ending2 = [vocab_token_to_id[x] if x in vocab_set else unknown_token_id for x in
                   lowercase_list(data_item["endings"][1][tokens_field], lowercase)]

        label = data_item["right_end_id"]
        story_id = data_item["id"]

        # append
        input_list_story_ids.append(story_id)
        input_list_stories.append(story)
        input_list_endings1.append(ending1)
        input_list_endings2.append(ending2)
        input_list_labels.append(label)

        if mutate_data:
            input_list_story_ids.append(story_id)
            input_list_stories.append(story[:])
            input_list_endings1.append(ending2[:])
            input_list_endings2.append(ending1[:])
            input_list_labels.append(1 if label == 0 else 0)

    return input_list_stories, input_list_endings1, input_list_endings2, input_list_labels, input_list_story_ids


def pad(seq, pad_value, to_size):
    pad_seq = []
    if len(seq) > to_size:
        pad_seq = seq[:to_size]
    else:
        pad_seq = seq[:] + [pad_value] * (to_size - len(seq))

    return pad_seq


def pad_data_and_return_seqlens(data, pad_value=0):
    batch_data_seqlens = np.asarray([len(a) for a in data])
    max_len = max(batch_data_seqlens)

    batch_data_padded_x = np.asarray([pad(a, pad_value, max_len) for a in data])

    return batch_data_padded_x, batch_data_seqlens


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class StoryCloze_v2_lstm_v1(object):
    def __init__(self, output_dir,
                 checkpoint_dir,
                 checkpoint_prefix,
                 checkpoint_best,
                 embeddings,
                 embeddings_vocab,
                 classifier_name,
                 run_name,
                 model_dir):

        self._output_dir = output_dir

        self._classifier_name = classifier_name
        self._run_name = run_name
        self._model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info("model_dir %s was created!" % model_dir)

        self._classifier_settings_dir = "%s/%s" % (model_dir, classifier_name)
        if not os.path.exists(self._classifier_settings_dir):
            os.makedirs(self._classifier_settings_dir)
            logging.info("%s was created!" % self._classifier_settings_dir)

        self._vocab_and_embeddings_file = "%s/vocab_and_embeddings.pickle" % (self._classifier_settings_dir)

        self._checkpoint_dir = "%s/%s" % (model_dir, checkpoint_dir)
        self._checkpoint_prefix = checkpoint_prefix
        self._checkpoint_best = checkpoint_best

        self.unknown_word = "<UNKNWN>"
        self.pad_word = "<PAD>"

        self._settings = {}
        pass

    def train(self,
              input_dataset,
              options,
              embeddings,
              embeddings_vocab
              ):
        scale_features = options.scale_features
        model_file = options.model_file
        scale_file = options.scale_file

        embeddings_type = options.emb_model_type
        embeddings_vocab_set = set([])
        if embeddings_type == "w2v":
            embeddings_vocab_set = set(embeddings.index2word)

        # train data
        input_data = DataUtilities_ROCStories.load_data_from_json_file(input_dataset)
        logging.info("input_data fields:\n%s" % str(input_data[0].keys()))

        logging.info("input_data[0] train:\n%s" % str(input_data[0]))
        if options.max_records and options.max_records > 0:
            logging.info("max_records:%s" % options.max_records)
            input_data = input_data[:options.max_records]
        # logging.info(input_data[0])
        logging.info("Items to process:%s" % len(input_data))

        # dev data
        input_data_dev = DataUtilities_ROCStories.load_data_from_json_file(options.input_data_eval)
        logging.info("input_data_dev[0] dev:\n%s" % str(input_data_dev[0]))
        # if options.max_records and options.max_records > 0:
        #     logging.info("max_records:%s" % options.max_records)
        #     input_data_dev = input_data_dev[:options.max_records]
        # logging.info(input_data_dev[0])
        logging.info("Dev Items to process:%s" % len(input_data_dev))

        # extract vocab
        word_usage_stat = extract_word_frequencies(input_data)

        min_word_freq = options.min_word_freq
        logging.info("min_word_freq:%s" % min_word_freq)
        # word_usage_stat = [xx for xx in word_usage_stat if xx[1] >= min_word_freq]
        word_usage_stat = [xx for xx in word_usage_stat]

        vocab_more_freq = [xx for xx in word_usage_stat if xx[1] >= min_word_freq]
        logging.info("words_in_vocab freq >= %s: %s" % (len(vocab_more_freq), min_word_freq))

        # clear the vocab
        vocab = [xx[0] for xx in word_usage_stat]
        logging.info("all words_in_vocab:%s" % len(vocab_more_freq))

        words_in_embeddings_vocab = [w for w in vocab if w in embeddings_vocab_set]
        logging.info("words_in_vocab in embedd vocab:%s" % len(words_in_embeddings_vocab))

        words_not_in_embeddings_vocab = [w for w in vocab if w not in embeddings_vocab_set]
        logging.info("words_in_vocab not in embedd vocab:%s" % len(words_not_in_embeddings_vocab))

        vocab = words_in_embeddings_vocab
        # Add pad and unknown tokens
        unknown_word = self.unknown_word
        pad_word = self.pad_word

        vocab.insert(0, pad_word)
        vocab.insert(0, unknown_word)

        vocab_dict = LabelDictionary(vocab, start_index=0)

        logging.info("Get average model vector for unknown_vec..")
        st = ti.time()

        vocab_and_embeddings = {}
        if embeddings_type == "w2v":
            unknown_vec = AverageVectorsUtilities.makeFeatureVec(words=list(embeddings_vocab_set),
                                                                 model=embeddings_model,
                                                                 num_features=embeddings_vec_size,
                                                                 index2word_set=embeddings_vocab_set)

            pad_vec = unknown_vec * 0.25
            logging.info("Done in %s s" % (ti.time() - st))

            logging.info("Loading embeddings for vocab..")
            st = ti.time()

            vocab_and_embeddings = VocabEmbeddingUtilities \
                .get_embeddings_for_vocab_from_model(vocabulary=vocab_dict,
                                                     embeddings_type='w2v',
                                                     embeddings_model=embeddings_model,
                                                     embeddings_size=embeddings_vec_size)

            vocab_and_embeddings["embeddings"][vocab_and_embeddings["vocabulary"][unknown_word], :] = unknown_vec
            vocab_and_embeddings["embeddings"][vocab_and_embeddings["vocabulary"][pad_word], :] = pad_vec

            logging.info("Done in %s s" % (ti.time() - st))

        elif embeddings_type == "rand":
            np.random.seed(123)
            random_embeddings = np.random.uniform(-0.1, 0.1, (len(vocab), embeddings_vec_size))
            vocab_dict_emb = dict([(k, i) for i, k in enumerate(vocab)])

            vocab_and_embeddings["embeddings"] = random_embeddings
            vocab_and_embeddings["vocabulary"] = vocab_dict_emb
        else:
            raise Exception("embeddings_type=%s is not supported!" % embeddings_type)

        # save vocab and embeddings
        pickle.dump(vocab_and_embeddings, open(self._vocab_and_embeddings_file, 'wb'))
        logging.info('Vocab and embeddings saved to: %s' % self._vocab_and_embeddings_file)

        # Load the data for labeling
        corpus_vocab_input = LabelDictionary()
        corpus_vocab_input.set_dict(vocab_and_embeddings["vocabulary"])

        # labels_lst = [u'O', u'B-EVENT', u'I-EVENT']
        classes_dict = {0: u'0', 1: u'1'}
        n_classes = len(classes_dict)
        ending_to_class_mapping = {0: 0, 1: 0}

        labels_lst = [x[1] for x in classes_dict.items()]

        self._settings["labels_lst"] = labels_lst
        self._settings["classes_dict"] = classes_dict

        # Settings
        batch_size = options.batch_size
        num_epochs = options.num_epochs

        # TRAIN DATA
        # convert words to vocab

        input_list_stories, input_list_endings1, input_list_endings2, input_list_labels, input_list_story_ids = transofrm_data_word_to_id(
            input_data,
            corpus_vocab_input,
            mutate_data=True,
            lowercase=True)  # mutate_data means that we generate 2 entries - Ending1, Ending 2 and Ending2, Ending 1

        data_train_all = list(
            zip(input_list_stories, input_list_endings1, input_list_endings2, input_list_labels, input_list_story_ids))
        data_len = len(data_train_all)
        evaluate_every = data_len / batch_size
        checkpoint_every = evaluate_every

        train_batches = batch_iter(
            data_train_all, batch_size, num_epochs, shuffle=True)

        # DEV data
        batch_size_train = 300
        in_dev_list_stories, in_dev_list_endings1, in_dev_list_endings2, in_dev_list_labels, in_dev_list_story_ids = transofrm_data_word_to_id(
            input_data_dev,
            corpus_vocab_input,
            mutate_data=False,
            lowercase=True
        )  # mutate_data means that we generate 2 entries - Ending1, Ending 2 and Ending2, Ending 1

        data_dev_all = list(
            zip(in_dev_list_stories, in_dev_list_endings1, in_dev_list_endings2, in_dev_list_labels,
                in_dev_list_story_ids))

        def eval_data(data_dev_all, gold_labels):
            dev_batches = batch_iter(
                data_dev_all, batch_size_train, 1, shuffle=False)

            predictions_all = []
            for batch in dev_batches:
                batch_stories, batch_endings1, batch_endings2, batch_labels, _ = zip(*batch)
                batch_stories_padded, batch_stories_seqlen = pad_data_and_return_seqlens(batch_stories)
                batch_endings1_padded, batch_endings1_seqlen = pad_data_and_return_seqlens(batch_endings1)
                batch_endings2_padded, batch_endings2_seqlen = pad_data_and_return_seqlens(batch_endings2)

                res_cost, res_acc, res_pred_y = dev_step(
                    zip(batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen,
                        batch_endings2_padded, batch_endings2_seqlen
                        ), batch_labels)

                predictions_all.extend(res_pred_y)

            logging.info("Confusion matrix:")
            conf_matrix = confusion_matrix(gold_labels, predictions_all)
            logging.info("\n" + str(conf_matrix))
            res = precision_recall_fscore_support(gold_labels, predictions_all)
            logging.info(
                "precision_recall_fscore_support:%s" % str(res))

            logging.info("accuracy_score:%s" % accuracy_score(gold_labels, predictions_all))
            return res

        allow_soft_placement = True
        log_device_placement = True

        embeddings_matrix = vocab_and_embeddings["embeddings"]
        lstm_layers = 1
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=allow_soft_placement,
                log_device_placement=log_device_placement)

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                dnn_model = StoryClozeLSTMReader(
                    n_classes=n_classes,
                    embeddings=embeddings_matrix,
                    embeddings_size=embeddings_matrix.shape[1],
                    embeddings_number=embeddings_matrix.shape[0],
                    bilstm_layers=options.lstm_layers,
                    hidden_size=options.lstm_hidden_size,
                    embeddings_trainable=options.train_embeddings,
                    dropout_prob_keep=options.dropout_prob_keep,
                    learn_rate=options.learn_rate,
                    loss_l2_beta=options.loss_l2_beta,
                    layer_out_noise=options.layer_out_noise
                )


                graph_ops = {}
                graph_ops["mean_loss"] = dnn_model.mean_loss
                graph_ops["apply_grads_op"] = dnn_model.apply_grads_op
                graph_ops["pred_y"] = dnn_model.predictions

                # graph_ops["learn_rate"] = learn_rate
                with tf.name_scope("accuracy"):
                    # Calculate the accuracy
                    # Used during training
                    # Mask the losses - padded values are zeros

                    # dnn_model.losses
                    # dnn_model.predictions
                    correct_pred = tf.equal(tf.cast(dnn_model.predictions, dtype=tf.int32), dnn_model.input_y)

                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                    graph_ops["accuracy"] = accuracy

                # global_step = tf.Variable(0, name="global_step", trainable=False)

                # Training loop. For each batch...
                def train_step(batch_x_zipped, batch_y):
                    batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen, batch_endings2_padded, batch_endings2_seqlen = zip(
                        *batch_x_zipped)

                    feed_dict = {
                        dnn_model.input_story_tokens: batch_stories_padded,
                        dnn_model.input_story_tokens_seq_len: batch_stories_seqlen,
                        dnn_model.input_answer1_tokens: batch_endings1_padded,
                        dnn_model.input_answer1_tokens_seq_len: batch_endings1_seqlen,
                        dnn_model.input_answer2_tokens: batch_endings2_padded,
                        dnn_model.input_answer2_tokens_seq_len: batch_endings2_seqlen,
                        dnn_model.input_y: batch_y
                    }

                    _, \
                    step, \
                    res_cost, \
                    res_acc \
                        = sess.run([
                        # graph_ops["learn_rate"],
                        graph_ops["apply_grads_op"],
                        dnn_model.global_step,
                        graph_ops["mean_loss"],
                        graph_ops["accuracy"],
                    ],
                        feed_dict=feed_dict)

                    return res_cost, res_acc  # , res_learn_rate

                def dev_step(batch_x_zipped, batch_y):
                    batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen, batch_endings2_padded, batch_endings2_seqlen = zip(
                        *batch_x_zipped)

                    feed_dict = {
                        dnn_model.input_story_tokens: batch_stories_padded,
                        dnn_model.input_story_tokens_seq_len: batch_stories_seqlen,
                        dnn_model.input_answer1_tokens: batch_endings1_padded,
                        dnn_model.input_answer1_tokens_seq_len: batch_endings1_seqlen,
                        dnn_model.input_answer2_tokens: batch_endings2_padded,
                        dnn_model.input_answer2_tokens_seq_len: batch_endings2_seqlen,
                        dnn_model.input_y: batch_y
                    }

                    # _, \
                    step, \
                    res_cost, \
                    res_acc, \
                    res_pred_y = sess.run([
                        # graph_ops["learn_rate"],
                        # graph_ops["apply_grads_op"],
                        dnn_model.global_step,
                        graph_ops["mean_loss"],
                        graph_ops["accuracy"],
                        graph_ops["pred_y"]
                    ],
                        feed_dict=feed_dict)

                    return res_cost, res_acc, res_pred_y  # , res_learn_rate

                checkpoint_dir = self._checkpoint_dir
                checkpoint_prefix = self._checkpoint_prefix
                checkpoint_best = self._checkpoint_best

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # Training
                saver = tf.train.Saver(tf.all_variables(),
                                       max_to_keep=10,
                                       keep_checkpoint_every_n_hours=1.0)

                init_vars = tf.initialize_all_variables()
                sess.run(init_vars)

                epoch_start = ti.time()
                epoch_steps = 0
                for batch in train_batches:
                    epoch_steps += 1
                    batch_stories, batch_endings1, batch_endings2, batch_labels, _ = zip(*batch)
                    batch_stories_padded, batch_stories_seqlen = pad_data_and_return_seqlens(batch_stories)
                    batch_endings1_padded, batch_endings1_seqlen = pad_data_and_return_seqlens(batch_endings1)
                    batch_endings2_padded, batch_endings2_seqlen = pad_data_and_return_seqlens(batch_endings2)

                    res_cost, res_acc = train_step(
                        zip(batch_stories_padded, batch_stories_seqlen, batch_endings1_padded, batch_endings1_seqlen,
                            batch_endings2_padded, batch_endings2_seqlen
                            ), batch_labels)
                    current_step = tf.train.global_step(sess, dnn_model.global_step)

                    logging.info("Step %s , loss=%s, batch_accuracy=%s" % (current_step, res_cost, res_acc))
                    if current_step % evaluate_every == 0:
                        logging.info("=== %s steps in %s" %(epoch_steps, ti.time()-epoch_start))
                        logging.info("\nEvaluation:")
                        eval_start = ti.time()
                        eval_data(data_dev_all, in_dev_list_labels)
                        logging.info("Done in %s" % (ti.time() - eval_start))
                        logging.info("")
                        if current_step % checkpoint_every == 0:
                            path = saver.save(sess, "%s/%s" % (checkpoint_dir, checkpoint_prefix), global_step=current_step)
                        logging.info("Saved model checkpoint to {}\n".format(path))

                            # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            # print("Saved model checkpoint to {}\n".format(path))
                        epoch_start = ti.time()
                        epoch_steps = 0
        pass

    def eval(self,
             input_dataset,
             options):
        scale_features = options.scale_features
        model_file = options.model_file
        scale_file = options.scale_file

        input_data = DataUtilities_ROCStories.load_data_from_json_file(input_dataset)
        logging.info("input_data[0] eval:\n%s" % str(input_data[0]))

        # if options.max_records and options.max_records > 0:
        #     logging.info("max_records:%s" % options.max_records)
        #     input_data = input_data[:options.max_records]
        logging.info(input_data[0])
        logging.info("Items to process:%s" % len(input_data))

        input_x_features = []
        input_x_features_sparse = []
        input_y = []

        id2docid = []
        for i in range(len(input_data)):
            # feat_vecs, feats_sparse
            feat_vecs = extract_features_as_vector_from_single_record_v2_jointendings(input_data[i],
                                                                                      self._embeddings,
                                                                                      self._embeddings_vocab,
                                                                                      return_sparse_feats=False,
                                                                                      lower_tokens=True)
            y = input_data[i]["right_end_id"]

            input_x_features.append(feat_vecs)
            # input_x_features_sparse.append(feats_sparse)
            input_y.append(y)
            id2docid.append(input_data[i]["id"])

        logging.info("Evaluation instances: %s" % len(input_x_features))

        if scale_features:
            scaler = pickle.load(open(scale_file, 'rb'))
            logger.info('Scaling is enabled!')

            logging.info("Scaling...")
            start = time.time()
            input_x_features = scaler.transform(input_x_features)
            end = time.time()
            logging.info("Done in %s s" % (end - start))
        else:
            logger.info('NO scaling!')

        classifier_current = pickle.load(open(model_file, 'rb'))
        logging.info('Classifier:\n%s' % classifier_current)

        start = time.time()
        logging.info('Predicting with %s items...' % len(input_x_features))
        predicted_y = classifier_current.predict(input_x_features)
        end = time.time()
        logging.info("Done in %s s" % (end - start))

        logging.info("Confusion matrix:")

        from sklearn.metrics.classification import confusion_matrix
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        conf_matrix = confusion_matrix(input_y, predicted_y)
        logging.info("\n" + str(conf_matrix))
        logging.info("precision_recall_fscore_support:%s" % str(precision_recall_fscore_support(input_y, predicted_y)))

        logging.info("accuracy_score:%s" % accuracy_score(input_y, predicted_y))

        if options.submission_data_eval:
            logging.info("Building submission file [%s]" % options.submission_data_eval)
            fw = codecs.open(options.submission_data_eval, "w", encoding='utf-8')
            fw.write("InputStoryid,AnswerRightEnding\n")
            for i in range(len(predicted_y)):
                fw.write("%s,%s\n" % (id2docid[i], predicted_y[i] + 1))
            logging.info("Done! %s items written!" % len(predicted_y))


from optparse import OptionParser

# Sample run
# Set logging info
logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Enable file logging
# logFileName = '%s/%s-%s.log' % ('logs', 'sup_parser_v1', '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
# fileHandler = logging.FileHandler(logFileName, 'wb')
# fileHandler.setFormatter(logFormatter)
# logger.addHandler(fileHandler)

# Enable console logging
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

if __name__ == "__main__":
    # train
    # test
    # word2vec

    parser = OptionParser()
    parser.add_option("--cmd", dest="cmd", choices=["train", "eval", "train-eval"], help="command - train, eval")

    # parser.add_option("-", dest="", default="", help="")
    parser.add_option("--run_name", dest="run_name", help="")
    parser.add_option("--tune", dest="tune", help="tune the classifier")
    parser.add_option("--input_data_train", dest="input_data_train", help="input dataset")
    parser.add_option("--input_data_eval", dest="input_data_eval", help="input dataset")
    parser.add_option("--scale_features", dest="scale_features", default=True, help="scale features")
    parser.add_option("--emb_model_type", dest="emb_model_type", choices=["w2v", "dep", "rand"], help="")  # "w2v"  #
    parser.add_option("--emb_model_file", dest="emb_model_file", help="")
    parser.add_option("--word2vec_load_bin", dest="word2vec_load_bin", default=False, help="")
    parser.add_option("--output_dir", dest="output_dir", help="")
    parser.add_option("--model_file", dest="model_file", help="")
    parser.add_option("--scale_file", dest="scale_file", help="")
    parser.add_option("--output_file", dest="output_file", help="")
    parser.add_option("--max_records", dest="max_records", help="", type="int")
    parser.add_option("--submission_data_eval", dest="submission_data_eval", help="")

    parser.add_option("--lstm_hidden_size", dest="lstm_hidden_size", help="", type="int")
    parser.add_option("--lstm_layers", dest="lstm_layers", help="", type="int")
    parser.add_option("--batch_size", dest="batch_size", help="", type="int")
    parser.add_option("--learn_rate", dest="learn_rate", help="", type="float")
    parser.add_option("--min_word_freq", dest="min_word_freq", help="", type="int", default=5)
    parser.add_option("--num_epochs", dest="num_epochs", help="", type="int")

    parser.add_option("--dropout_prob_keep", dest="dropout_prob_keep", default=0.5, help="", type="float")
    parser.add_option("--layer_out_noise", dest="layer_out_noise", default=0.00, help="", type="float")
    parser.add_option("--loss_l2_beta", dest="loss_l2_beta", default=0.00, help="", type="float")

    parser.add_option("--train_embeddings", dest="train_embeddings", default=True, help="")
    parser.add_option("--short_embeddings", dest="short_embeddings", default=False, help="")

    (options, args) = parser.parse_args()

    # print options
    options_print = ""
    logging.info("Options:")
    for k, v in options.__dict__.iteritems():
        options_print += "opt: %s=%s\r\n" % (k, v)
    logging.info(options_print)

    if options.emb_model_type == "w2v":
        logging.info("Loading w2v model..")
        if options.word2vec_load_bin and options.word2vec_load_bin == "True":
            embeddings_model = Word2Vec.load_word2vec_format(options.emb_model_file,
                                                             binary=True)  # use this for google vectors
        else:
            embeddings_model = Word2Vec.load(options.emb_model_file)
        embeddings_vec_size = embeddings_model.syn0.shape[1]
    else:
        raise Exception("embeddings_model_type=%s is not yet supported!" % options.emb_model_type)

    classifier_name = "ClozeSolver_v2_lstm_v1"
    run_name = options.run_name
    model_dir = "saved_models/%s" % run_name

    magic_box = StoryCloze_v2_lstm_v1(output_dir=options.output_dir,
                                      embeddings=embeddings_model,
                                      embeddings_vocab=set(embeddings_model.index2word),
                                      checkpoint_dir="checkpoint_dir",
                                      checkpoint_prefix="model_tf_",
                                      checkpoint_best="model_tf_best",
                                      classifier_name=classifier_name,
                                      run_name=run_name,
                                      model_dir=model_dir
                                      )

    if not options.input_data_train and options.cmd in ["train", "train-eval"]:
        parser.error("input_data_train is not specified")

    if not options.input_data_eval and options.cmd in ["eval", "train-eval"]:
        parser.error("input_data_eval is not specified")

    if options.cmd == "train-eval":

        logging.info("------TRAINING--------")
        magic_box.train(input_dataset=options.input_data_train,
                        options=options,
                        embeddings=embeddings_model,
                        embeddings_vocab=embeddings_model.index2word)
        # logging.info("------EVALUATION--------")
        # magic_box.eval(input_dataset=options.input_data_eval,
        #                options=options)
    elif options.cmd == "train":
        logging.info("------TRAINING--------")
        magic_box.train(input_dataset=options.input_data_train,
                        options=options)
    elif options.cmd == "eval":
        logging.info("------EVALUATION--------")
        magic_box.eval(input_dataset=options.input_data_eval,
                       options=options)
    else:
        raise Exception("cmd=%s is not supported!" % options.cmd)

    pass
