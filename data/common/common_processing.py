import logging
import os
import pickle
import shutil
import time
import time as ti

import numpy as np

from data.CBTest.DataUtilities_CBT import DataUtilities_CBT
from data.SQuAD.DataUtilities_SQUAD import DataUtilities_SQUAD
from utils.embedding_utilities import VocabEmbeddingUtilities


def prepare_vocab_and_embeddings(vocab_dict,
                                 vocab_and_embeddings_file_save,
                                 embeddings_type,
                                 embeddings_model,
                                 embeddings_vec_size,
                                 char_vocab_dict=None):
    logging.info("Prepare embeddings...")

    vocab_and_embeddings = {}
    if embeddings_type == "w2v":
        st = time.time()

        logging.info("Done in %s s" % (ti.time() - st))

        logging.info("Loading embeddings for vocab..")
        st = ti.time()

        vocab_and_embeddings = VocabEmbeddingUtilities \
            .get_embeddings_for_vocab_from_model(vocabulary=vocab_dict,
                                                 embeddings_type='w2v',
                                                 embeddings_model=embeddings_model,
                                                 embeddings_size=embeddings_vec_size)

        # these tokens are randomly initilized now
        # vocab_and_embeddings["embeddings"][vocab_and_embeddings["vocabulary"][unknown_word], :] = unknown_vec
        # vocab_and_embeddings["embeddings"][vocab_and_embeddings["vocabulary"][pad_word], :] = pad_vec

        logging.info("Done in %s s" % (ti.time() - st))

    elif embeddings_type == "rand":
        random_embeddings = np.random.uniform(-0.1, 0.1, (len(vocab_dict), embeddings_vec_size))
        vocab_dict_emb = dict([(k, i) for i, k in enumerate(vocab_dict)])

        vocab_and_embeddings["embeddings"] = random_embeddings
        vocab_and_embeddings["vocabulary"] = vocab_dict_emb
    else:
        raise Exception("embeddings_type=%s is not supported!" % embeddings_type)

    if char_vocab_dict is not None:
        vocab_and_embeddings["vocabulary_char"] = char_vocab_dict

    # save vocab and embeddings
    pickle.dump(vocab_and_embeddings, open(vocab_and_embeddings_file_save, 'wb'))
    pickle.dump(vocab_and_embeddings["vocabulary"], open(vocab_and_embeddings_file_save+".vocab_only.pickle", 'wb'))
    if char_vocab_dict is not None:
        pickle.dump(vocab_and_embeddings["vocabulary_char"], open(vocab_and_embeddings_file_save + ".char_vocab_only.pickle", 'wb'))

    logging.info('Vocab and embeddings saved to: %s' % vocab_and_embeddings_file_save)