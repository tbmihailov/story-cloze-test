

import codecs
import sys

import logging  # word2vec logging

import numpy as np
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

from utils.common_utilities import CommonUtilities

from gensim.models.word2vec import Word2Vec  # used for word2vec

import time  # used for performance measuring
import math

from scipy import spatial  # used for similarity calculation

# from sklearn.svm import libsvm
from utils.embedding_vector_utilities import AverageVectorsUtilities

import pickle

from utils.similarity_feature_extraction import Similarity_FeatureExtraction

from data.StoryClozeTest.DataUtilities_ROCStories import DataUtilities_ROCStories

from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from optparse import OptionParser


def extract_features_as_vector_from_single_record_v1_jointendings(data_item,
                                                               word2vec_model,
                                                               word2vec_index2word_set,
                                                               return_sparse_feats=False,
                                                               lower_tokens=True,
                                                               include_embeddings=True,
                                                               include_sentlast2=True,
                                                               include_sentprev=True):
    ''''
        Sum of all sentences - embeddings
        Last sentence embeddings
        Last sent - Ending 1 similarities
        Last sent - Ending 2 similarities
        All sent - Ending 1 similarities
        All sent - Ending 2 similarities
    '''

    features = []
    sparse_feats_dict = {}

    w2v_num_feats = len(word2vec_model.wv.syn0[0] if embeddings_model.wv else embeddings_model.syn0.shape[0])

    # FEATURE EXTRACTION HERE
    # doc_id = data_item['id']
    # print doc_id


    # Ending 0
    end0_tokens = zip([x.lower() if lower_tokens else x for x in data_item["endings"][0]["tokens"]],
                      [x.lower() if lower_tokens else x for x in data_item["endings"][0]["pos"]])
    end0_words = [x[0] for x in end0_tokens]
    # print 'arg2: %s' % end0_words
    end0_embedding = AverageVectorsUtilities.makeFeatureVec(end0_words, word2vec_model, w2v_num_feats,
                                                            word2vec_index2word_set)
    if include_embeddings:
        features.extend(end0_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, end0_embedding, 'W2V_End0_')

    # Ending 1
    end1_tokens = zip([x.lower() if lower_tokens else x for x in data_item["endings"][0]["tokens"]],
                      [x.lower() if lower_tokens else x for x in data_item["endings"][0]["pos"]])
    end1_words = [x[0] for x in end1_tokens]
    # print 'arg2: %s' % end1_words
    end1_embedding = AverageVectorsUtilities.makeFeatureVec(end1_words, word2vec_model, w2v_num_feats,
                                                            word2vec_index2word_set)
    if include_embeddings:
        features.extend(end1_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, end1_embedding, 'W2V_End1_')

    # Sentence last
    sentlast_tokens = zip([x.lower() if lower_tokens else x for x in data_item["sentences"][-1]["tokens"]],
                          [x.lower() if lower_tokens else x for x in data_item["sentences"][-1]["pos"]])
    sentlast_words = [x[0] for x in sentlast_tokens]

    # print 'arg1: %s' % sentlast_words
    sentlast_embedding = AverageVectorsUtilities.makeFeatureVec(sentlast_words, word2vec_model, w2v_num_feats,
                                                                word2vec_index2word_set)
    if include_embeddings:
        features.extend(sentlast_embedding)
        vec_feats = {}
        CommonUtilities.append_features_with_vectors(vec_feats, sentlast_embedding, 'W2V_sentlast_')


    if include_sentlast2:
        # Sentence last
        sentlast2_tokens = zip([x.lower() if lower_tokens else x for x in data_item["sentences"][-2]["tokens"]],
                              [x.lower() if lower_tokens else x for x in data_item["sentences"][-2]["pos"]])
        sentlast2_words = [x[0] for x in sentlast2_tokens]

        # print 'arg1: %s' % sentlast2_words
        sentlast2_embedding = AverageVectorsUtilities.makeFeatureVec(sentlast2_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)
        if include_embeddings:
            features.extend(sentlast2_embedding)
            vec_feats = {}
            CommonUtilities.append_features_with_vectors(vec_feats, sentlast2_embedding, 'W2V_sentlast2_')

    if include_sentprev:
        # Sentence prev
        sentprev_tkns = []
        sentprev_pos = []
        for sent_id in range(0, len(data_item["sentences"])-1):
            sentprev_tkns.extend(data_item["sentences"][sent_id]["tokens"])
            sentprev_pos.extend(data_item["sentences"][sent_id]["pos"])

        sentprev_tokens = zip([x.lower() if lower_tokens else x for x in sentprev_tkns],
                              [x.lower() if lower_tokens else x for x in sentprev_pos])

        sentprev_words = [x[0] for x in sentprev_tokens]

        # print 'arg1: %s' % sentprev_words
        sentprev_embedding = AverageVectorsUtilities.makeFeatureVec(sentprev_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)
        if include_embeddings:
            features.extend(sentprev_embedding)
            vec_feats = {}
            CommonUtilities.append_features_with_vectors(vec_feats, sentprev_embedding, 'W2V_sentprev_')



    # Last sent to End1 cosine similarity
    # Last sent to End1 cosine similarity
    feat_key = "sentlast_end0_sim"
    arg1arg2_similarity_end0 = 0.00
    if len(sentlast_words) > 0 and len(end0_words) > 0:
        arg1arg2_similarity_end0 = spatial.distance.cosine(sentlast_embedding, end0_embedding)
    features.append(arg1arg2_similarity_end0)
    CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0)

    # Last sent to End1 cosine similarity
    feat_key = "sentlast_end1_sim"
    arg1arg2_similarity_end1 = 0.00
    if len(sentlast_words) > 0 and len(end1_words) > 0:
        arg1arg2_similarity_end1 = spatial.distance.cosine(sentlast_embedding, end1_embedding)
    features.append(arg1arg2_similarity_end1)
    CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end1)

    # Last sent to End1 cosine similarity
    feat_key = "sentlast_end1_subs_end2_sim"
    arg1arg2_similarity_end0end1substr = arg1arg2_similarity_end0 - arg1arg2_similarity_end1
    features.append(arg1arg2_similarity_end0end1substr)
    CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0end1substr)

    if include_sentlast2:
        # Last sent to End1 cosine similarity
        feat_key = "sentlast2_end0_sim"
        arg1arg2_similarity_end0 = 0.00
        if len(sentlast2_words) > 0 and len(end0_words) > 0:
            arg1arg2_similarity_end0 = spatial.distance.cosine(sentlast2_embedding, end0_embedding)
        features.append(arg1arg2_similarity_end0)
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0)

        # Last sent to End1 cosine similarity
        feat_key = "sentlast2_end1_sim"
        arg1arg2_similarity_end1 = 0.00
        if len(sentlast2_words) > 0 and len(end1_words) > 0:
            arg1arg2_similarity_end1 = spatial.distance.cosine(sentlast2_embedding, end1_embedding)
        features.append(arg1arg2_similarity_end1)
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end1)

        # Last sent to End1 cosine similarity
        feat_key = "sentlast2_end1_subs_end2_sim"
        arg1arg2_similarity_end0end1substr = arg1arg2_similarity_end0 - arg1arg2_similarity_end1
        features.append(arg1arg2_similarity_end0end1substr)
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0end1substr)



    if include_sentprev:
        feat_key = "sentprev_end0_sim"
        arg1arg2_similarity_end0 = 0.00
        if len(sentprev_words) > 0 and len(end0_words) > 0:
            arg1arg2_similarity_end0 = spatial.distance.cosine(sentprev_embedding, end0_embedding)
        features.append(arg1arg2_similarity_end0)
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0)

        # Last sent to End1 cosine similarity
        feat_key = "sentprev_end1_sim"
        arg1arg2_similarity_end1 = 0.00
        if len(sentprev_words) > 0 and len(end1_words) > 0:
            arg1arg2_similarity_end1 = spatial.distance.cosine(sentprev_embedding, end1_embedding)
        features.append(arg1arg2_similarity_end1)
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end1)

        # Last sent to End1 cosine similarity
        feat_key = "sentprev_end1_subs_end2_sim"
        arg1arg2_similarity_end0end1substr = arg1arg2_similarity_end0 - arg1arg2_similarity_end1
        features.append(arg1arg2_similarity_end0end1substr)
        CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0end1substr)

    # SIMILARITIES
    pref = "sentlast_end0_"

    arg1_tokens = sentlast_tokens
    arg2_tokens = end0_tokens

    words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
    words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]

    # Maximized similarities
    maxsims_feats_vec, maxsims_feats_sparse = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
        words1=words1, words2=words2,
        word2vec_model=word2vec_model,
        word2vec_index2word_set=word2vec_index2word_set,
        w2v_num_feats=w2v_num_feats,
        pref=pref)
    features.extend(maxsims_feats_vec)
    sparse_feats_dict.update(maxsims_feats_sparse)

    # POS tags similarities
    postag_feats_vec, postag_feats_sparse = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
        tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
        model=word2vec_model, word2vec_num_features=w2v_num_feats,
        word2vec_index2word_set=word2vec_index2word_set)
    features.extend(postag_feats_vec)
    sparse_feats_dict.update(postag_feats_sparse)

    # SIMILARITIES
    pref = "sentlast_end1_"

    arg1_tokens = sentlast_tokens
    arg2_tokens = end1_tokens

    words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
    words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]

    # Maximized similarities
    maxsims_feats_vec, maxsims_feats_sparse = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
        words1=words1, words2=words2,
        word2vec_model=word2vec_model,
        word2vec_index2word_set=word2vec_index2word_set,
        w2v_num_feats=w2v_num_feats,
        pref=pref)
    features.extend(maxsims_feats_vec)
    sparse_feats_dict.update(maxsims_feats_sparse)

    # POS tags similarities
    postag_feats_vec, postag_feats_sparse = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
        tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
        model=word2vec_model, word2vec_num_features=w2v_num_feats,
        word2vec_index2word_set=word2vec_index2word_set)
    features.extend(postag_feats_vec)
    # logging.info(postag_feats_sparse) # debug
    sparse_feats_dict.update(postag_feats_sparse)

    if include_sentlast2:
        # SIMILARITIES
        pref = "sentlast2_end0_"

        arg1_tokens = sentlast2_tokens
        arg2_tokens = end0_tokens

        words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
        words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]

        # Maximized similarities
        maxsims_feats_vec, maxsims_feats_sparse = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
            words1=words1, words2=words2,
            word2vec_model=word2vec_model,
            word2vec_index2word_set=word2vec_index2word_set,
            w2v_num_feats=w2v_num_feats,
            pref=pref)
        features.extend(maxsims_feats_vec)
        sparse_feats_dict.update(maxsims_feats_sparse)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)
        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

        # SIMILARITIES
        pref = "sentlast2_end1_"

        arg1_tokens = sentlast2_tokens
        arg2_tokens = end1_tokens

        words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
        words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]

        # Maximized similarities
        maxsims_feats_vec, maxsims_feats_sparse = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
            words1=words1, words2=words2,
            word2vec_model=word2vec_model,
            word2vec_index2word_set=word2vec_index2word_set,
            w2v_num_feats=w2v_num_feats,
            pref=pref)
        features.extend(maxsims_feats_vec)
        sparse_feats_dict.update(maxsims_feats_sparse)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)
        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

    if include_sentprev:
        # SIMILARITIES
        pref = "sentprev_end0_"

        arg1_tokens = sentprev_tokens
        arg2_tokens = end0_tokens

        words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
        words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]

        # Maximized similarities
        maxsims_feats_vec, maxsims_feats_sparse = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
            words1=words1, words2=words2,
            word2vec_model=word2vec_model,
            word2vec_index2word_set=word2vec_index2word_set,
            w2v_num_feats=w2v_num_feats,
            pref=pref)
        features.extend(maxsims_feats_vec)
        sparse_feats_dict.update(maxsims_feats_sparse)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)
        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

        # SIMILARITIES
        pref = "sentprev_end1_"

        arg1_tokens = sentprev_tokens
        arg2_tokens = end1_tokens

        words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
        words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]

        # Maximized similarities
        maxsims_feats_vec, maxsims_feats_sparse = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
            words1=words1, words2=words2,
            word2vec_model=word2vec_model,
            word2vec_index2word_set=word2vec_index2word_set,
            w2v_num_feats=w2v_num_feats,
            pref=pref)
        features.extend(maxsims_feats_vec)
        sparse_feats_dict.update(maxsims_feats_sparse)

        # POS tags similarities
        postag_feats_vec, postag_feats_sparse = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
            tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
            model=word2vec_model, word2vec_num_features=w2v_num_feats,
            word2vec_index2word_set=word2vec_index2word_set)
        features.extend(postag_feats_vec)
        sparse_feats_dict.update(postag_feats_sparse)

    # Fix Nan features
    for i in range(0, len(features)):
        if math.isnan(features[i]):
            features[i] = 0.00

    if return_sparse_feats:
        return features, sparse_feats_dict
    else:
        return features

from nltk.corpus import stopwords
def extract_features_as_vector_from_single_record_v2_jointendings(data_item,
                                                               word2vec_model,
                                                               word2vec_index2word_set,
                                                               return_sparse_feats=False,
                                                               lower_tokens=True,
                                                               remove_stopwords=True,
                                                               include_embeddings=True,
                                                               include_sentlast=True,
                                                               include_sentlast2=True,
                                                               include_sentlast3=True,
                                                               include_sentlast4=True,
                                                               include_sentprev=True,
                                                               include_maxsim=True,
                                                               include_possim=True,
                                                               include_fullsims=True,
                                                               include_elem_multiply=True):
    ''''
        Sum of all sentences - embeddings
        Last sentence embeddings
        Last sent - Ending 1 similarities
        Last sent - Ending 2 similarities
        All sent - Ending 1 similarities
        All sent - Ending 2 similarities
    '''

    features = []
    sparse_feats_dict = {}

    w2v_num_feats = len(word2vec_model.wv.syn0[0] if embeddings_model.wv else embeddings_model.syn0.shape[0])

    stop_words = []
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))


    # Sentence prev
    sentprev_tkns = []
    sentprev_pos = []
    for sent_id in range(0, len(data_item["sentences"])-1):
        sentprev_tkns.extend(data_item["sentences"][sent_id]["tokens"])
        sentprev_pos.extend(data_item["sentences"][sent_id]["pos"])

    sentprev_tokens = zip([x.lower() if lower_tokens else x for x in sentprev_tkns],
                          [x.lower() if lower_tokens else x for x in sentprev_pos])

    sentprev_words = [x[0] for x in sentprev_tokens if remove_stopwords and x not in stop_words]

    # print 'arg1: %s' % sentprev_words
    sentprev_embedding = AverageVectorsUtilities.makeFeatureVec(sentprev_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)

    # FEATURE EXTRACTION HERE
    # doc_id = data_item['id']
    # print doc_id

    # Ending 0
    end0_tokens = zip([x.lower() if lower_tokens else x for x in data_item["endings"][0]["tokens"]],
                      [x.lower() if lower_tokens else x for x in data_item["endings"][0]["pos"]])
    end0_words = [x[0] for x in end0_tokens if remove_stopwords and x not in stop_words]
    # print 'arg2: %s' % end0_words
    end0_embedding = AverageVectorsUtilities.makeFeatureVec(end0_words, word2vec_model, w2v_num_feats,
                                                            word2vec_index2word_set)

    # Ending 1
    end1_tokens = zip([x.lower() if lower_tokens else x for x in data_item["endings"][1]["tokens"]],
                      [x.lower() if lower_tokens else x for x in data_item["endings"][1]["pos"]])
    end1_words = [x[0] for x in end1_tokens if remove_stopwords and x not in stop_words]
    # print 'arg2: %s' % end1_words
    end1_embedding = AverageVectorsUtilities.makeFeatureVec(end1_words, word2vec_model, w2v_num_feats,
                                                            word2vec_index2word_set)
    story_ending_diff = False

    vec_feats = {}
    if include_embeddings:
        end0_emb_curr = end0_embedding
        end1_emb_curr = end1_embedding
        if story_ending_diff:
            end0_emb_curr = np.asarray(end0_emb_curr) - np.asarray(sentprev_embedding)
            end1_emb_curr = np.asarray(end1_emb_curr) - np.asarray(sentprev_embedding)

        features.extend(end0_emb_curr)

        CommonUtilities.append_features_with_vectors(vec_feats, end0_emb_curr, 'W2V_End0_')
        features.extend(end1_emb_curr)
        CommonUtilities.append_features_with_vectors(vec_feats, end1_emb_curr, 'W2V_End1_')

        if include_elem_multiply:
            story_end0_mul_curr = np.multiply(np.asarray(end0_emb_curr), np.asarray(sentprev_embedding))
            story_end1_mulcurr = np.multiply(np.asarray(end1_emb_curr), np.asarray(sentprev_embedding))

            features.extend(end0_emb_curr)
            CommonUtilities.append_features_with_vectors(vec_feats, story_end0_mul_curr, 'W2V_SE0_mul_')

            features.extend(end1_emb_curr)
            CommonUtilities.append_features_with_vectors(vec_feats, story_end1_mulcurr, 'W2V_SE1_mul_')


    # Sentence last
    if include_sentlast:
        sentlast_tokens = zip([x.lower() if lower_tokens else x for x in data_item["sentences"][-1]["tokens"]],
                              [x.lower() if lower_tokens else x for x in data_item["sentences"][-1]["pos"]])
        sentlast_words = [x[0] for x in sentlast_tokens if remove_stopwords and x not in stop_words]

        # print 'arg1: %s' % sentlast_words
        sentlast_embedding = AverageVectorsUtilities.makeFeatureVec(sentlast_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)

    if include_sentlast2:
        # Sentence last
        sentlast2_tokens = zip([x.lower() if lower_tokens else x for x in data_item["sentences"][-2]["tokens"]],
                              [x.lower() if lower_tokens else x for x in data_item["sentences"][-2]["pos"]])
        sentlast2_words = [x[0] for x in sentlast2_tokens if remove_stopwords and x not in stop_words]

        # print 'arg1: %s' % sentlast2_words
        sentlast2_embedding = AverageVectorsUtilities.makeFeatureVec(sentlast2_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)

    if include_sentlast3:
        # Sentence last
        sentlast3_tokens = zip([x.lower() if lower_tokens else x for x in data_item["sentences"][-3]["tokens"]],
                              [x.lower() if lower_tokens else x for x in data_item["sentences"][-3]["pos"]])
        sentlast3_words = [x[0] for x in sentlast3_tokens if remove_stopwords and x not in stop_words]

        # print 'arg1: %s' % sentlast2_words
        sentlast3_embedding = AverageVectorsUtilities.makeFeatureVec(sentlast3_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)

    if include_sentlast4:
        # Sentence last
        sentlast4_tokens = zip([x.lower() if lower_tokens else x for x in data_item["sentences"][-4]["tokens"]],
                              [x.lower() if lower_tokens else x for x in data_item["sentences"][-4]["pos"]])
        sentlast4_words = [x[0] for x in sentlast4_tokens if remove_stopwords and x not in stop_words]

        # print 'arg1: %s' % sentlast2_words
        sentlast4_embedding = AverageVectorsUtilities.makeFeatureVec(sentlast4_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)
        
    if include_sentprev:
        # Sentence prev
        sentprev_tkns = []
        sentprev_pos = []
        for sent_id in range(0, len(data_item["sentences"])-1):
            sentprev_tkns.extend(data_item["sentences"][sent_id]["tokens"])
            sentprev_pos.extend(data_item["sentences"][sent_id]["pos"])

        sentprev_tokens = zip([x.lower() if lower_tokens else x for x in sentprev_tkns],
                              [x.lower() if lower_tokens else x for x in sentprev_pos])

        sentprev_words = [x[0] for x in sentprev_tokens if remove_stopwords and x not in stop_words]

        # print 'arg1: %s' % sentprev_words
        sentprev_embedding = AverageVectorsUtilities.makeFeatureVec(sentprev_words, word2vec_model, w2v_num_feats,
                                                                    word2vec_index2word_set)

    if include_embeddings:
        if include_sentlast:
            features.extend(sentlast_embedding)
            # vec_feats = {}
            # CommonUtilities.append_features_with_vectors(vec_feats, sentlast_embedding, 'W2V_sentlast_')

        if include_sentlast2:
            features.extend(sentlast2_embedding)
            # vec_feats = {}
            # CommonUtilities.append_features_with_vectors(vec_feats, sentlast2_embedding, 'W2V_sentlast2_')

        if include_sentlast3:
            features.extend(sentlast3_embedding)
            # vec_feats = {}
            # CommonUtilities.append_features_with_vectors(vec_feats, sentlast3_embedding, 'W2V_sentlast3_')

        if include_sentlast4:
            features.extend(sentlast4_embedding)
            # vec_feats = {}
            # CommonUtilities.append_features_with_vectors(vec_feats, sentlast4_embedding, 'W2V_sentlast4_')

        if include_sentprev:
            features.extend(sentprev_embedding)
            # vec_feats = {}
            # CommonUtilities.append_features_with_vectors(vec_feats, sentprev_embedding, 'W2V_sentprev_')


    def gen_and_add_features_for_sentence(featpref,
                                  sentcurr_tokens, sentcurr_embedding,
                                  end0_tokens, end0_embedding,
                                  end1_tokens, end1_embedding):
        

        # Last sent to End1 cosine similarity
        # Last sent to End1 cosine similarity
        if include_fullsims:
            feat_key = featpref + "end0_sim"
            arg1arg2_similarity_end0 = 0.00
            if not math.isnan(sentcurr_embedding[0]) and not math.isnan(end0_embedding[0]):
                arg1arg2_similarity_end0 = spatial.distance.cosine(sentcurr_embedding, end0_embedding)
            features.append(arg1arg2_similarity_end0)
            # CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0)

            # Last sent to End1 cosine similarity
            feat_key = featpref + "end1_sim"
            arg1arg2_similarity_end1 = 0.00
            if not math.isnan(sentcurr_embedding[0]) and not math.isnan(end1_embedding[0]):
                arg1arg2_similarity_end1 = spatial.distance.cosine(sentcurr_embedding, end1_embedding)
            features.append(arg1arg2_similarity_end1)
            # CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end1)
    
        # # Last sent to End1 cosine similarity
        # feat_key = featpref + "end1_subs_end2_sim"
        # arg1arg2_similarity_end0end1substr = arg1arg2_similarity_end0 - arg1arg2_similarity_end1
        # features.append(arg1arg2_similarity_end0end1substr)
        # CommonUtilities.increment_feat_val(sparse_feats_dict, feat_key, arg1arg2_similarity_end0end1substr)

    
        # SIMILARITIES
        pref = featpref + "end0_"
    
        arg1_tokens = sentcurr_tokens
        arg2_tokens = end0_tokens
    
        words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
        words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]
    
        # Maximized similarities
        if include_maxsim:
            maxsims_feats_vec_end0, maxsims_feats_sparse_end0 = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
                words1=words1, words2=words2,
                word2vec_model=word2vec_model,
                word2vec_index2word_set=word2vec_index2word_set,
                w2v_num_feats=w2v_num_feats,
                pref=pref)
            features.extend(maxsims_feats_vec_end0)
            # sparse_feats_dict.update(maxsims_feats_sparse_end0)
    
        # POS tags similarities
        if include_possim:
            postag_feats_vec_end0, postag_feats_sparse_end0 = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
                tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
                model=word2vec_model, word2vec_num_features=w2v_num_feats,
                word2vec_index2word_set=word2vec_index2word_set)
            features.extend(postag_feats_vec_end0)
            # sparse_feats_dict.update(postag_feats_sparse_end0)

        # SIMILARITIES
        pref = featpref + "end1_"
    
        arg1_tokens = sentcurr_tokens
        arg2_tokens = end1_tokens

        words1 = [x[0] for x in arg1_tokens if x[0] in word2vec_index2word_set]
        words2 = [x[0] for x in arg2_tokens if x[0] in word2vec_index2word_set]
    
        # Maximized similarities
        if include_maxsim:
            maxsims_feats_vec_end1, maxsims_feats_sparse_end1 = Similarity_FeatureExtraction.get_maxsims_sim_fetures(
                words1=words1, words2=words2,
                word2vec_model=word2vec_model,
                word2vec_index2word_set=word2vec_index2word_set,
                w2v_num_feats=w2v_num_feats,
                pref=pref)
            features.extend(maxsims_feats_vec_end1)
            # sparse_feats_dict.update(maxsims_feats_sparse_end1)

        # POS tags similarities
        if include_possim:
            postag_feats_vec_end1, postag_feats_sparse_end1 = Similarity_FeatureExtraction.get_postagged_sim_fetures_experiments(
                tokens_data_text1=arg1_tokens, tokens_data_text2=arg2_tokens,
                model=word2vec_model, word2vec_num_features=w2v_num_feats,
                word2vec_index2word_set=word2vec_index2word_set)
            features.extend(postag_feats_vec_end1)
            # logging.info(postag_feats_sparse) # debug
            # sparse_feats_dict.update(postag_feats_sparse_end1)


        # if include_possim:
        #     # print "end0:"+str(postag_feats_vec_end0) # debug
        #     # print "end1:" + str(postag_feats_vec_end1) # debug
        #     sim_feats = list(np.asarray(postag_feats_vec_end0) - np.asarray(postag_feats_vec_end1))
        #     # print sim_feats # debug
        #     features.extend(sim_feats)
        #     # sparse_feats_dict.update(postag_feats_sparse_end0)

    if include_sentlast:
        gen_and_add_features_for_sentence("sentlast_",
                                          sentlast_tokens, sentlast_embedding,
                                          end0_tokens, end0_embedding,
                                          end1_tokens, end1_embedding)

    if include_sentlast2:
        gen_and_add_features_for_sentence("sent2_",
                                          sentlast2_tokens, sentlast2_embedding,
                                          end0_tokens, end0_embedding,
                                          end1_tokens, end1_embedding)

    if include_sentlast3:
        gen_and_add_features_for_sentence("sent3_",
                                          sentlast3_tokens, sentlast3_embedding,
                                          end0_tokens, end0_embedding,
                                          end1_tokens, end1_embedding)

    if include_sentlast4:
        gen_and_add_features_for_sentence("sent4_",
                                          sentlast4_tokens, sentlast4_embedding,
                                          end0_tokens, end0_embedding,
                                          end1_tokens, end1_embedding)
    if include_sentprev:
        gen_and_add_features_for_sentence("sentprev_",
                                          sentprev_tokens, sentprev_embedding,
                                          end0_tokens, end0_embedding,
                                          end1_tokens, end1_embedding)

    # Fix Nan features
    for i in range(0, len(features)):
        if math.isnan(features[i]):
            features[i] = 0.00

    if return_sparse_feats:
        return features, sparse_feats_dict
    else:
        return features

class StoryCloze_Baseline_Similarity_v1(object):
    def __init__(self, output_dir, embeddings):
        self._output_dir = output_dir
        self._embeddings = embeddings
        self._embeddings_vocab = [set(x.wv.index2word) for x in embeddings]

        pass

    def train(self,
              input_dataset,
              options):
        scale_features = options.scale_features
        model_file = options.model_file
        scale_file = options.scale_file


        input_data = DataUtilities_ROCStories.load_data_from_json_file(input_dataset)
        logging.info("input_data[0] train:\n%s" % str(input_data[0]))
        if options.max_records and options.max_records>0:
            logging.info("max_records:%s" % options.max_records)
            input_data = input_data[:options.max_records]
        logging.info(input_data[0])
        logging.info("Items to process:%s" % len(input_data))

        input_x_features = []
        input_x_features_sparse = []
        input_y = []

        id2docid = []
        gen_swap_endings = True
        logging.info("Extracting features for %s items..." % len(input_data))

        input_data_feats = None
        if options.load_features == True:
            input_data_feats_file = input_dataset + ".neural_feat_repr.pickle"
            input_data_feats = pickle.load(open(input_data_feats_file, 'rb'))
            logging.info("Features loaded from %s" % input_data_feats_file)

        start = time.time()
        id_id = -1
        for i in range(len(input_data)):
            if ((i+1) % 100) == 0:
                logging.info("processed %s of  %s" % (i+1, len(input_data)))
            # feat_vecs, feats_sparse
            # normal item
            feat_vecs = []
            for emb_i in range(len(self._embeddings)):
                feat_vecs_curr = extract_features_as_vector_from_single_record_v2_jointendings(input_data[i],
                                                                           self._embeddings[emb_i],
                                                                           self._embeddings_vocab[emb_i],
                                                                           return_sparse_feats=False,
                                                                          lower_tokens=(
                                                                          options.lower_tokens == 'True'),
                                                                          remove_stopwords=(
                                                                          options.remove_stopwords == 'True'),
                                                                          include_embeddings=(
                                                                          options.include_embeddings == 'True'),
                                                                          include_sentlast=(
                                                                          options.include_sentlast == 'True'),
                                                                          include_sentlast2=(
                                                                          options.include_sentlast2 == 'True'),
                                                                          include_sentlast3=(
                                                                          options.include_sentlast3 == 'True'),
                                                                          include_sentlast4=(
                                                                          options.include_sentlast4 == 'True'),
                                                                          include_sentprev=(
                                                                          options.include_sentprev == 'True'),
                                                                          include_maxsim=(
                                                                          options.include_maxsim == 'True'),
                                                                          include_possim=(
                                                                          options.include_possim == 'True'),
                                                                          include_fullsims=(
                                                                          options.include_fullsims == 'True'),
                                                                          include_elem_multiply=(
                                                                          options.include_elem_multiply == 'True'),
                                                                          )
                feat_vecs.extend(feat_vecs_curr)

            #logging.info("feats type = %s" % str(type(feat_vecs)))
            id_id += 1
            if input_data_feats is not None:
                # logging.info("input_data_feats[i] type = %s" % str(type(input_data_feats[i].tolist())))
                feat_vecs.extend(input_data_feats[id_id].tolist())

            y = input_data[i]["right_end_id"]

            input_x_features.append(feat_vecs)
            # input_x_features_sparse.append(feats_sparse)
            input_y.append(y)

            id2docid.append(input_data[i]["id"])

            from copy import deepcopy
            if gen_swap_endings:
                id_id += 1
                curr_item = deepcopy(input_data[i])
                curr_item["endings"][0] = deepcopy(input_data[i]["endings"][1])
                curr_item["endings"][1] = deepcopy(input_data[i]["endings"][0])

                feat_vecs = []
                for emb_i in range(len(self._embeddings)):
                    feat_vecs_curr = extract_features_as_vector_from_single_record_v2_jointendings(curr_item,
                                                                                                   self._embeddings[
                                                                                                       emb_i],
                                                                                                   self._embeddings_vocab[
                                                                                                       emb_i],
                                                                                                   return_sparse_feats=False,
                                                                                                   lower_tokens=(
                                                                                                       options.lower_tokens == 'True'),
                                                                                                   remove_stopwords=(
                                                                                                       options.remove_stopwords == 'True'),
                                                                                                   include_embeddings=(
                                                                                                       options.include_embeddings == 'True'),
                                                                                                   include_sentlast=(
                                                                                                       options.include_sentlast == 'True'),
                                                                                                   include_sentlast2=(
                                                                                                       options.include_sentlast2 == 'True'),
                                                                                                   include_sentlast3=(
                                                                                                       options.include_sentlast3 == 'True'),
                                                                                                   include_sentlast4=(
                                                                                                       options.include_sentlast4 == 'True'),
                                                                                                   include_sentprev=(
                                                                                                       options.include_sentprev == 'True'),
                                                                                                   include_maxsim=(
                                                                                                       options.include_maxsim == 'True'),
                                                                                                   include_possim=(
                                                                                                       options.include_possim == 'True'),
                                                                                                   include_fullsims=(
                                                                                                       options.include_fullsims == 'True'),
                                                                                                   include_elem_multiply=(
                                                                                                       options.include_elem_multiply == 'True'),
                                                                                                   )
                    feat_vecs.extend(feat_vecs_curr)

                if input_data_feats is not None:
                    # logging.info("input_data_feats[i] type = %s" % str(type(input_data_feats[i].tolist())))
                    feat_vecs.extend(input_data_feats[id_id].tolist())

                y = input_data[i]["right_end_id"]

                input_x_features.append(feat_vecs)
                # input_x_features_sparse.append(feats_sparse)
                input_y.append(0 if y==1 else 1)

                id2docid.append(input_data[i]["id"])

        logging.info("Done in %s s" % (time.time() - start))
        logging.info("Features:%s" % (len(input_x_features[0])))
        logging.info("Feature vector 0:%s" % (str(input_x_features[0])))

        logging.info("Training instances: %s" % len(input_x_features))

        # logging.info('=====SCALING======')
        scale_range = [0, 1]
        scaler = preprocessing.MinMaxScaler(scale_range)
        if scale_features:
            logging.info('Scaling %s items with %s features..' % (len(input_x_features), len(input_x_features[0])))
            start = time.time()
            input_x_features = scaler.fit_transform(input_x_features)
            end = time.time()
            logging.info("Done in %s s" % (end - start))
            pickle.dump(scaler, open(scale_file, 'wb'))
            logging.info('Scale feats ranges saved to %s' % scale_file)
        else:
            logging.info("No scaling!")

        logging.info("Features:%s" % (len(input_x_features[0])))
        logging.info("Feature vector 0:%s" % (str(input_x_features[0])))
        # Training
        # Classifier params

        # classifier_current = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #                          degree=3, gamma='auto', kernel='rbf',
        #                          max_iter=-1, probability=False, random_state=None, shrinking=True,
        #                          tol=0.001, verbose=False)

        tune = False
        if options.tune and options.tune == "True":
            tune = True
        param_c = options.param_c

        if tune:
            param_grid = {'C': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 10]}
            # clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
            classifier_tune = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                                 intercept_scaling=1, class_weight=None, random_state=None,
                                                 solver='liblinear',
                                                 max_iter=300, multi_class='ovr', verbose=0, warm_start=False,
                                                 n_jobs=8)
            gcv = GridSearchCV(cv=None,
                               estimator=classifier_tune,
                               param_grid=param_grid)
            gcv.fit(input_x_features, input_y)

            logging.info("Estimated_best_params:%s" % gcv.best_params_)
            if 'C' in gcv.best_params_:
                param_c = gcv.best_params_['C']
                logging.info("best C=%s" % param_c)

        logging.info("curr C=%s" % param_c)


        classifier_current = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=param_c, fit_intercept=True,
                                                intercept_scaling=1, random_state=None,
                                                solver='liblinear',
                                                max_iter=300, multi_class='ovr', verbose=0, warm_start=False,
                                                n_jobs=8)

        # classifier_current = LogisticRegressionCV(penalty='l2', dual=False, tol=0.0001, fit_intercept=True,
        #                                         intercept_scaling=1, random_state=None,
        #                                         solver='liblinear',
        #                                         max_iter=300, multi_class='ovr', verbose=0,
        #                                         n_jobs=8)

        # from sklearn.ensemble import RandomForestClassifier
        # classifier_current = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

        logging.info('Classifier:\n%s' % classifier_current)

        start = time.time()
        logging.info('Training with %s items...' % len(input_x_features))
        classifier_current.fit(input_x_features, input_y)
        end = time.time()
        logging.info("Done in %s s" % (end - start))

        # Saving model
        pickle.dump(classifier_current, open(model_file, 'wb'))
        logging.info('Model saved to %s' % model_file)

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


        input_data_feats = None
        if options.load_features == True:
            input_data_feats_file = input_dataset + ".neural_feat_repr.pickle"

            input_data_feats = pickle.load(open(input_data_feats_file, 'rb'))
            logging.info("Features loaded from %s" % input_data_feats_file)

        id2docid = []

        input_x_features = []
        input_x_features_sparse = []
        input_y = []
        predicted_y = []

        # evaluation
        if scale_features:
            scaler = pickle.load(open(scale_file, 'rb'))
            logger.info('Scaling is enabled!')
        else:
            logger.info('NO scaling!')

        classifier_current = pickle.load(open(model_file, 'rb'))
        logging.info('Classifier:\n%s' % classifier_current)

        def eval_instances(input_x_features):
            logging.info("Evaluation instances: %s" % len(input_x_features))

            if scale_features:
                logging.info("Scaling...")
                start = time.time()
                input_x_features = scaler.transform(input_x_features)
                end = time.time()
                logging.info("Done in %s s" % (end - start))
            else:
                logger.info('NO scaling!')

            start = time.time()
            logging.info('Predicting with %s items...' % len(input_x_features))
            predicted_y_curr = classifier_current.predict(input_x_features)
            end = time.time()
            logging.info("Done in %s s" % (end - start))
            return predicted_y_curr

        eval_batch_size = 5000
        for i in range(len(input_data)):
            if ((i+1) % 1000) == 0:
                logging.info("processed %s of  %s" % (i+1, len(input_data)))
            # feat_vecs, feats_sparse
            feat_vecs = []
            for emb_i in range(len(self._embeddings)):
                feat_vecs_curr = extract_features_as_vector_from_single_record_v2_jointendings(input_data[i],
                                                                                               self._embeddings[emb_i],
                                                                                               self._embeddings_vocab[
                                                                                                   emb_i],
                                                                                               return_sparse_feats=False,
                                                                                               lower_tokens=(
                                                                                                   options.lower_tokens == 'True'),
                                                                                               remove_stopwords=(
                                                                                                   options.remove_stopwords == 'True'),
                                                                                               include_embeddings=(
                                                                                                   options.include_embeddings == 'True'),
                                                                                               include_sentlast=(
                                                                                                   options.include_sentlast == 'True'),
                                                                                               include_sentlast2=(
                                                                                                   options.include_sentlast2 == 'True'),
                                                                                               include_sentlast3=(
                                                                                                   options.include_sentlast3 == 'True'),
                                                                                               include_sentlast4=(
                                                                                                   options.include_sentlast4 == 'True'),
                                                                                               include_sentprev=(
                                                                                                   options.include_sentprev == 'True'),
                                                                                               include_maxsim=(
                                                                                                   options.include_maxsim == 'True'),
                                                                                               include_possim=(
                                                                                                   options.include_possim == 'True'),
                                                                                               include_fullsims=(
                                                                                                   options.include_fullsims == 'True'),
                                                                                               include_elem_multiply=(
                                                                                                   options.include_elem_multiply == 'True'),

                                                                                               )
                feat_vecs.extend(feat_vecs_curr)

            if input_data_feats is not None:
                feat_vecs.extend(input_data_feats[i])

            y = input_data[i]["right_end_id"]
            input_y.append(y)
            id2docid.append(input_data[i]["id"])

            input_x_features.append(feat_vecs)
            # input_x_features_sparse.append(feats_sparse)

            if ((i+1) % eval_batch_size) == 0:
                logging.info("processed %s of  %s" % (i+1, len(input_data)))

                predicted_y_curr = eval_instances(input_x_features)
                predicted_y.extend(predicted_y_curr)
                del input_x_features
                input_x_features = []

        if len(input_x_features) > 0:  # eval last batch
            predicted_y_curr = eval_instances(input_x_features)
            predicted_y.extend(predicted_y_curr)
            del input_x_features
            input_x_features = []

        logging.info("Confusion matrix:")

        conf_matrix = confusion_matrix(input_y, predicted_y)
        logging.info("\n" + str(conf_matrix))
        logging.info("precision_recall_fscore_support:%s" % str(precision_recall_fscore_support(input_y, predicted_y)))

        test_accuracy_score=accuracy_score(input_y, predicted_y)
        logging.info("accuracy_score:%s" % test_accuracy_score)

        if options.submission_data_eval:
            submission_file_name = options.submission_data_eval + "_acc_%s.txt" % test_accuracy_score
            logging.info("Building submission file [%s]" % submission_file_name)
            fw = codecs.open(submission_file_name, "w", encoding='utf-8')
            fw.write("InputStoryid,AnswerRightEnding\n")
            for i in range(len(predicted_y)):
                fw.write("%s,%s\n" % (id2docid[i], predicted_y[i]+1))
            logging.info("Done! %s items written!" % len(predicted_y))


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

    def callback_to_list(option, opt, value, parser):

        setattr(parser.values, option.dest, str(value).split(','))



    parser = OptionParser()


    parser = OptionParser()
    parser.add_option("--cmd", dest="cmd", choices=["train", "eval", "train-eval"], help="command - train, eval")

    # parser.add_option("-", dest="", default="", help="")
    parser.add_option("--run_name", dest="run_name", help="")
    parser.add_option("--tune", dest="tune", help="tune the classifier", default=True)
    parser.add_option("--param_c", dest="param_c", default=1.00, help="", type="float")
    parser.add_option("--input_data_train", dest="input_data_train", help="input dataset")
    parser.add_option("--input_data_eval", dest="input_data_eval", help="input dataset")
    parser.add_option("--scale_features", dest="scale_features", default=True, help="scale features")
    parser.add_option("--emb_model_type", dest="emb_model_type", choices=["w2v", "dep", "rand"], help="")  # "w2v"  #
    parser.add_option("--emb_model_file", dest="emb_model_file", help="")

    parser.add_option("--word2vec_load_bin", dest="word2vec_load_bin", default=False, help="")
    parser.add_option("--word2vec_load_format", dest="word2vec_load_format", default=True, help="")
    parser.add_option("--output_dir", dest="output_dir", help="")
    parser.add_option("--model_file", dest="model_file", help="")
    parser.add_option("--scale_file", dest="scale_file", help="")
    parser.add_option("--output_file", dest="output_file", help="")
    parser.add_option("--max_records", dest="max_records", help="", type="int")
    parser.add_option("--load_features", dest="load_features", default=False)
    parser.add_option("--submission_data_eval", dest="submission_data_eval", help="")

    # features
    parser.add_option('--lower_tokens', dest='lower_tokens', default=True)
    parser.add_option('--remove_stopwords', dest='remove_stopwords', default=True)
    parser.add_option('--inc_embedd_vectors', dest='include_embeddings', default=True)
    parser.add_option('--inc_slast1', dest='include_sentlast', default=True)
    parser.add_option('--inc_slast2', dest='include_sentlast2', default=True)
    parser.add_option('--inc_slast3', dest='include_sentlast3', default=True)
    parser.add_option('--inc_slast4', dest='include_sentlast4', default=True)
    parser.add_option('--inc_story', dest='include_sentprev', default=True)
    parser.add_option('--inc_maxsim', dest='include_maxsim', default=True)
    parser.add_option('--inc_possim', dest='include_possim', default=True)
    parser.add_option('--inc_fullsims', dest='include_fullsims', default=True)
    parser.add_option('--include_elem_multiply', dest='include_elem_multiply', default=False)


    (options, args) = parser.parse_args()

    # print options
    options_print = ""
    logging.info("Options:")
    for k, v in options.__dict__.iteritems():
        options_print += "opt: %s=%s\r\n" % (k, v)
    logging.info(options_print)


    embeddings_models = []
    embeddings_vec_sizes = []

    if options.emb_model_type == "w2v":
        logging.info("Loading w2v model..")

        for emb_model_file in options.emb_model_file.split(','):
            if options.word2vec_load_format and options.word2vec_load_format == "True":
                embeddings_model = Word2Vec.load_word2vec_format(emb_model_file,
                                                                 binary=options.word2vec_load_bin=="True")

            else:
                embeddings_model = Word2Vec.load(emb_model_file)

            embeddings_models.append(embeddings_model)
            embeddings_vec_size = embeddings_model.wv.syn0.shape[1] if embeddings_model.wv else embeddings_model.syn0.shape[1]
            embeddings_vec_sizes.append(embeddings_vec_size)
    else:
        raise Exception("embeddings_model_type=%s is not yet supported!" % options.emb_model_type)

    magic_box = StoryCloze_Baseline_Similarity_v1(output_dir=options.output_dir,
                                                  embeddings=embeddings_models)

    if not options.input_data_train and options.cmd in ["train", "train-eval"]:
        parser.error("input_data_train is not specified")

    if not options.input_data_eval and options.cmd in ["eval", "train-eval"]:
        parser.error("input_data_eval is not specified")

    if options.cmd == "train-eval":

        logging.info("------TRAINING--------")
        magic_box.train(input_dataset=options.input_data_train,
                        options=options)
        logging.info("------EVALUATION--------")
        magic_box.eval(input_dataset=options.input_data_eval,
                       options=options)
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

