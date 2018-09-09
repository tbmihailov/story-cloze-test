import sys
from __builtin__ import staticmethod

from utils.common_utilities import CommonUtilities
import time

import json
import codecs
from stanford_corenlp_pywrapper import CoreNLP
import logging
from copy import deepcopy

import random


# from random import random
def init_obj_raw_text(text):
    return {'raw_text': text}

####################
#####FEATS#########
####################
def get_token_key(token):
    return "t_%s" % token


def get_lemma_key(lemma):
    return "l_%s" % lemma


def get_pos_key(pos):
    return "p_%s" % pos


def get_pos2_key(pos2):
    return "p2_%s" % pos2


def get_lemma_pos_key(lemma, pos):
    return "lp_%s_%s" % (pos, lemma)


def get_lemma_pos2_key(lemma, pos2):
    return "lp2_%s_%s" % (pos2, lemma)


def get_token_pos_key(token, pos):
    return "tp_%s_%s" % (pos, token)


def get_token_pos2_key(token, pos2):
    return "tp2_%s_%s" % (pos2,token)


def extract_features(token, lemma, pos, pos_2):
    features = []

    features.append(get_token_key(token))
    features.append(get_lemma_key(lemma))
    # features.append(get_pos_key(pos))
    # features.append(get_pos2_key(pos_2))
    features.append(get_lemma_pos_key(lemma, pos))
    features.append(get_lemma_pos2_key(lemma, pos_2))
    features.append(get_token_pos_key(token, pos))
    features.append(get_token_pos2_key(token, pos_2))

    return features

class InvertedIndex(dict):
    def add_features(self, doc_id, features):
        for feat in features:
            if feat not in self:
                self[feat] = []
            self[feat].append(doc_id)

    def rank_similar_sentences(self, ending_item, include_features_fn):
        # get with similar PRP or NNP or NN
        feats = []
        for i in range(len(ending_item["tokens"])):
            token = ending_item["tokens"][i].lower()
            lemma = ending_item["lemmas"][i].lower()
            pos = ending_item["pos"][i].lower()
            pos_2 = pos[:min(2, len(pos))]

            curr_feats = include_features_fn(token, lemma, pos, pos_2)
            feats.extend(curr_feats)

        feats = list(set(feats))
        # logging.info("----------------------------------------------")
        # logging.info("rank_similar_sentences feats: %s" % (str(feats)))
        docs_ranking = {}
        for feat in feats:
            if feat in self:
                found_docs = list(set(self[feat]))
                for di in found_docs:
                    if di in docs_ranking:
                        docs_ranking[di] += 1
                    else:
                        docs_ranking[di] = 1
            # else:
            #     logging.info("feat %s is not presented!")

        if len(docs_ranking) == 0:
            return []
        # logging.info("docs found: %s" % (len(docs_ranking)))
        docs = [(k, v) for k, v in docs_ranking.items()]
        # logging.info("docs found tuple : %s" % (len(docs)))
        docs = sorted(docs, key=lambda arg: arg[1], reverse=True)
        # logging.info("docs found sorted : %s" % (len(docs)))
        return docs

class DataUtilities_ROCStories(object):
    @staticmethod
    def read_stories_data(input_file, max_items=0):
        # storyid	storytitle	sentence1	sentence2	sentence3	sentence4	sentence5
        # 9a51198e-96f1-42c3-b09d-a3e1e067d803	Overweight Kid	Dan's parents were overweight.	Dan was overweight as well.	The doctors told his parents it was unhealthy.	His parents understood and decided to make a change.	They got themselves and Dan on a diet.

        f = codecs.open(input_file, mode='rb', encoding='utf-8')
        line_id = 0

        items_list = []
        for line in f:
            if line_id == 0:  # skip header
                line_id += 1
                continue

            if max_items > 0 and line_id > max_items:
                break

            line_cols = line.split("\t")

            item = {}
            item['id'] = line_cols[0]  # storyid
            item['title'] = [init_obj_raw_text(line_cols[1])]
            item['sentences'] = []
            item['sentences'].append(init_obj_raw_text(line_cols[2]))  # InputSentence1
            item['sentences'].append(init_obj_raw_text(line_cols[3]))  # InputSentence2
            item['sentences'].append(init_obj_raw_text(line_cols[4]))  # InputSentence3
            item['sentences'].append(init_obj_raw_text(line_cols[5]))  # InputSentence4
            item['sentences'].append(init_obj_raw_text(line_cols[6]))  # InputSentence4

            items_list.append(item)
            line_id += 1

        return items_list

    @staticmethod
    def read_stories_annotated_data(input_file, max_items=0):
        # InputStoryid	InputSentence1	InputSentence2	InputSentence3	InputSentence4	RandomFifthSentenceQuiz1	RandomFifthSentenceQuiz2	AnswerRightEnding
        # b929f263-1dcd-4a0b-b267-5d5ff2fe65bb	My friends all love to go to the club to dance.	They think it's a lot of fun and always invite.	I finally decided to tag along last Saturday.	I danced terribly and broke a friend's toe.	My friends decided to keep inviting me out as I am so much fun.	The next weekend, I was asked to please stay home.	2
        f = codecs.open(input_file, mode='rb', encoding='utf-8')
        line_id = 0

        items_list = []
        for line in f:
            if line_id == 0:  # skip header
                line_id += 1
                continue

            if max_items > 0 and line_id > max_items:
                break

            line_cols = line.split("\t")

            item = {}
            item['id'] = line_cols[0]  # InputStoryid
            item['sentences'] = []
            item['sentences'].append(init_obj_raw_text(line_cols[1]))  # InputSentence1
            item['sentences'].append(init_obj_raw_text(line_cols[2]))  # InputSentence2
            item['sentences'].append(init_obj_raw_text(line_cols[3]))  # InputSentence3
            item['sentences'].append(init_obj_raw_text(line_cols[4]))  # InputSentence4

            item['endings'] = []
            item['endings'].append(init_obj_raw_text(line_cols[5]))  # RandomFifthSentenceQuiz1
            item['endings'].append(init_obj_raw_text(line_cols[6]))  # RandomFifthSentenceQuiz2
            item['right_end_id'] = int(line_cols[7]) - 1  # AnswerRightEnding

            items_list.append(item)

            line_id += 1

        return items_list

    @staticmethod
    def load_data_from_json_file(json_file):
        data_file = codecs.open(json_file, mode='r', encoding="utf-8")
        data = json.load(data_file)
        data_file.close()

        return data

    @staticmethod
    def save_data_to_json_file(data, output_json_file):
        data_file = codecs.open(output_json_file, mode='wb', encoding="utf-8")
        json.dump(data, data_file)
        data_file.close()

    @staticmethod
    def mutate_train_data_smart_1(data_in, seed=42):
        data = data_in

        words_stories_inverted_index = InvertedIndex()
        words_end_good_inverted_index = InvertedIndex()
        words_end_bad_inverted_index = InvertedIndex()

        logging.info("Building search...")
        for st_item_id in range(len(data)):
            if (st_item_id+1) % 1000 == 0:
                logging.info("Processed %s of %s" % (st_item_id+1, len(data)))
            st_item = data[st_item_id]
            # extract story
            for sent_i, sentence in st_item['sentences']:
                for i in range(len(sentence["tokens"])):
                    token = sentence["tokens"][i].lower()
                    lemma = sentence["lemmas"][i].lower()
                    pos = sentence["pos"][i].lower()
                    pos_2 = pos[:min(2, len(pos))]

                    feats = extract_features(token, lemma, pos, pos_2)
                    words_stories_inverted_index.add_features(st_item_id, feats)

            # extract endings
            for ending_i in range(len(st_item['endings'])):
                for i in range(len(st_item['endings'][ending_i]["tokens"])):
                    inverted_index = None
                    if ending_i == st_item["right_end_id"]:
                        inverted_index = words_end_good_inverted_index
                    else:
                        inverted_index = words_end_bad_inverted_index

                    token = st_item['endings'][ending_i]["tokens"][i].lower()
                    lemma = st_item['endings'][ending_i]["lemmas"][i].lower()
                    pos = st_item['endings'][ending_i]["pos"][i].lower()
                    pos_2 = pos[:min(2, len(pos))]

                    feats = extract_features(token, lemma, pos, pos_2)
                    if (st_item_id + 1) % 1000 == 0:
                        logging.info("Item: %s : feats:" % (str(st_item['endings'][ending_i]["raw_text"]), str(feats)))
                    inverted_index.add_features(st_item_id, feats)
            # print words_end_bad_inverted_index
        logging.info("Search built...")

        def get_features_pr_nn(token, lemma, pos, pos_2):
            feats = []
            if pos.startswith('pr') or pos.startswith('nn'):
                feats.append(get_lemma_pos_key(lemma, pos))
            return feats

        parsed_data = []
        random.seed(seed)
        items_cnt = len(data)
        for curr_item_id in range(len(data)):
            story_item = data[curr_item_id]
            story_item_copy = deepcopy(story_item)
            item = {}
            item['id'] = story_item_copy['id']  # InputStoryid
            item['sentences'] = []
            item['sentences'].append(story_item_copy['sentences'][0])  # InputSentence1
            item['sentences'].append(story_item_copy['sentences'][1])  # InputSentence2
            item['sentences'].append(story_item_copy['sentences'][2])  # InputSentence3
            item['sentences'].append(story_item_copy['sentences'][3])  # InputSentence4

            item['right_end_id'] = story_item_copy['right_end_id']
            right_ending_item = story_item_copy['endings'][story_item_copy['right_end_id']]
            wrong_ending_item = story_item_copy['endings'][int(not(bool(story_item_copy['right_end_id'])))]

            selected_endings = []
            for ri in range(9):
                selected_endings.append(wrong_ending_item)

            similar_wrong_endings = words_end_bad_inverted_index.rank_similar_sentences(wrong_ending_item, get_features_pr_nn)
            if (curr_item_id+1) % 1000 == 0:
                logging.info("")
                logging.info("Story %s - similar: %s" %(curr_item_id, str(similar_wrong_endings)))
                logging.info("story:%s" % wrong_ending_item["raw_text"])

            get_top = 3
            if len(similar_wrong_endings) > 0:
                for doc in similar_wrong_endings[:min(get_top, len(similar_wrong_endings))]:
                    sim_doc_id = doc[0]
                    if sim_doc_id != curr_item_id:
                        full_doc = data[sim_doc_id]
                        curr_sim_ending = full_doc["endings"][int(not(bool(full_doc['right_end_id'])))]
                        selected_endings.append(curr_sim_ending)
                        logging.info("Sim:%s" % curr_sim_ending["raw_text"])
            logging.info("-----------------")
            item['endings'] = []

            gold = True
            for sel_ending in selected_endings:
                new_item = deepcopy(item)
                new_item["gold"] = gold
                gold = False

                # get random ending
                right_end_id = random.randint(0, 1)

                if right_end_id == 0:
                    new_item['endings'].append(right_ending_item)
                    new_item['endings'].append(deepcopy(sel_ending))
                else:
                    new_item['endings'].append(deepcopy(sel_ending))
                    new_item['endings'].append(right_ending_item)

                new_item['right_end_id'] = right_end_id  # AnswerRightEnding

                parsed_data.append(new_item)

        print "Generated data %s" % (len(parsed_data))
        return parsed_data

    @staticmethod
    def mutate_rocstories_data_pos_v1(data_in, seed=42, take_number=20):
        data = data_in

        words_stories_inverted_index = InvertedIndex()
        words_end_last_sent_inverted_index = InvertedIndex()

        logging.info("Building search...")
        for st_item_id in range(len(data)):
            if (st_item_id + 1) % 1000 == 0:
                logging.info("Processed %s of %s" % (st_item_id + 1, len(data)))
            st_item = data[st_item_id]
            # extract story
            for sent_i, sentence in enumerate(st_item['sentences']):
                sent_feats = []
                for i in range(len(sentence["tokens"])):
                    token = sentence["tokens"][i].lower()
                    lemma = sentence["lemmas"][i].lower()
                    pos = sentence["pos"][i].lower()
                    pos_2 = pos[:min(2, len(pos))]

                    feats = extract_features(token, lemma, pos, pos_2)
                    sent_feats.extend(feats)

                # if (st_item_id + 1) % 1000 == 0:
                #     logging.info("Item: %s : feats:%s" % (str(sentence["raw_text"]), str(sent_feats)))

                if sent_i == 4:
                    words_end_last_sent_inverted_index.add_features(st_item_id, sent_feats)
                else:
                    words_stories_inverted_index.add_features(st_item_id, sent_feats)

        logging.info("Search built...")

        def get_features_pr_nn(token, lemma, pos, pos_2):
            feats = []
            if pos.startswith('pr') or pos.startswith('nn'):
                feats.append(get_lemma_pos_key(lemma, pos))
            return feats

        parsed_data = []
        random.seed(seed)
        items_cnt = len(data)
        for curr_item_id in range(len(data)):
            story_item = data[curr_item_id]
            story_item_copy = deepcopy(story_item)
            item = {}
            item['id'] = story_item_copy['id']  # InputStoryid
            item['sentences'] = []
            item['sentences'].append(story_item_copy['sentences'][0])  # InputSentence1
            item['sentences'].append(story_item_copy['sentences'][1])  # InputSentence2
            item['sentences'].append(story_item_copy['sentences'][2])  # InputSentence3
            item['sentences'].append(story_item_copy['sentences'][3])  # InputSentence4

            right_ending_item = story_item_copy['sentences'][-1]


            selected_endings = []

            sent_to_check = deepcopy(right_ending_item)

            include_sents = True
            if include_sents:
                for sent in story_item_copy['sentences'][:4]:
                    sent_to_check["tokens"].extend(sent["tokens"])
                    sent_to_check["lemmas"].extend(sent["lemmas"])
                    sent_to_check["pos"].extend(sent["pos"])

            similar_wrong_endings = words_end_last_sent_inverted_index.rank_similar_sentences(sent_to_check,
                                                                                              get_features_pr_nn)

            if (curr_item_id + 1) % 1000 == 0:
                logging.info("Processed %s of %s" % (curr_item_id + 1, len(data)))
                logging.info("Story %s - similar: %s" % (curr_item_id, str(len(similar_wrong_endings))))
                logging.info("story:%s" % right_ending_item["raw_text"])

            get_top = take_number
            get_top_sorted = 500
            endings_top_sorted = similar_wrong_endings[:min(get_top_sorted, len(similar_wrong_endings))]
            random_top = random.sample(xrange(len(endings_top_sorted)), min(get_top, len(endings_top_sorted)))
            random_top_docs = [endings_top_sorted[x] for x in random_top]
            if len(similar_wrong_endings) > 0:
                # shuffle(similar_wrong_endings)
                for doc in random_top_docs:
                    sim_doc_id = doc[0]
                    if sim_doc_id != curr_item_id:
                        full_doc = data[sim_doc_id]
                        curr_sim_ending = full_doc["sentences"][-1]
                        selected_endings.append((full_doc['id'], curr_sim_ending))
                        if (curr_item_id + 1) % 1000 == 0:
                            logging.info("Sim:%s" % curr_sim_ending["raw_text"])
                            logging.info("-----------------")
            item['endings'] = []

            for sel_ending_with_id in selected_endings:
                sel_ending_doc_id, sel_ending = sel_ending_with_id
                new_item = deepcopy(item)
                new_item["gold_story_id"] = new_item['id']
                new_item['id'] = "s_%s_e_%s" % (new_item['id'], sel_ending_doc_id)

                # get random ending
                right_end_id = random.randint(0, 1)

                if right_end_id == 0:
                    new_item['endings'].append(deepcopy(right_ending_item))
                    new_item['endings'].append(deepcopy(sel_ending))
                else:
                    new_item['endings'].append(deepcopy(sel_ending))
                    new_item['endings'].append(deepcopy(right_ending_item))

                new_item['right_end_id'] = right_end_id  # AnswerRightEnding

                parsed_data.append(new_item)

        print "Generated data %s" % (len(parsed_data))
        return parsed_data

    @staticmethod
    def mutate_train_data_smart_1(data_in, seed=42):
        data = data_in

        words_stories_inverted_index = InvertedIndex()
        words_end_good_inverted_index = InvertedIndex()
        words_end_bad_inverted_index = InvertedIndex()

        logging.info("Building search...")
        for st_item_id in range(len(data)):
            st_item = data[st_item_id]
            # extract story
            for sent_i, sentence in st_item['sentences']:
                for i in range(len(sentence["tokens"])):
                    token = sentence["tokens"][i].lower()
                    lemma = sentence["lemmas"][i].lower()
                    pos = sentence["pos"][i].lower()
                    pos_2 = pos[:min(2, len(pos))]

                    feats = extract_features(token, lemma, pos, pos_2)
                    words_stories_inverted_index.add_features(st_item_id, feats)

            # extract endings
            for ending_i in range(len(st_item['endings'])):
                for i in range(len(st_item['endings'][ending_i]["tokens"])):
                    inverted_index = None
                    if ending_i == st_item["right_end_id"]:
                        inverted_index = words_end_good_inverted_index
                    else:
                        inverted_index = words_end_bad_inverted_index

                    token = st_item['endings'][ending_i]["tokens"][i].lower()
                    lemma = st_item['endings'][ending_i]["lemmas"][i].lower()
                    pos = st_item['endings'][ending_i]["pos"][i].lower()
                    pos_2 = pos[:min(2, len(pos))]

                    feats = extract_features(token, lemma, pos, pos_2)
                    inverted_index.add_features(st_item_id, feats)
                    # print words_end_bad_inverted_index
        logging.info("Search built...")

        def get_features_pr_nn(token, lemma, pos, pos_2):
            feats = []
            if pos.startswith('pr') or pos.startswith('nn'):
                feats.append(get_lemma_pos_key(lemma, pos))
            return feats

        parsed_data = []
        random.seed(seed)
        items_cnt = len(data)
        for curr_item_id in range(len(data)):
            story_item = data[curr_item_id]
            story_item_copy = deepcopy(story_item)
            item = {}
            item['id'] = story_item_copy['id']  # InputStoryid
            item['sentences'] = []
            item['sentences'].append(story_item_copy['sentences'][0])  # InputSentence1
            item['sentences'].append(story_item_copy['sentences'][1])  # InputSentence2
            item['sentences'].append(story_item_copy['sentences'][2])  # InputSentence3
            item['sentences'].append(story_item_copy['sentences'][3])  # InputSentence4

            item['right_end_id'] = story_item_copy['right_end_id']
            right_ending_item = story_item_copy['endings'][story_item_copy['right_end_id']]
            wrong_ending_item = story_item_copy['endings'][int(not (bool(story_item_copy['right_end_id'])))]

            selected_endings = []
            for ri in range(9):
                selected_endings.append(wrong_ending_item)

            similar_wrong_endings = words_end_bad_inverted_index.rank_similar_sentences(wrong_ending_item,
                                                                                        get_features_pr_nn)
            logging.info("Story %s - similar: %s" % (curr_item_id, str(similar_wrong_endings)))
            logging.info("story:%s" % wrong_ending_item["raw_text"])

            get_top = 3
            if len(similar_wrong_endings) > 0:
                for doc in similar_wrong_endings[:min(get_top, len(similar_wrong_endings))]:
                    sim_doc_id = doc[0]
                    if sim_doc_id != curr_item_id:
                        full_doc = data[sim_doc_id]
                        curr_sim_ending = full_doc["endings"][int(not (bool(full_doc['right_end_id'])))]
                        selected_endings.append(curr_sim_ending)
                        logging.info("Sim:%s" % curr_sim_ending["raw_text"])
            logging.info("-----------------")
            item['endings'] = []

            gold = True
            for sel_ending in selected_endings:
                new_item = deepcopy(item)
                new_item["gold"] = gold
                gold = False

                # get random ending
                right_end_id = random.randint(0, 1)

                if right_end_id == 0:
                    new_item['endings'].append(right_ending_item)
                    new_item['endings'].append(deepcopy(sel_ending))
                else:
                    new_item['endings'].append(deepcopy(sel_ending))
                    new_item['endings'].append(right_ending_item)

                new_item['right_end_id'] = right_end_id  # AnswerRightEnding

                parsed_data.append(new_item)

        print "Generated data %s" % (len(parsed_data))
        return parsed_data

    @staticmethod
    def generate_random_train_data_from_raw_stories_json(data, random_number=3, seed=42):

        selected_endings = []

        parsed_data = []
        random.seed(seed)
        items_cnt = len(data)
        for story_item in data:
            story_item_copy = story_item

            item = {}
            item['id'] = story_item_copy['id']  # InputStoryid
            item['sentences'] = []
            item['sentences'].append(story_item_copy['sentences'][0])  # InputSentence1
            item['sentences'].append(story_item_copy['sentences'][1])  # InputSentence2
            item['sentences'].append(story_item_copy['sentences'][2])  # InputSentence3
            item['sentences'].append(story_item_copy['sentences'][3])  # InputSentence4

            item['endings'] = []
            for i in range(random_number):
                new_item = deepcopy(item)

                right_ending_item = deepcopy(story_item_copy['sentences'][4])

                # get random ending
                rand_story_idx = random.randint(0, items_cnt-1)
                selected_endings.append(rand_story_idx)

                rand_story_ending_cp = deepcopy(data[rand_story_idx]['sentences'][4])

                right_end_id = random.randint(0, 1)

                if right_end_id == 0:
                    new_item['endings'].append(right_ending_item)
                    new_item['endings'].append(rand_story_ending_cp)
                else:
                    new_item['endings'].append(rand_story_ending_cp)
                    new_item['endings'].append(right_ending_item)

                new_item['right_end_id'] = right_end_id  # AnswerRightEnding

                parsed_data.append(new_item)

        selected_endings_distinct = list(set(selected_endings))

        print "Distinct wrong endings:%s of %s" % (len(selected_endings_distinct), len(selected_endings))
        return parsed_data

    # @staticmethod
    # def generate_stories_with_random_data(data):
    #     parsed_data = []
    #     for story_item in data:
    #         story_item_copy = deepcopy(story_item)
    #         for list_field in ["sentences", "endings", "title"]:
    #             if not list_field in story_item_copy:
    #                 continue
    #             for i in range(0, len(story_item_copy[list_field])):
    #                 raw_text = story_item_copy[list_field][i]["raw_text"]
    #                 data_source_parse = parser.parse_doc(raw_text)
    #
    #                 sent_processed = deepcopy(data_source_parse["sentences"][0])
    #                 sent_processed["raw_text"] = raw_text
    #
    #                 story_item_copy[list_field][i] = sent_processed
    #
    #         parsed_data.append(story_item_copy)
    #
    #     return parsed_data

    @staticmethod
    def parse_stories_data_with_stanford_parser(data, parser):
        parsed_data = []
        for story_item in data:
            story_item_copy = deepcopy(story_item)
            for list_field in ["sentences", "endings", "title"]:
                if not list_field in story_item_copy:
                    continue
                for i in range(0, len(story_item_copy[list_field])):
                    raw_text = story_item_copy[list_field][i]["raw_text"]

                    data_source_parse = parser.parse_doc(raw_text)
                    # print "Raw text:%s"%raw_text
                    # logging.info(data_source_parse)
                    # print data_source_parse

                    sent_processed = deepcopy(data_source_parse["sentences"][0])
                    sent_processed["raw_text"] = raw_text

                    story_item_copy[list_field][i] = sent_processed

            parsed_data.append(story_item_copy)

        return parsed_data

    @staticmethod
    def parse_stories_annotated_data_with_stanford_parser(data, parser):
        return DataUtilities_ROCStories.parse_stories_data_with_stanford_parser(data, parser)

# Sample execution
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

# input_type=with_ending_choice  #raw_stories
# coreNlpPath="/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"
# input_files="data/cloze_test_test__spring2016-cloze_test_ALL_test.tsv"
# output_file="data/cloze_test_test__spring2016-cloze_test_ALL_test_processed.json"
# command=convert_to_json_with_parse
# parse_mode="pos"  # "pos", "parse"
# python DataUtilities_ROCStories.py -input_type:${input_type} -command:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}
if __name__ == "__main__":
    input_files = ""
    input_files = CommonUtilities.get_param_value("input_files", sys.argv, input_files)
    input_files_list = input_files.split(";")
    print "input_files:%s" % input_files

    input_type = "with_ending_choice"  # Story ending test/eval
    input_type = CommonUtilities.get_param_value("input_type", sys.argv, input_type)
    print "input_type:%s" % input_files

    has_error = False

    command = "convert_to_json_with_parse"
    command = CommonUtilities.get_param_value("cmd", sys.argv, command)

    output_files_clear = ""
    output_files_clear = CommonUtilities.get_param_value("output_files_clear", sys.argv, output_files_clear)
    output_files_clear_list = output_files_clear.split(";")
    print "output_files_clear:%s" % output_files_clear
    if len(output_files_clear_list) > 0 and len(input_files_list) <> len(
            output_files_clear_list) and command == "convert_to_json_with_parse":
        has_error = True
        print(
            'Error: Number of dirs in dir_src_out_clear_dirlist(%s) does not match the number of dirs in dir_src_txt_dirlist(%s)' % (
                len(output_files_clear_list), len(input_files_list)))


    output_file = "output_file"
    output_file = CommonUtilities.get_param_value("output_file", sys.argv, output_file)

    coreNlpPath = "/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"
    coreNlpPath = CommonUtilities.get_param_value("coreNlpPath", sys.argv, coreNlpPath)

    random_number = 10  # Story ending test/eval
    random_number = CommonUtilities.get_param_value_int("random_number", sys.argv, random_number)
    print "random_number:%s" % random_number

    parse_mode = "pos"  # "pos", "parse"
    parse_mode = CommonUtilities.get_param_value("parse_mode", sys.argv, parse_mode)
    if(command=="convert_to_json_with_parse"):
        data_format = "tac2014"
        print "Data format:%s" % data_format


        parser = CoreNLP(parse_mode, corenlp_jars=coreNlpPath.split(';'))

        start = time.time()

        data = []
        for dir_idx in range(len(input_files_list)):
            curr_input_file = input_files_list[dir_idx]

            json_data = []
            print "parsing"
            print "curr_input_file: %s" % curr_input_file
            if input_type == "with_ending_choice":
                curr_file_raw_data = DataUtilities_ROCStories.read_stories_annotated_data(curr_input_file)

                print("%s items loaded from %s", (len(curr_file_raw_data), curr_input_file))
                json_data = DataUtilities_ROCStories.parse_stories_annotated_data_with_stanford_parser(curr_file_raw_data, parser=parser)
            elif input_type == "raw_stories":
                curr_file_raw_data = DataUtilities_ROCStories.read_stories_data(curr_input_file)
                print("%s items loaded from %s", (len(curr_file_raw_data), curr_input_file))
                json_data = DataUtilities_ROCStories.parse_stories_data_with_stanford_parser(curr_file_raw_data, parser=parser)
            else:
                raise Exception("input_type not supported: %s" % input_type)

            data.extend(json_data)
        end = time.time()
        print("Done in %s s" % (end - start))

        print len(data)
        DataUtilities_ROCStories.save_data_to_json_file(data, output_json_file=output_file)
        print ("Data exported to %s" % output_file)
        print("Data exported to %s" % output_file)

    elif (command == "generate_train_random"):
        start = time.time()



        data = []
        for dir_idx in range(len(input_files_list)):
            curr_input_file = input_files_list[dir_idx]

            if input_type == "raw_stories_json":
                json_data = DataUtilities_ROCStories.load_data_from_json_file(curr_input_file)
            else:
                raise Exception("cmd generate_train_random: input_type not supported: %s" % input_type)
            data.extend(json_data)

        data_train = []
        data_train = DataUtilities_ROCStories.generate_random_train_data_from_raw_stories_json(data, random_number=random_number, seed=422)

        end = time.time()
        print("Done in %s s" % (end - start))

        print len(data)
        DataUtilities_ROCStories.save_data_to_json_file(data_train, output_json_file=output_file)
        print ("Data exported to %s" % output_file)
        print("Data exported to %s" % output_file)
    elif (command == "mutate_train_smart_1"):
        start = time.time()

        data = []
        for dir_idx in range(len(input_files_list)):
            curr_input_file = input_files_list[dir_idx]

            if input_type == "with_ending_choice":
                json_data = DataUtilities_ROCStories.load_data_from_json_file(curr_input_file)
            else:
                raise Exception("cmd mutate_train_smart_1: input_type not supported: %s" % input_type)

        data.extend(json_data)

        data_train=[]
        data_train = DataUtilities_ROCStories.mutate_train_data_smart_1(data, seed=422)

        end = time.time()
        print("Done in %s s" % (end - start))

        print len(data)
        DataUtilities_ROCStories.save_data_to_json_file(data_train, output_json_file=output_file)
        print ("Data exported to %s" % output_file)
        print("Data exported to %s" % output_file)
    elif (command == "randomize_data_90_10"):
        start = time.time()

        data = []
        for dir_idx in range(len(input_files_list)):
            curr_input_file = input_files_list[dir_idx]

            if input_type == "with_ending_choice":
                json_data = DataUtilities_ROCStories.load_data_from_json_file(curr_input_file)
            else:
                raise Exception("cmd mutate_train_smart_1: input_type not supported: %s" % input_type)

            data.extend(json_data)

        data_size = len(data)
        train_size_percent = 90

        select_top = int(train_size_percent * (float(data_size)/100.00))
        shuffle = True
        if shuffle:
            shuffle(data)

        data_1 = data[:select_top]
        data_2 = data[select_top:]

        print "data_1:%s" %len(data_1)
        DataUtilities_ROCStories.save_data_to_json_file(data_1, output_json_file="%s.split_%s.json" % (output_file, train_size_percent))
        print ("Data exported to %s" % output_file)

        print "data_2:%s" % len(data_2)
        DataUtilities_ROCStories.save_data_to_json_file(data_2, output_json_file="%s.split_%s.json" % (output_file, (100-train_size_percent)))
        print ("Data exported to %s" % output_file)

        end = time.time()
        print("Done in %s s" % (end - start))

    elif (command == "mutate_rocstories_data_pos_v1"):
        start = time.time()

        data = []
        for dir_idx in range(len(input_files_list)):
            curr_input_file = input_files_list[dir_idx]

            if input_type == "raw_stories":
                json_data = DataUtilities_ROCStories.load_data_from_json_file(curr_input_file)
            else:
                raise Exception("cmd mutate_rocstories_data_pos_v1: input_type not supported: %s" % input_type)

            data.extend(json_data)

        data_train = []
        data_train = DataUtilities_ROCStories.mutate_rocstories_data_pos_v1(data, seed=422, take_number = random_number)

        end = time.time()
        print("Done in %s s" % (end - start))

        print len(data)
        DataUtilities_ROCStories.save_data_to_json_file(data_train, output_json_file=output_file)
        print ("Data exported to %s" % output_file)
        print("Data exported to %s" % output_file)
    else:
        print "No command param specified: use -cmd:convert_to_json_2014 or -cmd:convert_to_json_2015 "