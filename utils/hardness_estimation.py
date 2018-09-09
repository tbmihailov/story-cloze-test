import codecs
import glob
import sys
from optparse import OptionParser

import logging

# Sample run
# Set logging info
from data.StoryClozeTest.DataUtilities_ROCStories import DataUtilities_ROCStories

logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Enable console logging
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

import os

def load_data(file_name, has_header=True):
    data = []
    f = codecs.open(file_name, mode='rb')
    lid = 0
    headers = []
    for line in f:
        lid += 1
        if lid == 1 and has_header:
            headers = line.strip().split('\t')
            continue
        fields = [x.strip() for x in line.strip().split('\t')]
        data.append(fields)

    f.close()

    return data, headers

import string
def get_ascii_only(txt):
    return ''.join(filter(lambda x: x in string.printable, txt))

def export_data(file_name, data, data_headers, has_header=True):

    f = codecs.open(file_name, mode='wb', encoding='utf-8')
    lid = 0

    line = '\t'.join(data_headers+['pos', 'neg'])

    f.write(line+'\n')


    for i in range(len(data)):
        line = '\t'.join([x.decode('utf-8', 'ignore') for x in data[i][:-2]]+[str(x) for x in data[i][-2:]])
        f.write(line + '\n')



    f.close()

def load_submission(file_name, has_header=True):
    data = []
    f = codecs.open(file_name, mode='rb')
    lid = 0
    for line in f:
        lid += 1
        if lid == 1 and has_header:
            continue
        fields = [x.strip() for x in line.strip().split(',')]
        data.append([fields[0], fields[1]])

    f.close()

    return data

# python hardness_estimation.py --gold_data "data/cloze_test_test__spring2016-cloze_test_ALL_test.tsv" --submissions "./submission_story_cloze\*"

# gold_data_json=resources/roc_stories_data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json
# out_file=${gold_data}.hardness.txt
# submissions="./submissions_roc1617-random3-2017-02-04-21-25/submission_\*"
# python hardness_estimation.py --gold_data_json ${gold_data_json} --submissions ${submissions} --out_file ${out_file}

# 2017-02-07
# gold_data_json=resources/roc_stories_data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_10_17-02-06-15-44-21.json
# out_file=${gold_data}.hardness.txt
# submissions="./submissions-rc1617pos10/submission_\*"
# python hardness_estimation.py --gold_data_json ${gold_data_json} --submissions ${submissions} --out_file ${out_file}

# 2017-02-20
# gold_data_json=resources/roc_stories_data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_20_17-02-20-12-01-35.json
# out_file=${gold_data}.hardness.goodfilter.txt
# submissions="./data/filtered_pp_rn-2017-02-20/submission_\*"
# python hardness_estimation.py --gold_data_json ${gold_data_json} --submissions ${submissions} --out_file ${out_file}

# 2017-02-20 random
# gold_data_json=resources/roc_stories_data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-random10-2017-02-20-16-47-v01.json
# out_file=${gold_data}.hardness.goodfilter.txt
# submissions="./data/filtered_rnd10-2017-02-20/subm\*"
# python hardness_estimation.py --gold_data_json ${gold_data_json} --submissions ${submissions} --out_file ${out_file}

# 2017-02-21-rnd10easy3.lstm filter
# gold_data_json=resources/roc_stories_data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-random10-2017-02-20-16-47-v01.json.easy3.txt
# out_file=${gold_data}.hardness.goodfilter.txt
# submissions="./data/filter-2017-02-21-te.rc1617rnd10v01.es3/subm_\*"
# python hardness_estimation.py --gold_data_json ${gold_data_json} --submissions ${submissions} --out_file ${out_file}

# 2017-02-21 filter-2017-02-21-rc1617prnnt10fx
# gold_data_json=resources/roc_stories_data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_top10_fx_17-02-21-00-56-18.json
# out_file=${gold_data}.hardness.goodfilter.txt
# submissions="./data/filter-2017-02-21-rc1617prnnt10fx/subm\*"
# python hardness_estimation.py --gold_data_json ${gold_data_json} --submissions ${submissions} --out_file ${out_file}

# 2017-02-21 filter-2017-02-21-rc1617prnnt10fx
# gold_ls -tr

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--gold_data", dest="gold_data", help="")
    parser.add_option("--gold_data_json", dest="gold_data_json", help="")
    parser.add_option("--out_file", dest="out_file", help="")
    parser.add_option("--submissions", dest="submissions", help="")

    (options, args) = parser.parse_args()

    # print options
    options_print = ""
    logging.info("Options:")
    for k, v in options.__dict__.iteritems():
        options_print += "opt: %s=%s\r\n" % (k, v)
    logging.info(options_print)

    submissions = []
    subm_files = glob.glob(options.submissions.replace("\\*", "*"))

    logging.info(subm_files)


    for fn in subm_files:
        #logging.info(fn)
        if os.path.isfile(fn):
            subm_data = load_submission(fn, has_header=True)
            logging.info("Submission %s" % fn)
            # logging.info("submission loaded")
            submissions.append(subm_data)
            print "Submissions %s" % len(submissions)
        else:
            print "Not a file!"

    if len(submissions) == 0:
        raise Exception("Submissions count %s" % len(submissions))

    print "Submissions %s" % len(submissions)
    print submissions[0][0]

    if options.gold_data_json:
        gold_data_parsed = DataUtilities_ROCStories.load_data_from_json_file(options.gold_data_json)
        gold_data_headers = "InputStoryid	InputSentence1	InputSentence2	InputSentence3	InputSentence4	RandomFifthSentenceQuiz1	RandomFifthSentenceQuiz2	AnswerRightEnding".split('\t')
        gold_data = []
        for i in range(len(gold_data_parsed)):
            gold_data_item = [get_ascii_only(x).strip() for x in [gold_data_parsed[i]["id"]]+[x["raw_text"] for x in gold_data_parsed[i]["sentences"]]+[x["raw_text"].strip() for x in gold_data_parsed[i]["endings"]] + [str(int(gold_data_parsed[i]["right_end_id"])+1)]]
            # print gold_data_item
            # break
            gold_data.append(gold_data_item)
    else:
        gold_data, gold_data_headers = load_data(options.gold_data, has_header=True)
    # print gold_data

    data_with_score = []
    for i in range(len(gold_data)):
        data_id = gold_data[i][0]
        true_value = gold_data[i][-1]
        pos = 0
        neg = 0

        for submission in submissions:
            res = submission[i][1]
            # logging.info("res %s | true_value: %s" % (res, true_value))
            if res == true_value:
                pos += 1
            else:
                neg += 1

        data_with_score.append(gold_data[i]+[pos, neg])

    # export data by hardness
    for i in range(20):
        print data_with_score[i]

    logging.info("Export data by hardness")
    # print statistics
    stat = {}
    subm_cnt = len(submissions)

    hardness = {}
    for i in range(len(data_with_score)):
        # hardness_weight = math.ceil(10*(float(data_with_score[i][-2])/float(subm_cnt)))
        hardness_weight = data_with_score[i][-2]
        if not hardness_weight in hardness:
            hardness[hardness_weight] = []
        hardness[hardness_weight].append(i)

    print [(k,len(v)) for k,v in hardness.iteritems()]


    # export to hardness files
    for k, v in hardness.iteritems():
        out_file_h = "%s.easy%s.txt.readable.tsv" % (options.gold_data_json, k)

        # export hardness file
        export_data(out_file_h, [data_with_score[x] for x in v], gold_data_headers)
        logging.info("Hardness %s - %s items saved to %s" %(k,len(v), out_file_h))

        if options.gold_data_json:
            gold_data_h = "%s.easy%s.txt" % (options.gold_data_json, k)
            golf_data_hardness = [gold_data_parsed[x] for x in v]
            DataUtilities_ROCStories.save_data_to_json_file(golf_data_hardness, gold_data_h)
            logging.info("Hardness %s - %s items saved to %s" % (k, len(v), gold_data_h))

    logging.info("data with score len %s" % len(data_with_score))
    hardness_f = options.out_file if options.out_file is not None else options.gold_data+'.hardness.tsv'
    export_data(hardness_f, data_with_score, gold_data_headers)

    logging.info("Data exported:\n %s" % hardness_f)














