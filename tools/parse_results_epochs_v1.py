import re
import codecs
import sys
import string

file_name = sys.argv[1]


def extract_epochs_accuracies(file_name):
    f = codecs.open(file_name, mode='rb', encoding='utf-8')
    # epoch_results = {}
    step_results = []

    option_params = {}
    re_opt_param = "opt:\s(.+)=(.+)"
    re_get_epoch_id = "Step ([\d]+) ,"
    re_get_curr_score_id = "\saccuracy_score:([\d]+\.*[\d]*)"

    epoch_id = None
    acc = None
    can_read_accuracy = False  # this is because we also report train accuracy
    for line in f:
        # print line
        m = re.search(re_get_epoch_id, line)
        if m:
            epoch_id_new = int(m.group(1))
            epoch_id = epoch_id_new
            print "new epoch %s" % epoch_id_new
            can_read_accuracy = True
            acc = None
            continue

        m1 = re.search(re_get_curr_score_id, line)
        if m1 and can_read_accuracy:
            acc = float(m1.group(1))
            print "new accuracy %s" % acc
            can_read_accuracy = False
            step_results.append((epoch_id, acc))
            continue

        mp = re.search(re_opt_param, line)
        if mp:
            param_name = mp.group(1)
            param_val = mp.group(2)
            option_params[param_name] = param_val
            print (param_name, param_val)
            continue


    f.close()

    return option_params, step_results


opt_params, results = extract_epochs_accuracies(file_name)
if len(results) == 0:
    results.append((0, 0))

res_vals = [x[1] for x in results]
import numpy as np

max_arg = np.argmax(np.asarray(res_vals))
max_val = res_vals[max_arg]
print "%s\t%s\t%s\t%s" % (file_name, max_arg, max_val, string.join(["%s" % x for x in res_vals], "\t"))

# test_file=${dir}/results/res_exp_story_cloze_v2_lstm_v1_l1_hs256_ep100_bs50_lr0.001_treTrue_sheFalse_17-01-30-23-01-29.log

# python

