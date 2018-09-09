
# tool for listing experiments

# columns:
# bash file name
# last time modified
# created
# last step
# last accuracy
# best step
# best accuracy


# 2. Select output files regarding a pattern, default - all files in outputs folder:
# https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python

# 3. Select the last results
import codecs
import json
from optparse import OptionParser

import datetime

parser = OptionParser()

# read params
parser.add_option("-f", "--files", dest="file_names",
                  help="Files filter", default="output/*/*_summary.json")

parser.add_option("-l", "--log_dir", dest="log_dir",
                  help="Files filter", default="")

parser.add_option("--max_step", "--max_step", dest="max_step",
                  help="Max step to select result", default=0, type="int")

parser.add_option("--acc_by_step", "--acc_by_step", dest="acc_by_step",
                  help="Print step statistics", default="False")

parser.add_option("--acc_by_step_loss", "--acc_by_step_loss", dest="acc_by_step_loss",
                  help="Include loss", default="True")


# parser.add_option("--res_from_log", "--res_from_log", dest="res_from_log",
#                   help="Check result values from log file", default="False")

(options, args) = parser.parse_args()

options_print = ""
for k, v in options.__dict__.iteritems():
    options_print += "opt: %s=%s\r\n" % (k, v)
print(options_print)


import os
from glob import glob

file_names = glob(options.file_names)

file_stats = [(f, os.stat(f)) for f in file_names]

file_stats.sort(key=lambda f:f[1].st_mtime)

for fs in file_stats:
    file_name = fs[0]
    file_info = fs[1]

    print_result = ""

    delim = "\t"
    try:
        summary = json.load(codecs.open(file_name, mode='rb'))
        run_options = summary["options"]
        run_name = run_options["run_name"]

        # get the log files to get bash name
        log_file_wildcard = ((options.log_dir + "/") if len(options.log_dir) > 0 else "") + "*" + run_name + "*.log"

        log_file_names = list(glob(log_file_wildcard))
        log_file_names = [x for x in log_file_names if os.path.isfile(x) and (run_name in x) ]
        log_file_name = "no-log-file-found"
        experiment_bash_file = "no-experiment-file-name-found"
        if len(log_file_names) > 0:

            log_file_name = log_file_names[-1]
            experiment_bash_file = "no experiment .sh name found"
            f_log = codecs.open(log_file_name, mode="rb")
            for line in f_log:
                if line.startswith("bash "):
                    index = line.find("\"resume\"")
                    experiment_bash_file = line[len("bash "):index]

        print_result += experiment_bash_file + delim
        summary_train_last_step = summary["data_dev"]["steps"][-1] if len(summary["data_dev"]["steps"]) > 0 else 0

        # best test
        best_test_acc_idx = 0
        best_test_acc = 0.00
        for i in range(len(summary["data_test"]["accuracy"])):
            if (options.max_step > 0) and (summary["data_test"]["steps"][i] > options.max_step):
                break
            if summary["data_test"]["accuracy"][i] > best_test_acc:
                best_test_acc_idx = i
                best_test_acc = summary["data_test"]["accuracy"][i]

        summary_test_acc = best_test_acc
        summary_test_step = summary["data_test"]["steps"][best_test_acc_idx]
        summary_train_acc = summary["data_train"]["accuracy"][best_test_acc_idx]
        summary_dev_acc = summary["data_dev"]["accuracy"][best_test_acc_idx]

        print_result += str(summary_test_step) + delim + str(summary_train_acc) + delim + str(summary_dev_acc) + delim + str(summary_test_acc) + delim

        # best dev
        best_dev_acc_idx = 0
        best_dev_acc = 0.00
        for i in range(len(summary["data_dev"]["accuracy"])):
            if (options.max_step > 0) and (summary["data_dev"]["steps"][i] > options.max_step):
                break

            if summary["data_dev"]["accuracy"][i] > best_dev_acc:
                best_dev_acc_idx = i
                best_dev_acc = summary["data_dev"]["accuracy"][i]

        summary_dev_acc = best_dev_acc
        summary_dev_step = summary["data_dev"]["steps"][best_dev_acc_idx]
        summary_train_acc = summary["data_train"]["accuracy"][best_dev_acc_idx]
        summary_test_acc = summary["data_test"]["accuracy"][best_dev_acc_idx]

        print_result += str(summary_dev_step) + delim + str(summary_train_acc) + delim + str(summary_dev_acc) + delim + str(summary_test_acc) + delim

        # best train
        best_train_acc_idx = 0
        best_train_acc = 0.00
        for i in range(len(summary["data_train"]["accuracy"])):
            if (options.max_step > 0) and (summary["data_train"]["steps"][i] > options.max_step):
                break
            if summary["data_train"]["accuracy"][i] > best_train_acc:
                best_train_acc_idx = i
                best_train_acc = summary["data_train"]["accuracy"][i]

        summary_train_acc = best_train_acc
        summary_train_step = summary["data_train"]["steps"][best_train_acc_idx]
        summary_dev_acc = summary["data_dev"]["accuracy"][best_train_acc_idx]
        summary_test_acc = summary["data_test"]["accuracy"][best_train_acc_idx]

        print_result += str(summary_dev_step) + delim + str(summary_train_acc) + delim + str(
            summary_dev_acc) + delim + str(summary_test_acc) + delim

        print_result += str(summary_train_last_step) + delim
        print_result += run_name + delim
        print_result += datetime.datetime.fromtimestamp(file_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S') + delim
        print_result += log_file_name + delim

        if options.acc_by_step == "True":
            # steps
            curr_steps_val_print = delim.join([str(x) for x in summary["data_dev"]["steps"]])
            print print_result + delim + "print_steps" + delim + curr_steps_val_print

            if options.acc_by_step_loss == "True":
                curr_steps_val_print = "no loss reported"
                if "loss" in summary["data_train"]:
                    curr_steps_val_print = delim.join([str(x) for x in summary["data_train"]["loss"]])
                print print_result + delim + "print_train_loss" + delim + curr_steps_val_print
            # train accuracy
            curr_steps_val_print = delim.join([str(x) for x in summary["data_train"]["accuracy"]])
            print print_result + delim + "print_train_acc" + delim + curr_steps_val_print

            # train accuracy
            curr_steps_val_print = delim.join([str(x) for x in summary["data_dev"]["accuracy"]])
            print print_result + delim + "print_dev_acc" + delim + curr_steps_val_print

            # train accuracy
            curr_steps_val_print = delim.join([str(x) for x in summary["data_test"]["accuracy"]])
            print print_result + delim + "print_test_acc" + delim + curr_steps_val_print

        else:
            print print_result
        #print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (experiment_bash_file, summary_train_last_step, summary_dev_step, summary_train_acc, summary_dev_acc, summary_test_acc, run_name, datetime.datetime.fromtimestamp(file_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')))
    except Exception as e:
        print("Error for %s\t%s" % (file_name, e))





