#!/usr/bin/env bash

#local
coreNlpPath="/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*;/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-srparser-2014-10-23-models.jar"
#coreNlpPath="/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*;"

#server
coreNlpPath="/home/mitarb/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"
coreNlpPath="/home/mitarb/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*;"

parse_mode="pos"  # "pos", "parse"
parse_mode="parse"  # "pos", "parse"
parse_mode="coref"  # "pos", "parse"
parse_mode="ner"  # "pos", "parse"
command=convert_to_json_with_parse

input_type=with_ending_choice  #raw_stories
input_files="resources/roc_stories_data/cloze_test_val__spring2016-cloze_test_ALL_val.tsv"
output_file="resources/roc_stories_data/processed_${parse_mode}_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json"
python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}

input_type=with_ending_choice  #raw_stories
input_files="resources/roc_stories_data/cloze_test_test__spring2016-cloze_test_ALL_test.tsv"
output_file="resources/roc_stories_data/processed_${parse_mode}_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json"
python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}

# 2017  release
input_type=raw_stories   #with_ending_choice, raw_stories
input_files="resources/roc_stories_data/ROCStories_winter2017_ROCStories_winter2017.tsv"
output_file="resources/roc_stories_data/processed_${parse_mode}_ROCStories_winter2017_ROCStories_winter2017.json"
python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}

# 2016 naacl release
input_type=raw_stories   #with_ending_choice, raw_stories
input_files="resources/roc_stories_data/ROCStories__spring2016-ROC-Stories-naacl-camera-ready.tsv"
output_file="resources/roc_stories_data/processed_${parse_mode}_ROCStories__spring2016-ROC-Stories-naacl-camera-ready.tsv.json"
python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}


################################################
##############Data Generation###################
################################################

# Generate train data
# coreNlpPath="/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"
coreNlpPath=~/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*
parse_mode="pos"  # "pos", "parse"

command=generate_train_random

input_type=raw_stories_json  #raw_stories
roc_sotries_2016=resources/roc_stories_data/processed_parse_ROCStories__spring2016-ROC-Stories-naacl-camera-ready.tsv.json
roc_sotries_2017=resources/roc_stories_data/processed_parse_ROCStories_winter2017_ROCStories_winter2017.json
roc_sotries_2016=resources/roc_stories_data/processed_pos_ROCStories__spring2016-ROC-Stories-naacl-camera-ready.tsv.json
roc_sotries_2017=resources/roc_stories_data/processed_pos_ROCStories_winter2017_ROCStories_winter2017.json
input_files="${roc_sotries_2016};${roc_sotries_2017}"

random_number=10
run_name=${command}_${random_number}_$(date +%y-%m-%d-%H-%M-%S)

output_file="resources/roc_stories_data/generated_${command}_proc_pos_ROCStories__ROC-Stories-2016-2017-random${random_number}.json"

python DataUtilities_ROCStories.py -random_number:${random_number} -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode} > ${run_name}.log


# Mutate data
# coreNlpPath="/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"
# randomize_data_90_10
coreNlpPath=~/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*
parse_mode="pos"  # "pos", "parse"

command=mutate_train_smart_1

input_type=with_ending_choice  #raw_stories
input_files="resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json"

random_number=10
run_name=${command}_${random_number}_$(date +%y-%m-%d-%H-%M-%S)
output_file="resources/roc_stories_data/generated__proc_pos_ROCStories__ROC-16-17-${run_name}.json"
output_file="resources/roc_stories_data/generated_cloze_test_val2016_${run_name}.json"

python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}

##############
#####ROC Stories Mutate######
##########
coreNlpPath=~/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*
parse_mode="pos"  # "pos", "parse"

command=mutate_rocstories_data_pos_v1

input_type=raw_stories # with_ending_choice

roc_sotries_2016=resources/roc_stories_data/processed_pos_ROCStories__spring2016-ROC-Stories-naacl-camera-ready.tsv.json
roc_sotries_2017=resources/roc_stories_data/processed_pos_ROCStories_winter2017_ROCStories_winter2017.json
input_files="${roc_sotries_2016};${roc_sotries_2017}"
random_number=10
run_name=${command}_${random_number}_$(date +%y-%m-%d-%H-%M-%S)
output_file="resources/roc_stories_data/generated_proc_pos_ROCStories_ROC-16-17-${run_name}.json"

log_file=gen_${run_name}_roc1617.log
python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode} -random_number:${random_number} > ${log_file}


# Mutate data
# coreNlpPath="/home/mihaylov/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"
# randomize_data_90_10
coreNlpPath=~/research/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*
parse_mode="pos"  # "pos", "parse"

command=randomize_data_90_10

input_type=with_ending_choice  #raw_stories
input_files="resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json"

random_number=10
run_name=${command}_${random_number}_$(date +%y-%m-%d-%H-%M-%S)
output_file="resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json"

python DataUtilities_ROCStories.py -input_type:${input_type} -cmd:${command} -input_files:${input_files} -output_file:${output_file} -coreNlpPath:${coreNlpPath} -parse_mode:${parse_mode}
