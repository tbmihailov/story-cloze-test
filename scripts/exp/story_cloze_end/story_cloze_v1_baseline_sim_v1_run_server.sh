#!/usr/bin/env bash

run_name=story_cloze_baseline_sim_v1
#model dir where output models are saved after train

resources_dir=~/research/resources
embeddings_files=(
"${resources_dir}/glove/glove.6B.50d.txt.gensim.bin"
"${resources_dir}/glove/glove.6B.100d.txt.gensim.bin"
"${resources_dir}/glove/glove.6B.200d.txt.gensim.bin"
"${resources_dir}/glove/glove.6B.300d.txt.gensim.bin"
"${resources_dir}/glove/glove.840B.300d.txt.gensim.bin"
"${resources_dir}/concepnet_vectors/conceptnet-numberbatch-201609_en_full.txt.gensim.bin"
"${resources_dir}/word2vec/GoogleNews-vectors-negative300.bin"
)

embeddings_run_names=(
"gl.6B.050d"
"gl.6B.100d"
"gl.6B.200d"
"gl.6B.300d"
"gl.840B.300d"
"concept.201609.full.en"
"w2v.GN.300d"
)

embeddings_load_binary=(
True
True
True
True
True
True
True
)

scale_features=True
tune=True
param_c=1.0

loop_emb_file=${embeddings_files[6]},${embeddings_files[7]}
loop_emb_name=${embeddings_run_names[6]}
loop_emb_binary=${embeddings_load_binary[6]}

emb_model_type=w2v
# emb_model_file=resources/word2vec/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
emb_model_file=${loop_emb_file}
word2vec_load_bin=True # True for google news w2v
load_features=False

lower_tokens=True
remove_stopwords=True
inc_embedd_vectors=True
inc_slast1=False
inc_slast2=False
inc_slast3=False
inc_slast4=False
inc_story=True
inc_maxsim=False
inc_possim=False
inc_fullsims=False
include_elem_multiply=False

run_setup=sl1${inc_slast1}_sl2${inc_slast2}_sl3${inc_slast3}_sl4${inc_slast4}_st${inc_story}_ms${inc_maxsim}_${inc_possim}_${inc_fullsims}_mul${include_elem_multiply}_c${param_c}

max_records=0

train_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
dev_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
test_data=resources/roc_stories_data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json

run_name=${run_name}_lf${load_features}_tune${tune}_${run_setup}
run_name=${run_name}_$(date +%y-%m-%d-%H-%M-%S)

model_dir=saved_models/${run_name}
output_dir=output/${run_name}

model_file=${model_dir}/logreg_model.pickle
scale_file=${model_dir}/scale_file.pickle

output_file=${run_name}_submission.txt
submission_data_eval="submission_${run_name}_test_spring2016.txt"

echo "=========TRAIN-eval========="
cmd=train-eval

# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

input_data_train=${train_data}
input_data_eval=${test_data}

log_file=res_exp_${run_name}.log
print "Saving log to ${log_file}"

venv/bin/python story_cloze_v1_features_and_similarities.py --cmd ${cmd} --input_data_train ${input_data_train} --input_data_eval ${input_data_eval} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records} --tune ${tune} --submission_data_eval ${submission_data_eval} --load_features ${load_features}  --lower_tokens ${lower_tokens}  --remove_stopwords ${remove_stopwords}  --inc_embedd_vectors ${inc_embedd_vectors}  --inc_slast1 ${inc_slast1}  --inc_slast2 ${inc_slast2}  --inc_slast3 ${inc_slast3}  --inc_slast4 ${inc_slast4}  --inc_story ${inc_story}  --inc_maxsim ${inc_maxsim}  --inc_possim ${inc_possim}  --inc_fullsims ${inc_fullsims}  --include_elem_multiply ${include_elem_multiply} --param_c ${param_c}> ${log_file}

exit
#echo "=========TRAIN========="
#cmd=train
#
## clear model folders
#mkdir -p ${output_dir}
#rm -rf -- ${model_dir}
#mkdir -p ${model_dir}
#
#
#input_data=${train_data}
#venv/bin/python story_cloze_v1_features_and_similarities.py --cmd ${cmd} --input_data_train ${input_data} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records}
#
#echo "=========EVAL========="
#cmd=eval
#
#input_data=${test_data}
#venv/bin/python story_cloze_v1_features_and_similarities.py --cmd ${cmd} --input_data ${input_data} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records}
