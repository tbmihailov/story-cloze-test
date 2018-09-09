#!/usr/bin/env bash

run_name=story_cloze_v2_lstm_v1
#model dir where output models are saved after train
model_dir=saved_models/${run_name}
output_dir=output/${run_name}

scale_features=True
tune=False
emb_model_type=w2v
emb_model_file=resources/word2vec/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
word2vec_load_bin=False # true for google news w2v

max_records=0
model_file=${model_dir}/${run_name}_logreg_model.pickle
scale_file=${model_dir}/${run_name}_scale_file.pickle

output_file=${run_name}_submission.txt

train_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
dev_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
test_data=resources/roc_stories_data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json

echo "=========TRAIN-eval========="
cmd=train-eval

# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

# NN model settings
lstm_hidden_size=256
lstm_layers=2
batch_size=20
learn_rate=0.01
min_word_freq=5
num_epochs=10
train_embeddings=True
short_embeddings=False
dropout_prob_keep=0.5
layer_out_noise=0.0
loss_l2_beta=0.00

run_name=${run_name}_l${lstm_layers}_hs${lstm_hidden_size}_ep${num_epochs}_bs${batch_size}_lr${learn_rate}_tre${train_embeddings}_she${short_embeddings}
log_file=res_exp_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
print "Saving log to ${log_file}"

input_data_train=${train_data}
input_data_eval=${test_data}
input_data_eval_test=${test_data}
submission_data_eval="${log_file}.submission.txt"
venv/bin/python story_cloze_v2_neural_lstm.py --cmd ${cmd} --input_data_train ${input_data_train} --input_data_eval ${input_data_eval} --input_data_eval_test ${input_data_eval_test} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records} --tune ${tune} --submission_data_eval ${submission_data_eval} --lstm_hidden_size ${lstm_hidden_size} --lstm_layers ${lstm_layers} --batch_size ${batch_size}  --learn_rate ${learn_rate} --min_word_freq ${min_word_freq} --num_epochs ${num_epochs} --train_embeddings ${train_embeddings} --short_embeddings ${short_embeddings} --dropout_prob_keep ${dropout_prob_keep} --layer_out_noise ${layer_out_noise} --loss_l2_beta ${loss_l2_beta}> ${log_file}
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
