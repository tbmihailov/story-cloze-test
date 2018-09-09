#!/usr/bin/env bash

run_name=story_cloze_v4_lstm_att
#model dir where output models are saved after train

scale_features=True
tune=False
emb_model_type=w2v
emb_model_file=resources/word2vec/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
word2vec_load_bin=False # true for google news w2v
word2vec_load_format=False

max_records=100

train_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
train_data_mutate=True
dev_data=resources/roc_stories_data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json
test_data=resources/roc_stories_data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json

echo "============ settings ================"

# NN model settings
lstm_hidden_size=128
lstm_layers=1
batch_size=20
learn_rate=-0.01
min_word_freq=5
only_words_in_emb_vocab=False

num_epochs=10
train_embeddings=False
short_embeddings=True
attention_repr=True
lstm_out_repr=Tru
# --lstm_out_repr${lstm_out_repr}
dropout_prob_keep=1.0
layer_out_noise=0.00
loss_l2_beta=0.00
save_features=False

eval_every_steps=0
eval_every_epochs=1

#--eval_every_steps ${eval_every_steps} --eval_every_epochs ${eval_every_epochs}

# story reading settings
cond_answ_on_story=True
story_reads_num=1
cond_answ2_on_answ1=False
# --cond_answ_on_story ${cond_answ_on_story} --story_reads_num ${story_reads_num} --cond_answ2_on_answ1 ${cond_answ2_on_answ1}

run_name=${run_name}_l${lstm_layers}_hs${lstm_hidden_size}_ep${num_epochs}_bs${batch_size}_lr${learn_rate}_tre${train_embeddings}_she${short_embeddings}_att${attention_repr}_lstmr${lstm_out_repr}_cAS${cond_answ_on_story}_srn${story_reads_num}_cA2A1${cond_answ2_on_answ1}
run_name=${run_name}_$(date +%y-%m-%d-%H-%M-%S)

log_file=res_exp_${run_name}.log

print "Saving log to ${log_file}"

submission_data_eval="submission_${run_name}_test_spring2016.txt"
save_min_acc=0.30
save_submission_dev=False
# --save_min_acc ${save_min_acc} --save_submission_dev ${save_submission_dev}


model_dir=saved_models/${run_name}
output_dir=output/${run_name}
model_file=${model_dir}/${run_name}_logreg_model.pickle
scale_file=${model_dir}/${run_name}_scale_file.pickle

output_file=${run_name}_submission.txt

echo "=========TRAIN-eval========="
cmd=train-eval

# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

input_data_train=${train_data}
input_data_eval=${dev_data}
input_data_eval_test=${test_data}
#input_data_eval_test ${input_data_eval_test}
venv/bin/python story_cloze_v4_neural_lstm_attention.py --cmd ${cmd} --input_data_train ${input_data_train} --input_data_eval ${input_data_eval} --input_data_eval_test ${input_data_eval_test} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records} --tune ${tune} --submission_data_eval ${submission_data_eval} --lstm_hidden_size ${lstm_hidden_size} --lstm_layers ${lstm_layers} --batch_size ${batch_size}  --learn_rate ${learn_rate} --min_word_freq ${min_word_freq} --num_epochs ${num_epochs} --train_embeddings ${train_embeddings} --short_embeddings ${short_embeddings} --dropout_prob_keep ${dropout_prob_keep} --layer_out_noise ${layer_out_noise} --loss_l2_beta ${loss_l2_beta} --save_features ${save_features} --attention_repr ${attention_repr} --lstm_out_repr ${lstm_out_repr} --word2vec_load_format ${word2vec_load_format}  --cond_answ_on_story ${cond_answ_on_story} --story_reads_num ${story_reads_num} --cond_answ2_on_answ1 ${cond_answ2_on_answ1} --train_data_mutate ${train_data_mutate} --eval_every_steps ${eval_every_steps} --eval_every_epochs ${eval_every_epochs} --only_words_in_emb_vocab ${only_words_in_emb_vocab}  --save_min_acc ${save_min_acc} --save_submission_dev ${save_submission_dev}> ${log_file}
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
