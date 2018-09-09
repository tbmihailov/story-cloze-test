#!/usr/bin/env bash

resources_dir=~/research/resources
embeddings_files=(
"${resources_dir}/glove/glove.6B.50d.txt.gensim.bin"
"${resources_dir}/glove/glove.6B.100d.txt.gensim.bin"
"${resources_dir}/glove/glove.6B.200d.txt.gensim.bin"
"${resources_dir}/glove/glove.6B.300d.txt.gensim.bin"
"${resources_dir}/glove/glove.840B.300d.txt.gensim.bin"
"${resources_dir}/conceptnet_vectors/conceptnet-numberbatch-201609_en_full.txt.gensim.bin"
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

datasets_train=(
data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json.split_90.json
data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-random10-2017-02-20-16-47-v01.json.easy6.txt
data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_top10_fx_17-02-21-00-56-18.json.easy6.txt
data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v2_top500_rnd10_17-02-21-02-06-28.json.easy6.txt
)

# Data generated with nearest ending with same NN or PR
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy7.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy6.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy5.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy4.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy3.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy2.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy1.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy0.txt
# data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json

#rc1617.es7
#rc1617.es6
#rc1617.es5
#rc1617.es4
#rc1617.es3
#rc1617.es2
#rc1617.es1
#rc1617.es0
#rc1617.rnd



datasets_train_names=(
sct16val90
sct2016val
rc1617rnd10v01
rc1617postop10fx
rc1617postop500rnd10
)

batch_size_list=(
50
100
200
500
)

hidden_size_list=(
128
256
384
512
1024
)

i=6 # Word2vec
END=7 # 7 is the max embedding
for ((i=6;i<END;i++)); do
    num_epochs=1
    num_runs=3 # 9

    loop_emb_file=${embeddings_files[$i]};
    loop_emb_name=${embeddings_run_names[$i]}
    loop_emb_binary=${embeddings_load_binary[$i]}

    run_id=0
    #for ((run_id=0;run_id<num_runs;run_id++)); do
    for ((run_id_it=0;run_id_it<num_runs;run_id_it++)); do

        for ((run_id=2;run_id<5;run_id++)); do
            for ((run_hs=2;run_hs<3;run_hs++)); do
                run_name=story_cloze_v4_lstm_att_gen_emb${loop_emb_name}
                #model dir where output models are saved after train

                scale_features=True
                tune=True
                emb_model_type=w2v # rnd
                # emb_model_file=resources/word2vec/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
                emb_model_file=${loop_emb_file}
                word2vec_load_bin=${loop_emb_binary} # True for google news w2v
                word2vec_load_format=True

                max_records=0

                train_data_mutate=False
                train_data=${datasets_train[$run_id]}
                train_data_name=${datasets_train_names[$run_id]}

        #        train_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val-mutated-smart1.tsv.json
        #        train_data_name=sp2016valmut1
                # train_data=resources/roc_stories_data/processed_pos_ROCStories__spring2016-ROC-Stories-naacl-camera-ready-train-random.tsv.json

        #        dev_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
        #        dev_data_name=sp16d

#                dev_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json.split_10.json
#                dev_data_name=sp16val10
                 dev_data=resources/roc_stories_data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
                 dev_data_name=sp16d

        #        dev_data=resources/roc_stories_data/processed_pos_ROCStories__spring2016-ROC-Stories-naacl-camera-ready-train-random.tsv.json
        #        dev_data_submission_name="roc_random_spring2016"

                test_data=resources/roc_stories_data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json
                test_data_name=sp16t

                run_name=${run_name}_tr.${train_data_name}_d.${dev_data_name}_te.${test_data_name}

                echo "=========Settings==========="
                # NN model settings
                lstm_hidden_size=384 # ${hidden_size_list[$run_hs]}  # 384
                lstm_layers=1
                batch_size=50 # ${batch_size_list[$run_bs]}  # 50
                eval_every_steps=50
                eval_every_epochs=1

                learn_rate=0.001 # < 0 for adaptive, e.g. -1.0
                min_word_freq=3
                only_words_in_emb_vocab=False
                train_embeddings=False

                short_embeddings=False
                attention_repr=True
                lstm_out_repr=True

                dropout_prob_keep=1.0
                layer_out_noise=0.00
                loss_l2_beta=0.00
                save_features=False

                # story reading settings
                cond_answ_on_story=True
                story_reads_num=1
                cond_answ2_on_answ1=False
                # --cond_answ_on_story ${cond_answ_on_story} --story_reads_num ${story_reads_num} --cond_answ2_on_answ1 ${cond_answ2_on_answ1}

                run_name=${run_name}_l${lstm_layers}_hs${lstm_hidden_size}_ep${num_epochs}_bs${batch_size}_lr${learn_rate}_te${train_embeddings}_she${short_embeddings}_at${attention_repr}_lstm${lstm_out_repr}
                # _cAS${cond_answ_on_story}_srn${story_reads_num}_cA2A1${cond_answ2_on_answ1}
                run_name=${run_name}_$(date +%y-%m-%d-%H-%M-%S)
                log_file=res_exp_${run_name}.log

                print "Saving log to ${log_file}"
                submission_data_eval="subm_${run_name}.txt"

                save_min_acc=0.25
                save_submission_dev=True
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

                print "Saving log to ${log_file}"

                input_data_train=${train_data}
                input_data_eval=${dev_data}
                input_data_eval_test=${test_data}

                venv/bin/python story_cloze_v4_neural_lstm_attention.py --cmd ${cmd} --input_data_train ${input_data_train} --input_data_eval ${input_data_eval} --input_data_eval_test ${input_data_eval_test} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records} --tune ${tune} --submission_data_eval ${submission_data_eval} --lstm_hidden_size ${lstm_hidden_size} --lstm_layers ${lstm_layers} --batch_size ${batch_size}  --learn_rate ${learn_rate} --min_word_freq ${min_word_freq} --num_epochs ${num_epochs} --train_embeddings ${train_embeddings} --short_embeddings ${short_embeddings} --dropout_prob_keep ${dropout_prob_keep} --layer_out_noise ${layer_out_noise} --loss_l2_beta ${loss_l2_beta} --save_features ${save_features} --attention_repr ${attention_repr} --lstm_out_repr ${lstm_out_repr} --word2vec_load_format ${word2vec_load_format}  --cond_answ_on_story ${cond_answ_on_story} --story_reads_num ${story_reads_num} --cond_answ2_on_answ1 ${cond_answ2_on_answ1} --train_data_mutate ${train_data_mutate} --only_words_in_emb_vocab ${only_words_in_emb_vocab}  --save_min_acc ${save_min_acc} --save_submission_dev ${save_submission_dev} --eval_every_steps ${eval_every_steps} --eval_every_epochs ${eval_every_epochs}> ${log_file}
            done
        done
    done
done
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
