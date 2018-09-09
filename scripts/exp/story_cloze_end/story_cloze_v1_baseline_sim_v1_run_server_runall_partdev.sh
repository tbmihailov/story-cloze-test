#!/usr/bin/env bash

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

max_train_items=(
50
100
200
300
400
500
600
700
800
900
1000
1100
1200
1300
1400
1500
1600
1700
1800
1900
)

# For in files
#for file in ${embeddings_files[@]}; do
#	echo "Running with embeddings file $file"
#done

echo "Test for scirpt..."
END=7
for ((i=6;i<END;i++)); do
    loop_emb_file=${embeddings_files[$i]}
    loop_emb_name=${embeddings_run_names[$i]}
    loop_emb_binary=${embeddings_load_binary[$i]}
    echo "Running for i=$i file ${loop_emb_file} ${loop_emb_name} binary=${loop_emb_binary}"
done

param_c_tune_list=(
0.1
0.2
0.3
0.4
0.5
1.0
2.0
3.0
5.0
10.0
20.0
)

datasets_train=(
data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy7.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy6.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy5.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy4.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy3.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy2.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy1.txt
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-nearest-pr-nn-2017-02-04-21-25.json.easy0.txt
)

datasets_train_names=(
sct16val
rc1617.es7
rc1617.es6
rc1617.es5
rc1617.es4
rc1617.es3
rc1617.es2
rc1617.es1
rc1617.es0
)

##############################
############DEV##############
##############################
data_dev_id=0

datasets_dev=(
data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json
)

datasets_dev_names=(
sct16val
sct16test
)

##############################
############TEST##############
##############################
data_test_id=1

datasets_test=(
data/processed_pos_cloze_test_val__spring2016-cloze_test_ALL_val.tsv.json
data/processed_pos_cloze_test_test__spring2016-cloze_test_ALL_test.tsv.json
data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_10_17-02-06-15-44-21.json
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-random50.json
data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_20_17-02-20-12-01-35.json
data/generated_generate_train_random_proc_pos_ROCStories__ROC-Stories-2016-2017-random10-2017-02-20-16-47-v01.json
data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v1_top10_fx_17-02-21-00-56-18.json
data/generated_proc_pos_ROCStories_ROC-16-17-mutate_rocstories_data_pos_v2_top500_rnd10_17-02-21-02-06-28.json
)

datasets_test_names=(
sct16val
sct16test
rc1617pos10
rc1617rnd50
rc1617alsentp20
rc1617rnd10v01
rc1617prnnt10fx
rc1617prnntp500r10
)

param_c_tune_len=11

# exit
echo "------------------------"
echo "--Running Experiments---"
echo "------------------------"

END=7
for ((pc_i=0;pc_i<1;pc_i++)); do
    param_c_curr=${param_c_tune_list[$pc_i]}
    for ((i=0;i<20;i++)); do
        for ((j=0;j<1;j++)); do
            loop_emb_file=${embeddings_files[6]}
            loop_emb_name=${embeddings_run_names[6]}
            loop_emb_binary=${embeddings_load_binary[6]}
            echo "Running for i=$i file ${loop_emb_file} ${loop_emb_name} binary=${loop_emb_binary}"

            run_name=storycloze_bl_sim_v1_${loop_emb_name}
            #model dir where output models are saved after train

            scale_features=True
            tune=False
            param_c=0.3 # ${param_c_curr}

            emb_model_type=w2v
            # emb_model_file=resources/word2vec/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
            # emb_model_file=resources/word2vec/GoogleNews-vectors-negative300.bin
            emb_model_file=${loop_emb_file}
            word2vec_load_bin=${loop_emb_binary} # True for google news w2v
            word2vec_load_format=True # True for word23vec.load_word2vec_format
            load_features=False

            lower_tokens=True
            remove_stopwords=True
            inc_embedd_vectors=True
            inc_slast1=False
            inc_slast2=False
            inc_slast3=False
            inc_slast4=False
            inc_story=True
            inc_maxsim=True
            inc_possim=True
            inc_fullsims=True
            include_elem_multiply=False

            run_setup=sl1${inc_slast1}_sl2${inc_slast2}_sl3${inc_slast3}_sl4${inc_slast4}_st${inc_story}_ms${inc_maxsim}_${inc_possim}_${inc_fullsims}_mul${include_elem_multiply}_c${param_c}

            max_records=${max_train_items[$i]}

            train_data=${datasets_train[$j]}
            train_data_name=${datasets_train_names[$j]}

            dev_data=${datasets_dev[${data_dev_id}]}
            dev_data_name=${datasets_dev_names[${data_dev_id}]}

            test_data=${datasets_test[${data_test_id}]}
            test_data_name=${datasets_test_names[${data_test_id}]}

            run_name=${run_name}_tr.${train_data_name}_te.${test_data_name}_max${max_records}

            input_data_train=${train_data}
            input_data_eval=${test_data}

            run_name=${run_name}_lf${load_features}_tune${tune}${run_setup}
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

            log_file=res_exp_${run_name}.log
            print "Saving log to ${log_file}"


            venv/bin/python story_cloze_v1_features_and_similarities.py --cmd ${cmd} --input_data_train ${input_data_train} --input_data_eval ${input_data_eval} --output_file ${output_file} --model_file ${model_file} --scale_file ${scale_file} --scale_features ${scale_features} --emb_model_type ${emb_model_type} --emb_model_file ${emb_model_file} --word2vec_load_bin ${word2vec_load_bin} --output_dir ${output_dir} --run_name ${run_name} --max_records ${max_records} --tune ${tune} --submission_data_eval ${submission_data_eval} --load_features ${load_features}  --word2vec_load_format ${word2vec_load_format}  --lower_tokens ${lower_tokens}  --remove_stopwords ${remove_stopwords}  --inc_embedd_vectors ${inc_embedd_vectors}  --inc_slast1 ${inc_slast1}  --inc_slast2 ${inc_slast2}  --inc_slast3 ${inc_slast3}  --inc_slast4 ${inc_slast4}  --inc_story ${inc_story}  --inc_maxsim ${inc_maxsim}  --inc_possim ${inc_possim}  --inc_fullsims ${inc_fullsims} --include_elem_multiply ${include_elem_multiply} --param_c ${param_c}> ${log_file}

            # exit
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
        done
    done
done
