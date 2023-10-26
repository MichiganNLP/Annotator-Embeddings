#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=36g
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embedding/experiment-results/commitmentbank/%x-%j.log
#SBATCH --job-name=commitmentbank
#SBATCH --account=mihalcea98

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dnaihao/dnaihao-scratch/anaconda3/envs/ann-embed/lib/
# export CUDA_VISIBLE_DEVICES=0

model_name=random
broadcast_annotation_embedding=False
broadcast_annotator_embedding=False
include_pad_annotation=True
method=add
test_mode=random
use_annotator_embed=False
use_annotation_embed=False
# split_method=annotation

train_batch_size=64
eval_batch_size=64

output_ckpt_dir=ckpts/${model_name}/text_finetuned/
num_train_epochs=3
wandb_name=random-${dataset}
log_dir=logs/${model_name}/
log_path=logs/${model_name}/baseline.log
# train_data_path=example-data/${dataset}-processed/${split_method}_split_train_0.8.json
# dev_data_path=example-data/${dataset}-processed/${split_method}_split_test_0.8.json
# test_data_path=example-data/${dataset}-processed/${split_method}_split_test_0.8.json
model_name_or_path=bert-base-uncased
SEEDS=(32 42 52 62 72 82 92 102 112 122)

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}


datasets=("berkeley-hate-speech" "commitmentbank" "friends_qia" "goemotions" "hs_brexit" "humor" "md-agreement" "pejorative" "sentiment")
all_tasks=("hate_speech" "certainty" "indirect_ans" "emotion" "hs_brexit" "humor" "offensive" "pejorative" "sentiment")
dataset=berkeley-hate-speech
tasks=hate_speech

for dataset_i in {0..9}
do 
    dataset=${datasets[$dataset_i]}
    tasks=${all_tasks[$dataset_i]}

    annotator_id_path=example-data/${dataset}-processed/stats.json
    annotation_label_path=example-data/${dataset}-processed/annotation_labels.json
    for split_method in annotation annotator
    # for split_method in annotation
    do  
        train_data_paths=(example-data/${dataset}-processed/${split_method}_split_train.json \
                    example-data/${dataset}-processed/${split_method}_split_train_0.8.json \
                    example-data/${dataset}-processed/${split_method}_split_train_0.7.json \
                    example-data/${dataset}-processed/${split_method}_split_train_0.6.json \
                    example-data/${dataset}-processed/${split_method}_split_train_0.8_smaller.json \
                    example-data/${dataset}-processed/${split_method}_split_train_0.7_smaller.json \
                    example-data/${dataset}-processed/${split_method}_split_train_0.6_smaller.json)
        dev_data_paths=(example-data/${dataset}-processed/${split_method}_split_test.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.8.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.7.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.6.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.8_smaller.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.7_smaller.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.6_smaller.json)
        test_data_paths=(example-data/${dataset}-processed/${split_method}_split_test.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.8.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.7.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.6.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.8_smaller.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.7_smaller.json \
                    example-data/${dataset}-processed/${split_method}_split_test_0.6_smaller.json)
        for tt_idx in 0
        do  
            train_data_path=${train_data_paths[$tt_idx]}
            dev_data_path=${dev_data_paths[$tt_idx]}
            test_data_path=${test_data_paths[$tt_idx]}

            for i in {0..9}
            do  
                seed=${SEEDS[$i]}
                output_fn=use_annotator_embed-${use_annotator_embed}-use_annotation_embed-${use_annotation_embed}-pad-${include_pad_annotation}-method-${method}-test_mode-${test_mode}-seed-$seed-$i-$split_method-$tt_idx            
                echo $output_fn
                pred_fn_path=../experiment-results/$dataset/$output_fn.jsonl
                if [ -e "$pred_fn_path" ]; then
                    echo "File exists: $pred_fn_path"
                else
                    python -m src ${model_name} \
                        --train_data_path ${train_data_path} \
                        --dev_data_path ${dev_data_path} \
                        --test_data_path ${test_data_path} \
                        --train_batch_size $train_batch_size \
                        --eval_batch_size $eval_batch_size \
                        --add_output_tokens True \
                        --model_name_or_path ${model_name_or_path} \
                        --output_ckpt_dir ${output_ckpt_dir} \
                        --training_paradigm learn_from_scratch \
                        --tasks ${tasks} \
                        --annotator_id_path ${annotator_id_path}\
                        --annotation_label_path ${annotation_label_path} \
                        --pred_fn_path ${pred_fn_path} \
                        --method "add" \
                        --n_gpu 0 \
                        --wandb_name $wandb_name \
                        --seed $seed
                fi
            done
        done
    done
done