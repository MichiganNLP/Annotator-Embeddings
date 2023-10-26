#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=36g
#SBATCH --output=/home/dnaihao/dnaihao-scratch/annotator-embedding/experiment-results/humor/embeddings_wo_weight/%x-%j.log
#SBATCH --job-name=humor
#SBATCH --account=linmacse0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dnaihao/dnaihao-scratch/anaconda3/envs/ann-embed/lib/
export CUDA_VISIBLE_DEVICES=0

model_name=bertmultichoice
dataset=humor
tasks=humor
broadcast_annotation_embedding=False
broadcast_annotator_embedding=False
include_pad_annotation=True
method=add
test_mode=normal
use_annotator_embed=False
use_annotation_embed=False
# split_method=annotation

train_batch_size=128
eval_batch_size=128

output_ckpt_dir=ckpts/${model_name}/text_finetuned/
num_train_epochs=3
wandb_name=bert_multi_choice-${dataset}
log_dir=logs/${model_name}/
log_path=logs/${model_name}/baseline.log
# train_data_path=example-data/${dataset}-processed/${split_method}_split_train_0.8.json
# dev_data_path=example-data/${dataset}-processed/${split_method}_split_test_0.8.json
# test_data_path=example-data/${dataset}-processed/${split_method}_split_test_0.8.json
model_name_or_path=bert-base-uncased
annotator_id_path=example-data/${dataset}-processed/stats.json
annotation_label_path=example-data/${dataset}-processed/annotation_labels.json
SEEDS=(32 42 52 62 72 82 92 102 112 122)
# SEEDS=(32 42 52)

mkdir -p ${output_ckpt_dir}
mkdir -p ${log_dir}

# for split_method in annotation annotator
for split_method in annotation
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

        # # annotation-split experiments
        # # question-only experiment
        # use_annotation_embed=False
        # use_annotator_embed=False
        # for i in {0..9}
        # do  
        #     seed=${SEEDS[$i]}
        #     output_fn=use_annotator_embed-${use_annotator_embed}-use_annotation_embed-${use_annotation_embed}-pad-${include_pad_annotation}-method-${method}-test_mode-${test_mode}-seed-$seed-$i-$split_method-$tt_idx            
        #     echo $output_fn
        #     pred_fn_path=../experiment-results/$dataset/bert-base-naiive-concat/$output_fn.jsonl
        #     if [ -e "$pred_fn_path" ]; then
        #         echo "File exists: $pred_fn_path"
        #     else
        #         python -m src ${model_name} \
        #             --train_data_path ${train_data_path} \
        #             --dev_data_path ${dev_data_path} \
        #             --test_data_path ${test_data_path} \
        #             --train_batch_size $train_batch_size \
        #             --eval_batch_size $eval_batch_size \
        #             --add_output_tokens True \
        #             --model_name_or_path ${model_name_or_path} \
        #             --output_ckpt_dir ${output_ckpt_dir} \
        #             --num_train_epochs ${num_train_epochs} \
        #             --wandb_name ${wandb_name} \
        #             --n_gpu 1 \
        #             --learning_rate 1e-5 \
        #             --linear_scheduler False \
        #             --wandb_offline \
        #             --training_paradigm learn_from_scratch \
        #             --tasks ${tasks} \
        #             --use_annotator_embed ${use_annotator_embed} \
        #             --use_annotation_embed ${use_annotation_embed} \
        #             --broadcast_annotation_embedding ${broadcast_annotation_embedding} \
        #             --broadcast_annotator_embedding ${broadcast_annotator_embedding} \
        #             --annotator_id_path ${annotator_id_path}\
        #             --annotation_label_path ${annotation_label_path} \
        #             --pred_fn_path ${pred_fn_path} \
        #             --include_pad_annotation ${include_pad_annotation} \
        #             --method ${method} \
        #             --test_mode ${test_mode} \
        #             --seed $seed \
        #             --use_naiive_concat
        #     fi
        # done
        # exit 0
        # annotation-embed
        use_annotation_embed=True
        use_annotator_embed=False
        for i in {0..9}
        do  
            seed=${SEEDS[$i]}
            output_fn=use_annotator_embed-${use_annotator_embed}-use_annotation_embed-${use_annotation_embed}-pad-${include_pad_annotation}-method-${method}-test_mode-${test_mode}-seed-$seed-$i-$split_method-$tt_idx                
            echo $output_fn
            pred_fn_path=../experiment-results/$dataset/embeddings_wo_weight/$output_fn.jsonl
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
                    --num_train_epochs ${num_train_epochs} \
                    --wandb_name ${wandb_name} \
                    --n_gpu 1 \
                    --learning_rate 1e-5 \
                    --linear_scheduler False \
                    --wandb_offline \
                    --training_paradigm learn_from_scratch \
                    --tasks ${tasks} \
                    --use_annotator_embed ${use_annotator_embed} \
                    --use_annotation_embed ${use_annotation_embed} \
                    --broadcast_annotation_embedding ${broadcast_annotation_embedding} \
                    --broadcast_annotator_embedding ${broadcast_annotator_embedding} \
                    --annotator_id_path ${annotator_id_path}\
                    --annotation_label_path ${annotation_label_path} \
                    --pred_fn_path ${pred_fn_path} \
                    --include_pad_annotation ${include_pad_annotation} \
                    --method ${method} \
                    --test_mode ${test_mode} \
                    --seed $seed \
                    --embed_wo_weight
            fi
        done

        # annotator-embed
        # restore the values
        use_annotation_embed=False
        use_annotator_embed=True
        for i in {0..9}
        do  
            seed=${SEEDS[$i]}
            output_fn=use_annotator_embed-${use_annotator_embed}-use_annotation_embed-${use_annotation_embed}-pad-${include_pad_annotation}-method-${method}-test_mode-${test_mode}-seed-$seed-$i-$split_method-$tt_idx                
            echo $output_fn
            pred_fn_path=../experiment-results/$dataset/embeddings_wo_weight/$output_fn.jsonl
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
                    --num_train_epochs ${num_train_epochs} \
                    --wandb_name ${wandb_name} \
                    --n_gpu 1 \
                    --learning_rate 1e-5 \
                    --linear_scheduler False \
                    --wandb_offline \
                    --training_paradigm learn_from_scratch \
                    --tasks ${tasks} \
                    --use_annotator_embed ${use_annotator_embed} \
                    --use_annotation_embed ${use_annotation_embed} \
                    --broadcast_annotation_embedding ${broadcast_annotation_embedding} \
                    --broadcast_annotator_embedding ${broadcast_annotator_embedding} \
                    --annotator_id_path ${annotator_id_path}\
                    --annotation_label_path ${annotation_label_path} \
                    --pred_fn_path ${pred_fn_path} \
                    --include_pad_annotation ${include_pad_annotation} \
                    --method ${method} \
                    --test_mode ${test_mode} \
                    --seed $seed \
                    --embed_wo_weight
            fi
        done


        # both-embed
        # restore the values
        use_annotation_embed=True
        use_annotator_embed=True
        for i in {0..9}
        do  
            seed=${SEEDS[$i]}
            output_fn=use_annotator_embed-${use_annotator_embed}-use_annotation_embed-${use_annotation_embed}-pad-${include_pad_annotation}-method-${method}-test_mode-${test_mode}-seed-$seed-$i-$split_method-$tt_idx                    
            echo $output_fn
            pred_fn_path=../experiment-results/$dataset/embeddings_wo_weight/$output_fn.jsonl
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
                    --num_train_epochs ${num_train_epochs} \
                    --wandb_name ${wandb_name} \
                    --n_gpu 1 \
                    --learning_rate 1e-5 \
                    --linear_scheduler False \
                    --wandb_offline \
                    --training_paradigm learn_from_scratch \
                    --tasks ${tasks} \
                    --use_annotator_embed ${use_annotator_embed} \
                    --use_annotation_embed ${use_annotation_embed} \
                    --broadcast_annotation_embedding ${broadcast_annotation_embedding} \
                    --broadcast_annotator_embedding ${broadcast_annotator_embedding} \
                    --annotator_id_path ${annotator_id_path}\
                    --annotation_label_path ${annotation_label_path} \
                    --pred_fn_path ${pred_fn_path} \
                    --include_pad_annotation ${include_pad_annotation} \
                    --method ${method} \
                    --test_mode ${test_mode} \
                    --seed $seed \
                    --embed_wo_weight
            fi
        done
    done
done