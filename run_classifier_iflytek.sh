CURRENT_DIR=`pwd`
export GLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="iflytek"

python run_classifier.py \
  --model_type=albert \
  --model_name_or_path=voidful/albert_chinese_tiny \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --evaluate_during_training \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-4 \
  --num_train_epochs=6.0 \
  --logging_steps=759 \
  --save_steps=759 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

python run_classifier.py \
  --model_type=roberta \
  --model_name_or_path=hfl/chinese-roberta-wwm-ext \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --evaluate_during_training \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-4 \
  --num_train_epochs=6.0 \
  --logging_steps=759 \
  --save_steps=759 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

python run_classifier.py \
  --model_type=xlnet \
  --model_name_or_path=hfl/chinese-xlnet-base \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --evaluate_during_training \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-4 \
  --num_train_epochs=6.0 \
  --logging_steps=759 \
  --save_steps=759 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
#
python run_classifier.py \
  --model_type=bert \
  --model_name_or_path=bert-base-chinese \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --evaluate_during_training \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-4 \
  --num_train_epochs=6.0 \
  --logging_steps=759 \
  --save_steps=759 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42