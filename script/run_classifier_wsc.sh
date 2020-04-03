CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export CLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="wsc"

python run_classifier.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --evaluate_during_training \
  --data_dir=$CLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=50.0 \
  --logging_steps=78 \
  --save_steps=78 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

#python run_classifier.py \
#  --model_type=roberta \
#  --model_name_or_path=hfl/chinese-roberta-wwm-ext \
#  --task_name=$TASK_NAME \
#  --do_train \
#  --do_lower_case \
#  --evaluate_during_training \
#  --data_dir=$CLUE_DIR/${TASK_NAME}/ \
#  --max_seq_length=128 \
#  --per_gpu_train_batch_size=16 \
#  --per_gpu_eval_batch_size=16 \
#  --learning_rate=2e-4 \
#  --num_train_epochs=6.0 \
#  --logging_steps=78 \
#  --save_steps=78 \
#  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
#  --overwrite_output_dir \
#  --seed=42
