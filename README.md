## CLUE_pytorch

中文语言理解测评基准(Language Understanding Evaluation benchmark for Chinese)

**备注**：此版本为个人开发版(目前支持所有的分类型任务)，正式版见https://github.com/CLUEbenchmark/CLUE

## 更新

* **2020-03-08**: 模型加载使用[Huggingface-Transformers](https://github.com/huggingface/transformers)

## 模型列表

|      model type       |                      model_name_or_path                      |
| :-------------------: | :----------------------------------------------------------: |
|        albert         | [voidful/albert_chinese_base](https://huggingface.co/voidful/albert_chinese_base) |
|        albert         | [voidful/albert_chinese_larg](https://huggingface.co/voidful/albert_chinese_large) |
|        albert         | [`voidful/albert_chinese_small`](https://huggingface.co/voidful/albert_chinese_small) |
|        albert         | [`voidful/albert_chinese_tiny`](https://huggingface.co/voidful/albert_chinese_tiny) |
|        albert         | [`voidful/albert_chinese_xlarge`](https://huggingface.co/voidful/albert_chinese_xlarge) |
|        albert         | [`voidful/albert_chinese_xxlarge`](https://huggingface.co/voidful/albert_chinese_xxlarge) |
|         bert          | [`bert-base-chinese`](https://huggingface.co/bert-base-chinese) |
|     bert-wwm-ext      | [`hfl/chinese-bert-wwm-ext`](https://huggingface.co/hfl/chinese-bert-wwm-ext) |
|       bert-wwm        | [`hfl/chinese-bert-wwm`](https://huggingface.co/hfl/chinese-bert-wwm) |
| roberta-wwm-ext-large | [`hfl/chinese-roberta-wwm-ext-large`](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) |
|    roberta-wwm-ext    | [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |
|      xlnet-base       | [`hfl/chinese-xlnet-base`](https://huggingface.co/hfl/chinese-xlnet-base) |
|       xlnet-mid       | [`hfl/chinese-xlnet-mid`](https://huggingface.co/hfl/chinese-xlnet-mid) |
|         rbt3          |        [`hfl/rbt3`](https://huggingface.co/hfl/rbt3)         |
|         rbt3          |       [`hfl/rbtl3`](https://huggingface.co/hfl/rbtl3)        |
| RoBERTa-tiny-clue      | [`clue/roberta_chinese_clue_tiny`](https://huggingface.co/clue/roberta_chinese_clue_tiny) |
| RoBERTa-tiny-pair      | [`clue/roberta_chinese_pair_tiny`](https://huggingface.co/clue/roberta_chinese_pair_tiny) |
| RoBERTa-tiny3L768-clue | [`clue/roberta_chinese_3L768_clue_tiny`](https://huggingface.co/clue/roberta_chinese_3L768_clue_tiny) |
| RoBERTa-tiny3L312-clue | [`clue/roberta_chinese_3L312_clue_tiny`](https://huggingface.co/clue/roberta_chinese_3L312_clue_tiny) |
| RoBERTa-large-clue    | [`clue/roberta_chinese_clue_large`](https://huggingface.co/clue/roberta_chinese_clue_large) |
| RoBERTa-large-pair     | [`clue/roberta_chinese_pair_large`](https://huggingface.co/clue/roberta_chinese_pair_large) |

## 代码目录说明

```text
├── CLUEdatasets   #　存放数据
|  └── tnews　　　
|  └── wsc　
|  └── ...
├── metrics　　　　　　　　　# metric计算
|  └── clue_compute_metrics.py　　　
├── outputs              # 模型输出保存
|  └── tnews_output
|  └── wsc_output　
|  └── ...
├── prev_trained_model　# 预训练模型
|  └── albert_base
|  └── bert-wwm
|  └── ...
├── processors　　　　　# 数据处理
|  └── clue.py
|  └── ...
├── tools　　　　　　　　#　通用脚本
|  └── progressbar.py
|  └── ...
├── run_classifier.py       # 主程序
├── run_classifier_tnews.sh   #　任务运行脚本
```
### 依赖模块

- pytorch=1.1.0
- boto3=1.9
- regex
- sacremoses
- sentencepiece
- python3.6+
- transformers=2.5.1

### 运行方式
**1. 安装Transformers**
```shell
pip install transformers
```

**2. 下载CLUE数据集，运行以下命令：**
```python
python download_clue_data.py --data_dir=./CLUEdatasets --tasks=all
```
上述命令默认下载全CLUE数据集，你也可以指定`--tasks`进行下载对应任务数据集，默认存在在`./CLUEdatasets/{对应task}`目录下。

**注意**: 如果使用本地已经下载好的模型权重，需要在对应的文件夹内存放`config.json`和`vocab.txt`文件，比如：
```text
├── prev_trained_model　# 预训练模型
|  └── bert-base
|  | └── vocab.txt
|  | └── config.json
|  | └── pytorch_model.bin

```
如果使用本地已有的模型权重文件，直接修改参数`--model_name_or_path=your_local_model_weight_path`即可

**3. 直接运行对应任务sh脚本，如：**

```shell
sh run_classifier_tnews.sh
```
具体运行方式如下：
```python
CURRENT_DIR=`pwd`
export CLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="iflytek"

python run_classifier.py \
  --model_type=albert \
  --model_name_or_path=voidful/albert_chinese_tiny \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --evaluate_during_training \
  --data_dir=$CLUE_DIR/${TASK_NAME}/ \
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
```
**注意**:

> model_name_or_path=voidful/albert_chinese_tiny默认自动下载albert_chinese_tiny
>当前只支持google版本的中文albert模型


**4. 评估**

当前默认使用最后一个checkpoint模型作为评估模型，你也可以指定`--predict_checkpoints`参数进行对应的checkpoint进行评估，比如：
```python
CURRENT_DIR=`pwd`
export CLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="copa"

python run_classifier.py \
  --model_type=bert \
  --model_name_or_path=voidful/albert_chinese_tiny \
  --task_name=$TASK_NAME \
  --do_predict \
  --predict_checkpoints=100 \
  --do_lower_case \
  --data_dir=$CLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=2.0 \
  --logging_steps=50 \
  --save_steps=50 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
```

### 模型列表
```
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "ernie": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "roberta": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, BertTokenizer),
```
### 结果

CLUEWSC2020: WSC Winograd模式挑战中文版,新版2020-03-25发布 [CLUEWSC2020数据集下载](https://storage.googleapis.com/cluebenchmark/tasks/cluewsc2020_public.zip)

| 模型 | 开发集(Dev) |
| :------- | :---------: |
| bert_base | 79.94 |




