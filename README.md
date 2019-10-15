# chineseGLUE_pytorch


**详细信息见于https://github.com/chineseGLUE/chineseGLUE**

## 代码目录说明
```text
├── chineseGLUEdatasets   #　存放数据
|  └── inews　　　
|  └── lcqmc　
|  └── ...
├── metrics　　　　　　　　　# metric计算
|  └── glue_compute_metrics.py　　　
├── outputs              # 模型输出保存
|  └── inews_output
|  └── lcqmc_output　
|  └── ...
├── prev_trained_model　# 预训练模型
|  └── albert_base
|  └── bert-wwm
|  └── ...
├── processors　　　　　# 数据处理
|  └── glue.py
|  └── ...
├── tools　　　　　　　　#　通用脚本
|  └── progressbar.py
|  └── ...
├── transformers　　　# 模型
|  └── modeling_albert.py
|  └── modeling_bert.py
|  └── ...
├── convert_albert_original_tf_checkpoint_to_pytorch.py　#　模型文件转换
├── run_chineseglue.py       # 主程序
├── run_classifier_inews.sh   #　任务运行脚本
```

若下载对应tf模型权重，则运行转换脚本，比如转换`albert_base_tf`:
```python
python convert_albert_original_tf_checkpoint_to_pytorch.py \
      --tf_checkpoint_path=./prev_trained_model/albert_base_tf \
      --bert_config_file=./prev_trained_model/albert_base_tf/albert_config_base.json \
      --pytorch_dump_path=./prev_trained_model/albert_base/pytorch_model.bin
```
**注意**: 当转换完模型之后，需要在对应的文件夹内存放`config.json`和`vocab.txt`文件

## 测试结果

### Tnews文本分类任务

    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | | | 	||

### Xlni自然语言推理

    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    |  | | 	||

### Lcqmc语义相似度匹配

    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    |  | | 	||

### Inews 互联网情感分析

    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    |  | | 	||

