# Hierarchical Attention Networks for Entity Linking

This is a  Pytorch Implementation of "*Hierarchical Attention Networks for Entity Linking*"

## Repo Structure

```
HAN-for-Entity-Linking
├── Data.py
├── Nets.py
├── main.py
├── log.txt
├── logs/
│   └── tensorboard_log_file
└── dataset/
    ├── train.pkl
    ├── test.pkl
    ├── word_dict.json
    ├── word_embedding.json
    ├── stop_word.txt
    └── entity_vectors.json

```

- `Data.py`: `DocumentDataset` (用于数据读取)和 `Vectorizer `（用于将文本向量化）
- `Nets.py`: 神经网络模型
- `main.py`: 训练参数设置等内容
- `log.txt`: 训练日志
- `dataset`: 实验对大部分原始数据文件做了处理，具体说明如下：
  - `train.pkl` : 将`documents_train.json`文件做了简单处理。字典结构完全不变，将每个document内容中的句号改为“句号 + 一个空格" 作为每个句子的结尾。做这个处理的本意是想使`spacy`的分句更加精确以正确取得`mention`对应的句子表征向量。实际上，`spacy`在处理后文件上的分句效果还是不够好，因此本实验最终使用字符串分割的方式进行分句和分词。
  - `test.pkl`：同`train.pkl`
  - `word_dict`: 由原始数据`word_info.txt`处理所得，结构为 `{"word": word_id}`
  - `word_embedding.json`:由原始数据`word_info.txt`所得，结构为`{word_id: 300-d word vector}`
  - `stopword.txt`：停用词，与原始数据集中的文件一致。
  - `entity_vectors`: 实体表征项链，处理后保存为`{entity_id: 300-d entity vector}`
  
  - **You can download the dataset zip at [this link](https://drive.google.com/file/d/1Lw6lBlih0pvoQOcJR1DHtJaF3c3MC1sO/view?usp=drive_link})**

## Environment

**hardware:** 

- **CPU:** `AMD Ryzen 5 5500U with Radeon Graphics 2.10 GHz`
- **time cost:** `0:13:43.45`

**requirements:**

- `Python 3.10`
- `torch 1.13.0`
- `tqdm 4.65.0`

## Result 

| Dataset | Best Accuracy(%) |
| ------- | ---------------- |
| test    | 75.5             |

## Settings

| **Hyper-parameter**s | Value |
| -------------------- | ----- |
| batch size           | 4     |
| learning rate        | 1e-5  |
| weight decay         | 1e-2  |
| epochs               | 60    |
| embedding size       | 300   |

## Run

```
python main.py
```

`pipline`描述如下（**以训练为例， 具体的实现细节请参照源代码注释**）：

- `train_dataset`按照以下格式提供数据：

  - `doc: 文档内容
  - `id`:文档id
  - `sent_idx`: 当前实体所属句子在文档中的index
  - `candidates`: 所有候选实体的tensor, shape = (候选实体个数， 300)
  - `target`: 标注实体在候选实体列表中的index

- 将一个batch的数据送入`collate_fn：tuple_batcher_builder`进行处理：

  - 将同类数据打包到一个元组中
  - 使用`vectorizer`对`doc`进行向量化， 并对短句`vector`做 `zero-padding`。
  - 计算`stats`（句子长度， 文档长度，batch， 句子索引）
  - 将所有处理好的数据返回用于训练

- 将`train_loader`返回的数据送入HAN：

  - 前向计算所有句子的表征向量和整个文档的表征向量`v_d`，并根据`sent_idx`取出所有`mention`所在句子的表征向量`v_dl`
  - 计算`sim_1 = sim(v_d, candidates)`和`sim_2 = sim(v_dl, candidates)`， 将`sim_1`和`sim_2`拼接后通过一个`Linear(2, 1)`得到`score`
  - 从`score`中取出`topk`的分数作为`label`， 根据`target`从`score`中取得标注实体的分数并将其复制`k`倍作为输出`out`

- 根据网络输出计算损失

  - 计算损失 = `MarginRankingLoss(out, label, torch.ones(out.shape[0]))`, 即优化目标是使标注实体的分数高于所有`topk`分数(实际取`k = 1`的效果最好)。

  - 反向传播更新参数

    
