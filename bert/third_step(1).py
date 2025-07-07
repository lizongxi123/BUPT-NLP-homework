from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification,DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch
import random
import numpy as np

# 固定随机种子
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# 1. 加载数据集
data_dir = "E:/natrual-language-processing/nlp数据集/数据集/wikipedia-cn-20230720-filtered-train.arrow"
dataset = Dataset.from_file(data_dir)

# 2. 随机抽取10%数据
dataset = dataset.train_test_split(test_size=0.1, seed=random_seed)["test"]
print(f"抽取用于测试的条数：{len(dataset)}")

# 3. 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained("E:/natrual-language-processing/nlp数据集/bert/saved-mlm-model/final")

# 4. 编码（tokenize）
def tokenize_function(example):
    return tokenizer(example["completion"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 5. 设置 data collator（用于动态 mask）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

import torch
from transformers import BertForSequenceClassification, BertForMaskedLM, BertConfig

# 1. 初始化 cls_model（结构必须与保存时一致）
cls_model = BertForSequenceClassification.from_pretrained("E:/natrual-language-processing/nlp数据集/bert/saved-mlm-model/final", num_labels=15)

# 2. 加载你保存的权重文件（state_dict 格式）
state_dict = torch.load("E:/natrual-language-processing/nlp数据集/bert/bert_multitask.pth", map_location="cpu")

# 3. 加载权重到 cls_model
cls_model.load_state_dict(state_dict, strict=False)

# 4. 初始化 MLM 模型
mlm_model = BertForMaskedLM.from_pretrained("E:/natrual-language-processing/nlp数据集/bert/saved-mlm-model/final")

# 5. 将主干（bert）部分权重从分类模型复制到 MLM 模型
mlm_model.bert.load_state_dict(cls_model.bert.state_dict(), strict=False)

print("✅ 模型主干参数已复制到新的 MLM 模型中")
# 7. 设置 Trainer（只用于推理）
training_args = TrainingArguments(
    output_dir="./tmp-cls-eval",
    per_device_eval_batch_size=4,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
)

# 8. 使用 Trainer 进行评估
trainer = Trainer(
    model=mlm_model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=tokenized_dataset,
)

eval_results = trainer.evaluate()
print(f"平均 MLM Loss: {eval_results['eval_loss']:.4f}")
