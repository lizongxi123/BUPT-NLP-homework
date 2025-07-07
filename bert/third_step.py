from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
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
tokenizer = BertTokenizer.from_pretrained("E:/natrual-language-processing/nlp数据集/bert-base-chinese")

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

# 6. 加载模型
model = BertForMaskedLM.from_pretrained("E:/natrual-language-processing/nlp数据集/bert/checkpoint-900")  # 使用之前训练的模型

# 7. 设置 Trainer（只用于推理）
training_args = TrainingArguments(
    output_dir="E:/natrual-language-processing/nlp数据集/bert",
    per_device_eval_batch_size=4,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
)

# 8. 使用 Trainer 进行评估
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=tokenized_dataset,
)

eval_results = trainer.evaluate()
print(f"平均 MLM Loss: {eval_results['eval_loss']:.4f}")
