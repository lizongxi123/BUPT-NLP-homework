from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import torch
import random
import numpy as np
import os

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
print(f"抽取用于训练/评估的条数：{len(dataset)}")

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
model = BertForMaskedLM.from_pretrained("E:/natrual-language-processing/nlp数据集/bert-base-chinese")

# ✅ 7. 设置 TrainingArguments，启用训练 & 自动保存模型
training_args = TrainingArguments(
    output_dir="./saved-mlm-model",               # ✅ 模型保存路径
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,                           # ✅ 训练轮数（可调）
    evaluation_strategy="epoch",                  # 每个 epoch 做一次评估
    save_strategy="epoch",                        # 每个 epoch 自动保存一次模型
    logging_dir="./logs",
    logging_steps=50,
    do_train=True,                                # ✅ 启用训练
    do_eval=True,                                 # ✅ 启用评估
    save_total_limit=1,                           # 最多保存一个模型（可改）
    load_best_model_at_end=True,                  # 根据 eval_loss 自动加载最好的
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# 8. 使用 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,     # ✅ 训练集和评估集相同（你只有10%样本）
    eval_dataset=tokenized_dataset,
)

# ✅ 9. 开始训练并评估
trainer.train()

# ✅ 10. 手动保存最终模型（可选）
trainer.save_model("./saved-mlm-model/final")
tokenizer.save_pretrained("./saved-mlm-model/final")

print("✅ 训练完成，模型已保存至 './saved-mlm-model/final'")
