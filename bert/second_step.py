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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 读取txt文件，一行一条文本
txt_path = "E:/natrual-language-processing/nlp数据集/bert/混合数据.txt"
with open(txt_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# 转成 datasets.Dataset 格式，假设文本键名为 "fact"
dataset = Dataset.from_dict({"fact": lines})

# 随机划分，先抽取15%为test集
split_ratio = 0.15
total_size = len(dataset)
indices = list(range(total_size))
random.seed(random_seed)
random.shuffle(indices)

test_size = int(total_size * split_ratio)
test_indices = indices[:test_size]
train_val_indices = indices[test_size:]

test_dataset = dataset.select(test_indices)
train_val_dataset = dataset.select(train_val_indices)

# 再划分训练和验证（10%作为验证）
train_val_split = train_val_dataset.train_test_split(test_size=0.1, seed=random_seed)
train_dataset = train_val_split["train"]
eval_dataset = train_val_split["test"]

print(f"训练集条数：{len(train_dataset)}，验证集条数：{len(eval_dataset)}，测试集条数：{len(test_dataset)}")

# 2. 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained("E:/natrual-language-processing/nlp数据集/bert-base-chinese")

# 3. 编码（tokenize），注意文本键为"fact"
def tokenize_function(examples):
    return tokenizer(examples["fact"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# 4. 设置 data collator（用于动态 mask）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# 5. 加载模型
model = BertForMaskedLM.from_pretrained("E:/natrual-language-processing/nlp数据集/bert-base-chinese")
model.to(device)

# 6. 设置 Trainer（用于训练和评估）
training_args = TrainingArguments(
    output_dir="E:/natrual-language-processing/nlp数据集/bert",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_dir="./logs",
    logging_steps=300,
    do_train=True,
    do_eval=True,
    save_total_limit=2,
)

# 7. 使用 Trainer 进行训练和评估
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print(f"当前使用的设备：{'GPU（' + torch.cuda.get_device_name(0) + '）' if torch.cuda.is_available() else 'CPU'}")

trainer.train()
eval_results = trainer.evaluate()
print(f"平均 MLM Loss: {eval_results['eval_loss']:.4f}")
