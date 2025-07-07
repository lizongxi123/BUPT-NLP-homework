from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 固定随机种子
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# 1. 读取并解析txt数据
data_path = "E:/natrual-language-processing/nlp数据集/toutiao_cat_data.txt"
texts = []
labels = []
label2id = {}
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("_!_")
        if len(parts) < 4:
            continue
        news_id = parts[0]
        label = parts[1]
        text = parts[3]
        # 构建标签映射
        if label not in label2id:
            label2id[label] = len(label2id)
        texts.append(text)
        labels.append(label2id[label])

print(texts[:5])  # 打印前5条文本
print(labels[:5])  # 打印前5条标签

# 构建数据集
dataset = Dataset.from_dict({"text": texts, "label": labels})

#随机抽取20%数据
dataset = dataset.train_test_split(test_size=0.1, seed=random_seed)["test"]
print(f"抽取用于测试的条数：{len(dataset)}")

# 划分训练和验证集
dataset = dataset.train_test_split(test_size=0.2, seed=random_seed)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 2. 加载tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 3. 编码
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 4. 加载分类模型
num_labels = len(label2id)
print(f"标签数量: {num_labels}")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./tmp-cls",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_dir="./logs",
    logging_steps=500,
    do_train=True,
    do_eval=True,
    save_total_limit=2,
)

# 6. 定义评价指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# 7. 训练与评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

if torch.cuda.is_available():
    print(f"当前使用的设备：GPU（{torch.cuda.get_device_name(0)}）")
else:
    print("当前使用的设备：CPU")

trainer.train()
eval_results = trainer.evaluate()
print(f"准确率: {eval_results['eval_accuracy']:.4f}, F1: {eval_results['eval_f1']:.4f}")