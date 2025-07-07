import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_scheduler,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from datasets import Dataset
import random
import numpy as np

# 设随机种子，保证可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 联合模型定义
class BertForMultiTask(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.cls_head = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                task_type="classification", labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        if task_type == "classification":
            logits = self.cls_head(outputs.pooler_output)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        elif task_type == "mlm":
            logits = self.mlm_head(outputs.last_hidden_state)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained("E:/natrual-language-processing/nlp数据集/bert/saved-mlm-model/final")

# 加载分类数据
cls_path = "E:/natrual-language-processing/nlp数据集/toutiao_cat_data.txt"
texts, labels, label2id = [], [], {}
with open(cls_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("_!_")
        if len(parts) >= 4:
            label = parts[1]
            text = parts[3]
            if label not in label2id:
                label2id[label] = len(label2id)
            texts.append(text)
            labels.append(label2id[label])

cls_dataset = Dataset.from_dict({"text": texts, "label": labels})

# 先拆分一次大测试集，后续训练和验证拆分用的是“train”和“test”键
cls_dataset = cls_dataset.train_test_split(test_size=0.1, seed=42)
train_val_dataset = cls_dataset["train"]
test_dataset = cls_dataset["test"]

# 再拆分训练集为训练集和验证集
train_val_split = train_val_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

# 定义tokenize函数，保持label不丢失
def tokenize_and_keep_label(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["label"] = examples["label"]
    return tokens

train_dataset = train_dataset.map(tokenize_and_keep_label, batched=True)
val_dataset = val_dataset.map(tokenize_and_keep_label, batched=True)
test_dataset = test_dataset.map(tokenize_and_keep_label, batched=True)

# 设置格式，确保label字段存在且为torch tensor
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 加载MLM任务数据
mlm_path = "E:/natrual-language-processing/nlp数据集/数据集/wikipedia-cn-20230720-filtered-train.arrow"
mlm_dataset = Dataset.from_file(mlm_path)
mlm_dataset = mlm_dataset.train_test_split(test_size=0.1, seed=42)["test"]

tokenized_mlm = mlm_dataset.map(
    lambda e: tokenizer(e["completion"], truncation=True, padding="max_length", max_length=128),
    batched=True,
    remove_columns=mlm_dataset.column_names
)
tokenized_mlm.set_format(type='torch')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 自定义分类任务collate_fn，确保label被保留且批量tensor化
def collate_fn_with_label(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }

# DataLoader
cls_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_with_label)
mlm_dataloader = DataLoader(tokenized_mlm, batch_size=8, shuffle=True, collate_fn=data_collator)

# 初始化模型和优化器
num_labels = len(label2id)
model = BertForMultiTask("E:/natrual-language-processing/nlp数据集/bert/saved-mlm-model/final", num_labels).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=1000)

# 训练
epochs = 3
lambda_mlm = 0.1#联合任务的损失比例
model.train()

cls_iter = iter(cls_dataloader)
mlm_iter = iter(mlm_dataloader)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for step in range(2000):
        try:
            batch_cls = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_dataloader)
            batch_cls = next(cls_iter)

        try:
            batch_mlm = next(mlm_iter)
        except StopIteration:
            mlm_iter = iter(mlm_dataloader)
            batch_mlm = next(mlm_iter)

        # 分类任务前向
        input_ids_cls = batch_cls["input_ids"].to(device)
        attn_mask_cls = batch_cls["attention_mask"].to(device)
        labels_cls = batch_cls["label"].to(device)
        loss_cls, _ = model(input_ids_cls, attn_mask_cls, labels=labels_cls, task_type="classification")

        # MLM任务前向
        input_ids_mlm = batch_mlm["input_ids"].to(device)
        attn_mask_mlm = batch_mlm["attention_mask"].to(device)
        labels_mlm = batch_mlm["labels"].to(device)
        loss_mlm, _ = model(input_ids_mlm, attn_mask_mlm, labels=labels_mlm, task_type="mlm")

        # 总损失
        total_loss = loss_cls*(1-lambda_mlm) + lambda_mlm * loss_mlm
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Step {step} | Loss_cls: {loss_cls.item():.4f} | Loss_mlm: {loss_mlm.item():.4f} | Total: {total_loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "./bert_multitask.pth")
print("\n✅ 模型已保存为 bert_multitask.pth")
