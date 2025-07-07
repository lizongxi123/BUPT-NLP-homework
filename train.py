import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from typing import Tuple
from tqdm import tqdm
import torch.nn.functional as F
import random
import json


# -------- 数据处理部分 --------
class WikiDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset, tokenizer: BertTokenizer, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['completion']
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }


def get_dataloader(arrow_path: str, batch_size: int = 8, sample_size: float = 0.1, seed: int = 42) -> Tuple[DataLoader, BertTokenizer]:
    tokenizer = BertTokenizer.from_pretrained("E:/natrual-language-processing/nlp数据集/bert-base-chinese")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = HFDataset.from_file(arrow_path)

    if sample_size < 1.0:
        total_len = len(dataset)
        random.seed(seed)
        selected_indices = random.sample(range(total_len), int(total_len * sample_size))
        dataset = dataset.select(selected_indices)

    train_set = WikiDataset(dataset, tokenizer)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), tokenizer


# -------- 模型定义部分 --------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        residual = x
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        x = self.norm2(x + residual)
        return x


class Generator(nn.Module):
    def __init__(self, vocab_size=21128, hidden_dim=1024, num_layers=10, num_heads=8, max_length=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, None)
        x = self.norm(x)
        x = x.transpose(0, 1)
        logits = self.linear(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self, vocab_size=21128, hidden_dim=512, num_layers=2, num_heads=8, max_length=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, None)
        x = self.norm(x)
        x = x.mean(dim=0)
        out = self.classifier(x)
        return out


# -------- 辅助函数 --------
def sample_from_logits(logits):
    return torch.argmax(logits, dim=-1)


# -------- 训练函数 --------
def train_gan(dataloader, gen, disc, device, epochs=3, lr=1e-4, save_path='gan_model.pth', tokenizer=None):
    criterion = nn.BCEWithLogitsLoss()
    recon_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optim_g = optim.Adam(gen.parameters(), lr=lr)
    optim_d = optim.Adam(disc.parameters(), lr=lr)

    gen.train()
    disc.train()
    step = 0
    generated_texts = []

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in loop:
            step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # 判别器更新
            optim_d.zero_grad()
            real_out = disc(input_ids, attention_mask)
            loss_real = criterion(real_out, real_labels)

            fake_logits = gen(input_ids, attention_mask)
            fake_ids = sample_from_logits(fake_logits)
            fake_out = disc(fake_ids.detach(), attention_mask)
            loss_fake = criterion(fake_out, fake_labels)

            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optim_d.step()

            # 生成器更新
            optim_g.zero_grad()
            fake_logits = gen(input_ids, attention_mask)
            fake_ids = sample_from_logits(fake_logits)
            pred = disc(fake_ids, attention_mask)
            adv_loss = criterion(pred, real_labels)

            recon_loss = recon_criterion(fake_logits.view(-1, fake_logits.size(-1)), input_ids.view(-1))
            loss_g = adv_loss + recon_loss
            loss_g.backward()
            optim_g.step()

            # 记录生成文本
            if tokenizer:
                decoded = tokenizer.batch_decode(fake_ids, skip_special_tokens=True)
                generated_texts.extend(decoded)

            loop.set_postfix(loss_d=loss_d.item(), loss_g=loss_g.item())

    torch.save({
        'generator_state_dict': gen.state_dict(),
        'discriminator_state_dict': disc.state_dict(),
        'tokenizer': tokenizer
    }, save_path)
    print(f"Model saved to {save_path}")

    print(f"\n共生成 {len(generated_texts)} 条文本")
    with open('generated_output.json', 'w', encoding='utf-8') as f:
        json.dump(generated_texts, f, ensure_ascii=False, indent=2)


# -------- 主函数 --------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, tokenizer = get_dataloader(
        "E:/natrual-language-processing/nlp数据集/数据集/wikipedia-cn-20230720-filtered-train.arrow",
        batch_size=4, sample_size=0.005, seed=42
    )
    vocab_size = tokenizer.vocab_size
    gen = Generator(vocab_size=vocab_size).to(device)
    disc = Discriminator(vocab_size=vocab_size).to(device)
    train_gan(
        dataloader, gen, disc, device,
        epochs=3,
        lr=1e-4,
        save_path='gan_model.pth',
        tokenizer=tokenizer
    )
