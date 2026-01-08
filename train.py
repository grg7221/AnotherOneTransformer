import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from transformer import Transformer
import metrics
from tiktoken import get_encoding
import numpy as np
import time
import math
import csv

tk = get_encoding('gpt2')

# -------- ПАРАМЕТРЫ ---------
batch_size= 48
d_model = 512
seq_len = 192
n_layers = 8
n_heads = 8
vocab_size = tk.n_vocab
lr_max = 8e-5

training_steps = 50000
warmup_steps = training_steps * 0.05
dataset_path = 'E:/datasets/AOT/dataset.bin'

CONTINUE = False

# --------- ДАТАСЕТ ----------
class Dataset:
    def __init__(self, memmap_path, seq_len):
        self.data = np.memmap(memmap_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, batch_size):
        x_batch = []
        y_batch = []

        for _ in range(batch_size):
            pos = np.random.randint(0, len(self) - self.seq_len - 1)
            x = self.data[pos : pos + self.seq_len]
            y = self.data[pos+1 : pos + self.seq_len + 1]
            x_batch.append(x)
            y_batch.append(y)

        x_batch = torch.from_numpy(np.array(x_batch, dtype=np.uint16)).long()
        y_batch = torch.from_numpy(np.array(y_batch, dtype=np.uint16)).long()

        return x_batch, y_batch

# ---- LEARNING RATE SCHEDULER ----
def get_lr(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (training_steps - warmup_steps)
        return 0.5 * (1 + math.cos(torch.pi * progress))


# ------ ИНИЦИАЛИЗАЦИЯ ------
model = Transformer(vocab_size, d_model, n_layers, n_heads).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.1)
scheduler = LambdaLR(optimizer, get_lr)
scaler = torch.amp.GradScaler()
start_step = 1
ema_loss = 0

if CONTINUE:
    ckpt = torch.load("E:/AOT/checkpoint.pt")
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_step = ckpt['step']

if not CONTINUE:
    with open('train_metrics.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'ema_loss', 'lr', 'ppl', 'grad_norm'])

# --------- ОБУЧЕНИЕ ----------
def main():
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    ds = Dataset(dataset_path, seq_len)
    print(f"Total tokens in dataset: {len(ds)}")
    model.train()
    
    for step in range(start_step, training_steps+1):
        t0 = time.time()

        x, y = ds.get_batch(batch_size)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)

            y = y.reshape(B*T)

            loss = F.cross_entropy(logits, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        t1 = time.time()

        if step == 1:
            ema_loss = loss.item()
        else:
            ema_loss = 0.01 * loss.item() + (1 - 0.01) * ema_loss

        metrics.get_metrics(
        {
            "step": step,
            "model": model,
            "t": t1-t0,
            "lr": scheduler.get_last_lr()[0],
            "loss": loss.item(),
            "ema_loss": ema_loss,
            "optimizer": optimizer,
            "scaler": scaler
        })


if __name__ == "__main__":
    main()