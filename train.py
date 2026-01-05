from transformer import Transformer
from tiktoken import get_encoding
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch
import numpy as np
import time
import math

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


if CONTINUE:
    ckpt = torch.load("E:/AOT/checkpoint.pt")
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_step = ckpt['step']

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

        # Выводим лосс каждые 100 шагов
        if step % 100 == 0:
            print('-----------------------------')
            print(f"Step {step}: loss={loss.item():.3f}")
            print(f"Step time: {(t1-t0):.2f}s")
            print(f"Learning rate: {scheduler.get_last_lr()[0]}")

        # Генерируем ответ каждые 500 шагов
        if step % 1000 == 0 and step > 0:
            prompt = torch.tensor([[50256]], dtype=torch.long).cuda()
            out = model.generate(prompt, max_new_tokens=50).tolist()
            print(tk.decode(out[0]))
            model.train()

        # Сохраняем промежуточное состояние каждые 5000 шагов
        if step % 5000 == 0 and step > 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }, f"E:/AOT/checkpoint.pt")

if __name__ == "__main__":
    main()