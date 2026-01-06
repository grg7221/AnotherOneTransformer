import torch
from tiktoken import get_encoding
import math
import csv

tk = get_encoding('gpt2')

def get_metrics(state_dict: dict):
    step = state_dict["step"]

    if step % 100 == 0:
        loss = state_dict["loss"]
        ema_loss = state_dict["ema_loss"]
        model = state_dict["model"]
        t = state_dict["t"]
        lr = state_dict["lr"]
        ppl = math.exp(loss)
        grad_norm = get_grad_norm(model)

        print('-----------------------------')
        print(f"Step: {step}")
        print(f"Step time: {t:.2f}s")
        print(f"Loss: {loss:.3f}")
        print(f"EMA Loss: {ema_loss:.3f}")
        print(f"Perplexity: {ppl:.3f}")
        print(f"Gradient norm: {grad_norm:.3f}")
        print(f"Learning rate: {lr:.1e}")

        with open('train_metrics.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, loss, ema_loss, lr, ppl, grad_norm])

    if step % 1000 == 0 and step > 0:
        prompt = torch.tensor([[50256]], dtype=torch.long).cuda()
        out = model.generate(prompt, max_new_tokens=50).tolist()
        print(tk.decode(out[0]))
        model.train()

    # Сохраняем промежуточное состояние каждые 5000 шагов
    if step % 5000 == 0 and step > 0:
        optimizer = state_dict["optimizer"]
        scaler = state_dict["scaler"]
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }, f"E:/AOT/checkpoint.pt")

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm