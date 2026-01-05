import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rope(q, k, seq_len, dim):
    # половина головы — x, половина — y
    half = dim // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    # позиционный индекс: [T]
    pos = torch.arange(seq_len, device=q.device)

    # частоты (j = 0...half-1)
    freqs = 1.0 / (10000 ** (torch.arange(half, device=q.device) / half))

    # угол для каждой позиции и каждой частоты → [T, half]
    angles = pos[:, None] * freqs[None, :]

    # синус и косинус
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)   # [1,1,T,half]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)   # [1,1,T,half]

    # поворот для Q
    q_rotated = torch.cat([q1 * cos - q2 * sin,
                           q1 * sin + q2 * cos], dim=-1)

    # поворот для K
    k_rotated = torch.cat([k1 * cos - k2 * sin,
                           k1 * sin + k2 * cos], dim=-1)

    return q_rotated, k_rotated

class MHAFlash(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K = apply_rope(Q, K, T, self.d_head)

        # Не удалось подружить GradScaler + Flash Attention без прямого каста активаций к fp16 (плохая практика)
        # Гипотеза: 
        # Веса в fp32 (условие GradScaler) + PreLN (fp32 must-have) -> x (fp32) + Q_weight (fp32) -> Q (fp32) -/> FA

        #assert Q.is_cuda
        #assert Q.dtype in (torch.float16, torch.bfloat16), Q.dtype

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)

class TransformerBlock(nn.Module): 
    def __init__(self, d_model, n_heads, mlp_ratio, dropout_rate):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.Attention = MHAFlash(d_model, n_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model*mlp_ratio, d_model),
            nn.Dropout(dropout_rate)
        )

        self.LayerDropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # att
        x = x + self.LayerDropout(self.Attention(self.ln1(x)))
        # MLP
        x = x + self.LayerDropout(self.MLP(self.ln2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=4, dropout_rate=0.1) for _ in range(n_layers)
        ])

        self.lnOut = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        x = self.tok_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.lnOut(x)
        logits = self.head(x)

        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx) # (1, T, vocab)
            
            logits_last = logits[:, -1, :]  # (1, vocab)
            
            probs = F.softmax(logits_last, dim=-1)  # (1, vocab)

            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
