import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'train_metrics.csv' 
save_plots = True              
output_dir = 'plots/'

df = pd.read_csv(csv_path) # таблица
print(df.head())

# Loss
plt.figure(figsize=(10,6))

plt.plot(df['step'], df['loss'], label='Raw Loss', alpha=0.4)
plt.plot(df['step'], df['ema_loss'], label='EMA Loss', linewidth=2)

plt.xlabel('Step')
plt.ylabel('Loss')

plt.title('Loss (Raw & EMA)')
plt.legend()
plt.grid(True)

if save_plots:
    plt.savefig(f'{output_dir}loss.png', dpi=150)
plt.show()

# Perplexity
plt.figure(figsize=(10,6))

plt.plot(df['step'], df['ppl'], label='Perplexity', color='tab:orange')

plt.yscale('log')  # логарифмическая шкала
plt.xlabel('Step')
plt.ylabel('Perplexity')

plt.title('Perplexity')
plt.grid(True, which='both', ls='--', lw=0.5)

if save_plots:
    plt.savefig(f'{output_dir}perplexity.png', dpi=150)
plt.show()

# LR vs EMA Loss
fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:blue'
ax1.set_xlabel('Step')
ax1.set_ylabel('EMA Loss', color=color)

ax1.plot(df['step'], df['ema_loss'], color=color, linewidth=2, label='EMA Loss')

ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Learning Rate', color=color)

ax2.plot(df['step'], df['lr'], color=color, linewidth=1, label='LR')

ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('EMA Loss and Learning Rate')

if save_plots:
    plt.savefig(f'{output_dir}loss_lr.png', dpi=150)
plt.show()

# Gradient Norm
plt.figure(figsize=(10,6))

plt.plot(df['step'], df['grad_norm'], label='Gradient Norm', color='tab:green')

plt.xlabel('Step')
plt.ylabel('Grad Norm')

plt.title('Gradient Norm')
plt.grid(True)

if save_plots:
    plt.savefig(f'{output_dir}grad_norm.png', dpi=150)
plt.show()

# Last steps Loss
zoom_start = 45000
zoom_df = df[df['step'] >= zoom_start]

plt.figure(figsize=(10,6))

plt.plot(zoom_df['step'], zoom_df['loss'], alpha=0.4, label='Raw Loss')
plt.plot(zoom_df['step'], zoom_df['ema_loss'], linewidth=2, label='EMA Loss')

plt.xlabel('Step')
plt.ylabel('Loss')

plt.title(f'Loss (last steps)')
plt.legend()
plt.grid(True)

if save_plots:
    plt.savefig(f'{output_dir}zoom_loss.png', dpi=150)
plt.show()
