from datasets import load_dataset
from tiktoken import get_encoding
import numpy as np
from tqdm import tqdm

CPU_cores = 6
tk = get_encoding('gpt2') 
ds = load_dataset('parquet', data_files="E:/datasets/Wikipedia/*.parquet", cache_dir="E:/datasets/cache") 

def tokenize(batch):
    return {
        "tokens": [tk.encode(x) + [50256] for x in batch['text']] # 50256 - EOS
    }

def main():
    ds_tokens = ds.map(tokenize, batched=True, num_proc=CPU_cores, remove_columns=['text', 'id', 'url', 'title'])
    #ds_tokens.save_to_disk('E:/datasets/tokens')
    print("Токенизация завершена")

    bin_file = 'E:/datasets/AOT/dataset.bin'

    total_tokens = sum(len(document) for document in tqdm(ds_tokens['train']['tokens'], desc="Подсчет total_tokens"))

    ds_memmap = np.memmap(bin_file, np.uint16, 'w+', shape=(total_tokens, ))

    idx = 0
    for document in tqdm(ds_tokens['train']['tokens'], desc="Сохраняем в .bin"):
        ds_memmap[idx: idx + len(document)] = document
        idx += len(document)

    ds_memmap.flush()
    print(f"Токены сохранены в {bin_file} через memmap")

if __name__ == "__main__":
    main()